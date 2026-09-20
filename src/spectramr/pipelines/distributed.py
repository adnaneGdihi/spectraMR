"""Multi-process training entry point (torchrun).

Drives every process-group-backed strategy -- ``ddp``, ``fsdp`` and
``deepspeed`` -- not just DDP. The strategy comes from
``parallel.strategy`` in the YAML and is NOT rewritten here; it used to be forced
to ``"ddp"`` on every distributed launch, which is what made ``fsdp`` and
``deepspeed`` unreachable from this entry point.

``--nproc_per_node`` is checked against the scheduler's GPU grant here, because
this is the only point where the rank count, the visible devices and the
allocation are all in scope -- see :func:`idle_device_refusal`.

Usage::

    torchrun --nproc_per_node=4 -m spectramr.cli train-distributed --config <arm>.yaml
    torchrun --nproc_per_node=4 -m spectramr.pipelines.distributed --config <arm>.yaml
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist

from spectramr.infrastructure.logging.rank_console import quiet_secondary_ranks

logger = logging.getLogger(__name__)


def setup_distributed(backend: str = "nccl") -> tuple[int, int]:
    """Initialize the distributed process group.

    Reads rank/world_size from environment variables set by torchrun.

    Returns:
        (rank, world_size) tuple.

    Raises:
        RuntimeError: If environment variables are not set (i.e. not launched via torchrun).
    """
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))

    if rank == -1 or world_size == -1:
        raise RuntimeError(
            "Distributed environment variables (RANK, WORLD_SIZE) not set. "
            "Launch with: torchrun --nproc_per_node=N ..."
        )

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    # `device_id` BINDS the group to this rank's GPU at init. Without it every
    # device-taking collective infers a device from "the current context",
    # which is what emits
    #     barrier(): using the device under current context. You can specify
    #     `device_id` in `init_process_group` to mute this warning.
    # The warning is the mild symptom. The real one is that `barrier()` with no
    # bound device guesses, and a rank that guesses differently from its peers
    # barriers on the wrong GPU -- a hang or a corrupt collective rather than an
    # error. Binding here also lets NCCL eagerly initialise the communicator
    # instead of lazily on first use.
    #
    # Only for device-backed backends: `gloo` runs on CPU and rejects a CUDA
    # `device_id`, so this must follow the backend rather than `set_device`
    # above (which the caller already scoped to the CUDA path).
    pg_kwargs: dict[str, object] = {}
    if backend.lower() == "nccl" and torch.cuda.is_available():
        pg_kwargs["device_id"] = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size, **pg_kwargs
    )

    if rank == 0:
        logger.info(
            f"[DDP] Initialized: rank={rank}, world_size={world_size}, "
            f"local_rank={local_rank}, backend={backend}"
        )
    else:
        # Suppress verbose logging on non-zero ranks. Idempotent and normally
        # already applied at the entry point (``cli/app.py::main``) -- doing it
        # again here covers ``python -m spectramr.pipelines.distributed``, which
        # bypasses that entry point entirely.
        quiet_secondary_ranks()

    return rank, world_size


class IdleDeviceError(RuntimeError):
    """The scheduler granted GPUs this launch will never touch.

    Raised rather than warned, following ``DataParallelStrategy.adopt``'s
    ``device_count() <= 1`` refusal: that check used to warn and continue, and
    the consequence was arms whose provenance claimed multi-GPU while they ran
    on one. The same reasoning applies here with the sign flipped.
    """


def idle_device_refusal(
    *,
    allocated_on_node: int | None,
    visible: int | None,
    local_world_size: int,
    allow_idle_devices: bool,
) -> str | None:
    """Return the refusal message, or ``None`` when this launch wastes nothing.

    Pure so every branch is reachable without a GPU or a scheduler. The wiring in
    :func:`run_distributed_training` only fetches the four inputs.

    The predicate is deliberately conservative -- it must never fire on a
    correct launch, because a false refusal on a cluster costs a queue slot:

    * **No scheduler grant -> no finding.** A workstation with four idle cards is
      not burning anyone's allocation, and ``ALLOC_GPU_ENV`` is unset there. This
      check is about a grant that was made and not used.
    * **Visible bounds allocated.** ``srun --gpu-bind=single:1`` gives each task
      one device out of the node's four; the other three are *another rank's*,
      not idle. Taking the ``min`` makes that read as 1-of-1 rather than 3 wasted.
    * **Equality passes.** 4 ranks on 4 GPUs is the shape this exists to protect.

    The reverse imbalance -- more ranks than devices -- is NOT decided here. It is
    already reported by ``provenance._rank_device_record``'s collision detector,
    which can see every rank's resolved device and so distinguishes real sharing
    from a per-rank ``CUDA_VISIBLE_DEVICES`` mask. This function would only be
    guessing.

    Args:
        allocated_on_node: GPUs the scheduler granted on this node, or ``None``.
        visible: CUDA devices this process can address, or ``None`` if unprobed.
        local_world_size: Ranks torchrun started on this node.
        allow_idle_devices: The operator's explicit acknowledgement.

    Returns:
        A message naming both counts and the two ways out, or ``None``.
    """
    if allow_idle_devices:
        return None
    if allocated_on_node is None:
        return None
    usable = allocated_on_node if visible is None else min(allocated_on_node, visible)
    if usable <= local_world_size:
        return None
    return (
        f"This node was allocated {allocated_on_node} GPU(s) "
        f"({visible} visible to this process) but torchrun started only "
        f"{local_world_size} rank(s) on it, so {usable - local_world_size} "
        f"allocated GPU(s) will sit idle for the whole run.\n"
        f"This is refused rather than warned because it is otherwise invisible: "
        f"the process group initialises, the strategy adopts, the health report "
        f"passes, and the run trains correctly -- just on a fraction of the "
        f"hardware it is being charged for.\n"
        f"Fix the launcher: torchrun --nproc_per_node={usable} (a hardcoded "
        f"nproc_per_node is the usual cause; derive it from the allocation).\n"
        f"Or, for a deliberate single-rank debug run on a multi-GPU allocation, "
        f"acknowledge it: -O parallel.allow_idle_devices=true"
    )


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("[DDP] Process group destroyed.")


def run_distributed_training(
    config_path: str,
    backend: str | None = None,
    resume_path: str | None = None,
    overrides: list[str] | None = None,
) -> dict:
    """Main distributed training function invoked by each torchrun worker.

    1. Sets up the distributed process group.
    2. Loads config and applies overrides.
    3. Delegates to ``run_training_pipeline()``, which dispatches on the
       DECLARED ``config.parallel.strategy`` (ddp / fsdp / deepspeed).
    4. Cleans up the process group.

    Args:
        config_path: Path to YAML config file.
        backend: Override for ``parallel.backend``; ``None`` means use the
            config value (that is how the YAML wins).
        resume_path: Optional checkpoint path for resume.
        overrides: Optional list of ``KEY=VALUE`` config override strings.
    """
    # Backend is resolved from the config AFTER it loads, so parallel.backend
    # can win. The process group therefore starts below, not here.

    # Clamp non-zero ranks FIRST. ``cli/app.py::main`` already did this for the
    # ``spectramr train-distributed`` path, but ``python -m
    # spectramr.pipelines.distributed`` bypasses it -- and everything between here
    # and ``setup_distributed`` (the config load, ``[Parallel] process-group
    # backend=...``) logs at INFO, so on that path it would still print once per
    # rank. Idempotent; a no-op on a single-process run.
    quiet_secondary_ranks()

    try:
        from spectramr.config.settings import TrainingSettings
        from spectramr.core.execution_ledger import ExecutionLedger
        from spectramr.pipelines.train import run_training_pipeline

        # Arm the ledger BEFORE the config loads, exactly as the single-process
        # entry points in ``main.py`` do. It records the substitutions made
        # during ``from_yaml`` -- so arming it afterwards is not "late", it is
        # empty. Skipping it meant the first consumer hit ``current_or_begin``,
        # which self-armed with a loud warning on every rank ("the ledger in
        # this run's artifacts is incomplete, not empty") and stamped a
        # ``ledger armed late`` note into the run's own provenance.
        #
        # Armed on EVERY rank, not just rank 0: the ledger lives in a
        # ContextVar, each rank is its own interpreter, and the rank is not
        # known here anyway (``setup_distributed`` runs below, because the
        # backend it needs comes from the config this line is about to load).
        ExecutionLedger.begin_run(source=str(config_path))

        # Load config (all ranks load the same config)
        settings = TrainingSettings.from_yaml(config_path)

        # Apply overrides. Import from the config layer (rightward), NOT from
        # ``spectramr.main`` — a pipelines→entry-layer leftward import (CLAUDE.md
        # #13). ``apply_overrides`` now lives in ``config/overrides.py``.
        if overrides:
            from spectramr.config.overrides import apply_overrides

            settings = apply_overrides(settings, overrides)

        # parallel.backend is the SSOT; --backend wins only when explicitly
        # typed (its argparse default is None, which is what makes "the user
        # asked for nccl" distinguishable from "argparse filled it in").
        from spectramr.infrastructure.distributed.backend import (
            resolve_distributed_backend,
        )

        resolved_backend, backend_source = resolve_distributed_backend(settings, backend)
        rank, world_size = setup_distributed(backend=resolved_backend)
        logger.info("[DDP] backend=%s (source=%s)", resolved_backend, backend_source)

        # The launcher no longer rewrites `strategy`.
        #
        # It used to force it to "ddp" whenever a distributed launch was
        # detected. That is a silent config rewrite (non-negotiables #1 and #3),
        # and it is precisely what made `strategy: 'fsdp'` and
        # `strategy: 'deepspeed'` unreachable from `train-distributed`: the
        # declaration was overwritten before dispatch ever saw it.
        #
        # `num_devices`/`num_nodes` ARE overwritten, because those are observed
        # facts about the launcher, not user declarations. `strategy` is a
        # declaration.
        parallel = getattr(settings, "parallel", None)
        declared = parallel.strategy if parallel is not None else "none"
        if declared == "none":
            raise ValueError(
                "train-distributed was launched (RANK/WORLD_SIZE are set) but "
                "parallel.strategy is 'none'. Declare ddp | fsdp | deepspeed in "
                "the YAML -- the launcher no longer silently rewrites it, because "
                "doing so made 'fsdp' and 'deepspeed' unreachable from here."
            )
        if declared == "dp":
            raise ValueError(
                "parallel.strategy='dp' is single-process (DataParallel over N "
                "GPUs); do not launch it with torchrun. Use 'ddp' for one process "
                "per device."
            )

        # WORLD_SIZE is global; LOCAL_WORLD_SIZE is per node. The old code set
        # num_devices=world_size, which is wrong on any multi-node run.
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

        # Refuse a launch that leaves allocated GPUs idle -- BEFORE the
        # model_copy below, which overwrites `num_devices` with the observed
        # rank count and so erases the last trace of the disagreement.
        #
        # This is the only site where all four facts are in scope at once: the
        # rank count torchrun actually started, the devices this process can
        # address, and the scheduler's grant. The launcher-side guard added in
        # PR #1101 lives entirely in `scripts/*.sbatch`, so a hand-rolled
        # `srun` + `torchrun` -- which is how #1274 happened -- never reaches it.
        # Every rank evaluates its own node's facts rather than gating on rank 0,
        # so a heterogeneous allocation is caught on whichever node is wrong;
        # torchrun reaps the group when any rank exits non-zero.
        # The two GPU facts come from `core.resources`, the same implementation
        # `RunTopology` stamps into provenance -- so the number a reader sees in
        # the artifact and the number this refusal was decided on are one number.
        # `resolve_run_topology()` is deliberately NOT called: the rank counts are
        # already in hand here, and re-validating the whole environment would make
        # an unrelated inconsistency fail in a new place.
        from spectramr.core.resources import (
            allocated_gpus_per_node,
            visible_gpu_count,
        )

        _num_nodes = max(1, world_size // max(1, local_world_size))
        _allocated, _alloc_source = allocated_gpus_per_node(_num_nodes)
        _visible = visible_gpu_count()
        # A DIRECT read, not `getattr(..., False)`: `parallel` is provably a
        # ParallelismConfigSchema by here (None and 'none'/'dp' all raised above),
        # so a missing attribute would mean the knob was renamed out from under
        # this call -- and a defaulted read would then silently arm the guard with
        # no way to switch it off, i.e. the documented `-O
        # parallel.allow_idle_devices=true` would stop working and say nothing.
        # Non-negotiable 3: a default must never stand in for a declared value.
        _allow_idle = bool(parallel.allow_idle_devices)

        _refusal = idle_device_refusal(
            allocated_on_node=_allocated,
            visible=_visible,
            local_world_size=local_world_size,
            allow_idle_devices=_allow_idle,
        )
        if _refusal is not None:
            raise IdleDeviceError(f"{_refusal}\n(GPU grant read from {_alloc_source})")
        if _allow_idle and rank == 0:
            # An acknowledged waste is still a waste; say so once, so the artifact
            # records that the check ran and was overridden rather than that it
            # never applied.
            logger.warning(
                "[Parallel] parallel.allow_idle_devices=true: not checking the "
                "%s rank(s) on this node against its GPU grant of %s (%s visible, "
                "source=%s).",
                local_world_size,
                _allocated,
                _visible,
                _alloc_source,
            )

        settings = settings.model_copy(
            update={
                "parallel": parallel.model_copy(
                    update={
                        "num_devices": local_world_size,
                        "num_nodes": max(1, world_size // max(1, local_world_size)),
                    }
                )
            }
        )

        # Determine device for this rank
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = f"cuda:{local_rank}"

        result = run_training_pipeline(
            settings,
            device=device,
            resume_path=resume_path,
        )

        return result

    finally:
        cleanup_distributed()


def main() -> None:
    """CLI entry point for ``python -m spectramr.pipelines.distributed``."""
    parser = argparse.ArgumentParser(description="Distributed training via DDP")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (path, 'auto', or 'if-present')",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["nccl", "gloo", "mpi"],
        help="Override parallel.backend. Default None (NOT 'nccl') so the YAML "
        "value can win when this is not passed.",
    )
    parser.add_argument(
        "--override",
        "-O",
        action="append",
        metavar="KEY=VALUE",
        help="Override config values",
    )

    args = parser.parse_args()

    run_distributed_training(
        config_path=args.config,
        backend=args.backend,
        resume_path=args.resume,
        overrides=args.override,
    )


if __name__ == "__main__":
    main()
