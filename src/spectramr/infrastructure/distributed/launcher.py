"""Distributed launcher (PR-14 / L1) — campaign-aware.

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-14.

Two entry points:

- :func:`launch_distributed` — spawns ``num_devices × num_nodes``
  worker processes via ``torch.multiprocessing.spawn`` (single-node)
  or returns the ``torchrun`` command line (multi-node).
- :func:`dispatch_for_campaign` — invoked by the campaign runner with
  the resolved per-arm parallelism config; chooses single-process,
  single-node multi-GPU, or multi-node based on
  ``num_nodes`` / ``num_devices``.

The launcher is deliberately *generic* — it does not import any
strategy / model class. Callers pass a ``run_fn`` that takes
``(rank, world_size, settings)`` and the launcher handles process-
group setup + clean teardown.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _free_port() -> int:
    """A currently unused TCP port, so two jobs on one node do not share a rendezvous."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_process_group(rank: int, world_size: int, backend: str) -> None:
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _destroy_process_group() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _spawn_worker(
    rank: int,
    world_size: int,
    backend: str,
    run_fn: Callable[..., Any],
    settings: Any,
) -> None:
    try:
        # Select the device BEFORE creating the group, so NCCL binds this rank's GPU
        # rather than whatever the current context happens to be.
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        _init_process_group(rank, world_size, backend)
        run_fn(rank=rank, world_size=world_size, settings=settings)
    finally:
        _destroy_process_group()


def launch_distributed(
    run_fn: Callable[..., Any],
    settings: Any,
    *,
    num_devices: int = 1,
    num_nodes: int = 1,
    backend: str = "nccl",
) -> Any:
    """Launch ``run_fn`` across ``num_devices × num_nodes`` ranks.

    Single-node, multi-GPU: uses ``torch.multiprocessing.spawn``.
    Multi-node: returns the ``torchrun`` command line for the cluster
    job-script to execute (the wrapper does not assume Slurm /
    Kubernetes).

    For ``num_devices=1, num_nodes=1`` the function calls ``run_fn``
    directly in the current process — no overhead.
    """
    world_size = num_devices * num_nodes
    if world_size <= 1:
        logger.info("launch_distributed: world_size=1, running in current process")
        return run_fn(rank=0, world_size=1, settings=settings)

    if num_nodes == 1:
        # Single-node, multi-GPU — spawn workers locally.
        import torch.multiprocessing as mp

        # Workers inherit the environment under spawn. A fixed default port made
        # two concurrent jobs on one node fight over the same rendezvous.
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        mp.spawn(
            _spawn_worker,
            args=(world_size, backend, run_fn, settings),
            nprocs=num_devices,
            join=True,
        )
        return None

    # Multi-node: return the canonical torchrun command — the cluster
    # job script is responsible for launching this on each node.
    #
    # The subcommand is `train-distributed`, NOT `train`. This emitted the
    # latter, which is the one spelling that cannot work: `train` never calls
    # `setup_distributed`, so no process group exists, and every
    # group-requiring strategy then raises out of `_require_process_group`
    # during `adopt` — at Stage B, after the model and data are already built.
    # A user who copied this line got a crash that read like a DeepSpeed/DDP
    # problem rather than a launcher typo. `cli/app.py` registers
    # `train-distributed` and its own help string already documents this exact
    # invocation; the two are pinned together by the paired test.
    nproc = num_devices
    nnodes = num_nodes
    return (
        f"torchrun --nproc_per_node={nproc} --nnodes={nnodes} "
        f"--rdzv_id=${{SLURM_JOB_ID:-spectramr}} "
        f"--rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:${{MASTER_PORT:-29500}} "
        f"-m spectramr.cli train-distributed --config $CONFIG_PATH"
    )


def dispatch_for_campaign(
    run_fn: Callable[..., Any],
    settings: Any,
    *,
    arm_name: str | None = None,
) -> Any:
    """Read ``settings.parallel.{strategy, num_devices, num_nodes}`` and
    launch ``run_fn`` accordingly.

    The campaign runner calls this once per arm; if the arm declares
    DDP / FSDP, the launcher spins up the right topology.
    """
    parallel = getattr(settings, "parallel", None)
    if parallel is None:
        logger.info("[campaign:%s] no parallel block — single-process run", arm_name)
        return run_fn(rank=0, world_size=1, settings=settings)
    strategy = (parallel.strategy or "none").lower()
    num_devices = max(1, getattr(parallel, "num_devices", 1))
    num_nodes = max(1, getattr(parallel, "num_nodes", 1))

    if strategy == "none" or (num_devices == 1 and num_nodes == 1):
        return run_fn(rank=0, world_size=1, settings=settings)

    logger.info(
        "[campaign:%s] launching strategy=%s num_devices=%d num_nodes=%d",
        arm_name,
        strategy,
        num_devices,
        num_nodes,
    )
    return launch_distributed(
        run_fn,
        settings,
        num_devices=num_devices,
        num_nodes=num_nodes,
        backend=parallel.backend or "nccl",
    )


__all__ = ["dispatch_for_campaign", "launch_distributed"]
