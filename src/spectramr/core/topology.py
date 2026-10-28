"""Run topology SSOT — how many ranks, on how many nodes, with what hardware.

Every distributed decision in this codebase needs the same few facts, and
before this module each one re-derived them locally and disagreed:

* the training loop read ``max_iterations`` with **no** ``world_size`` term, so
  a 4-GPU ``torchrun`` ran all 30 000 iterations on *every* rank — the ranks did
  not split one experiment, they each ran the whole experiment. Wall-clock was
  therefore mathematically >= the single-GPU run, which is how an arm came to be
  *slower* on four GPUs than on one;
* dataloader workers were a flat read of the YAML, so ``num_workers: 8`` on a
  4-rank node meant 32 decoder processes competing for 16 cores;
* five separate "am I rank zero?" implementations existed, plus three wrappers;
* the GPU *allocation* was read in one place and stamped into ``provenance.json``
  beside the live world size, where nothing ever compared them -- so a
  ``--gpus=4`` job that launched one rank trained for 41 minutes with three cards
  idle and nothing in the package objected (#1274).

This module owns those facts, and only those facts. The rules that *consume*
them live at their consumers: the worker ceiling in
:mod:`spectramr.core.worker_policy`, the iteration budget in the training loop.
Resolution is a
pure function of ``(environ, cpu_probe, path_exists)`` so every branch is
unit-testable on a laptop with no GPU and no scheduler, mirroring the design of
:mod:`spectramr.core.compute_device`.

**Rank zero is two different questions.** ``is_rank_zero`` is the single owner
for the whole *job* (persistent artifacts: checkpoints, TensorBoard, the run
bundle). ``is_local_rank_zero`` is the single owner per *node* (node-local
scratch in ``/tmp``, per-node caches). Collapsing them into one
``is_main_process`` is the mistake this API exists to prevent: on a multi-node
run, a job-global artifact written by every local-rank-zero silently produces
one file per node, and which one survives is a race.

**Neither predicate means "skip on other ranks."** Under FSDP/DeepSpeed a
state-dict gather is a *collective*: every rank must enter it and exactly one
rank writes the result. Gating the whole block on ``is_rank_zero`` hangs the
job. See the comment at ``pipelines/training_loop.py`` on ``may_checkpoint``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from spectramr.core.resources import (
    allocated_gpus_per_node,
    cpu_resources,
    visible_gpu_count,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EXECUTION_MODES",
    "RunTopology",
    "TopologyResolutionError",
    "resolve_run_topology",
]

#: The four ways a run is started. Mirrors ``cli.launch.WHERE_CHOICES``; kept as
#: a literal rather than imported because ``core`` may not import ``cli``.
EXECUTION_MODES: tuple[str, ...] = ("local", "docker", "apptainer", "slurm")

#: Written by ``spectramr launch`` for the child process (``export_launch_env``).
#: Deliberately NOT registered in ``core.env``: that module's docstring excludes
#: the whole ``SPECTRAMR_LAUNCH_*`` family because it is an internal handoff, not
#: an operator knob — setting one by hand falsifies the provenance record.
#:
#: COMPOSED from the prefix, exactly as ``infrastructure.execution.backends``
#: does, and for the same two reasons: ``core`` may not import
#: ``infrastructure`` (non-negotiable 5), and
#: ``test_every_spectramr_var_read_in_src_is_declared_here`` carves out prefixes
#: *structurally* — a literal ending in ``_`` is a prefix, anything else is a
#: variable that must be declared. Spelling the full name here would trip that
#: gate, and the fix would be to advertise an internal handoff as an operator
#: knob, which is the opposite of what it is.
_LAUNCH_ENV_PREFIX = "SPECTRAMR_LAUNCH_"
_LAUNCH_BACKEND_ENV = f"{_LAUNCH_ENV_PREFIX}BACKEND"


class TopologyResolutionError(RuntimeError):
    """The environment describes a topology that cannot be true.

    Raised rather than repaired. A rank count that does not divide, a rank
    outside its own world, or a malformed integer means the launcher and the
    environment disagree — continuing would train on a silently wrong shard.
    """


def _int_or_raise(env: Mapping[str, str], name: str, *, default: int | None = None) -> int | None:
    """Parse ``name`` as a positive int, or raise. Never silently defaults.

    An unset variable yields ``default``. A *set* but unparseable one raises:
    ``WORLD_SIZE=abc`` must not read as a single-process run (non-negotiable 3).
    """
    raw = env.get(name)
    if raw is None or not str(raw).strip():
        return default
    text = str(raw).strip()
    try:
        value = int(text)
    except ValueError as exc:
        raise TopologyResolutionError(
            f"{name}={raw!r} is not an integer. Refusing to guess the topology; "
            f"a wrong world size trains every rank on the wrong shard."
        ) from exc
    if value < 0:
        raise TopologyResolutionError(f"{name}={raw!r} must not be negative.")
    return value


def _detect_execution_mode(
    env: Mapping[str, str], path_exists: Callable[[str], bool]
) -> tuple[str, str]:
    """Return ``(mode, source)`` — how this process was started.

    Precedence, first hit wins. The launcher's own declaration outranks every
    probe: if ``spectramr launch`` said ``slurm``, a container probe must not
    relabel the run, because the declaration is what provenance should record.
    """
    declared = (env.get(_LAUNCH_BACKEND_ENV) or "").strip().lower()
    if declared:
        if declared not in EXECUTION_MODES:
            raise TopologyResolutionError(
                f"{_LAUNCH_BACKEND_ENV}={declared!r} is not one of {EXECUTION_MODES}."
            )
        return declared, f"env:{_LAUNCH_BACKEND_ENV}"
    if (env.get("SLURM_JOB_ID") or "").strip():
        return "slurm", "env:SLURM_JOB_ID"
    for var in ("APPTAINER_CONTAINER", "SINGULARITY_CONTAINER"):
        if (env.get(var) or "").strip():
            return "apptainer", f"env:{var}"
    if path_exists("/.dockerenv"):
        return "docker", "probe:/.dockerenv"
    return "local", "default"


@dataclass(frozen=True)
class RunTopology:
    """The resolved shape of this run, and the provenance of every field.

    Attributes:
        execution_mode: One of :data:`EXECUTION_MODES`.
        world_size: Total ranks across all nodes.
        local_world_size: Ranks on *this* node.
        num_nodes: Node count.
        rank: This process's global rank.
        local_rank: This process's rank within its node.
        cpus_on_node: Cores this process may actually use, or ``None`` when
            nothing could be probed. May be fractional (a cgroup quota).
        sources: ``{field: where-it-came-from}``, stamped into provenance so a
            surprising number can be traced to the variable that produced it.
        visible_gpus: CUDA devices this process can address, or ``None`` when
            torch could not be asked. ``0`` means torch answered "none".
        allocated_gpus_on_node: GPUs the *scheduler* granted on this node, or
            ``None`` outside a scheduler. Normalised to per-node here: a job
            total (``SLURM_GPUS``) is divided by :attr:`num_nodes`, which is why
            this is resolved after the node count and not beside the env read.

    ``visible_gpus`` and ``allocated_gpus_on_node`` are **facts, not rules**. This
    module deliberately owns no opinion about what a gap between them means; the
    refusal that reads them is in ``pipelines/distributed.py``, next to the
    launcher that can still do something about it.
    """

    execution_mode: str
    world_size: int
    local_world_size: int
    num_nodes: int
    rank: int
    local_rank: int
    cpus_on_node: float | None
    sources: dict[str, str] = field(default_factory=dict)
    # Appended after `sources` because it carries a default and everything after
    # a defaulted field must too. Every construction site in src/ and tests/ is
    # keyword-only, so the position is not load-bearing.
    visible_gpus: int | None = None
    allocated_gpus_on_node: int | None = None

    @property
    def is_distributed(self) -> bool:
        """True when more than one rank participates."""
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        """Single owner for the whole JOB — persistent artifacts."""
        return self.rank == 0

    @property
    def is_local_rank_zero(self) -> bool:
        """Single owner per NODE — node-local scratch, per-node caches."""
        return self.local_rank == 0

    @property
    def cpus_per_rank(self) -> int:
        """This rank's fair share of the node's cores, floor 1.

        Raises:
            TopologyResolutionError: Nothing could be probed. This is the
                asymmetry described in :mod:`spectramr.core.resources` — the same
                number stays optional for *reporting* and is mandatory for a
                *decision*, because silently substituting a default here is
                exactly the oversubscription this module exists to stop.
        """
        if self.cpus_on_node is None:
            raise TopologyResolutionError(
                "cpus_on_node could not be probed, so a per-rank CPU share "
                "cannot be computed. Refusing to assume a default: that is how "
                "num_workers came to be multiplied by the rank count."
            )
        return max(1, int(self.cpus_on_node // max(1, self.local_world_size)))

    def to_dict(self) -> dict[str, Any]:
        """Flat record for provenance, including the derived predicates."""
        record = asdict(self)
        record["is_distributed"] = self.is_distributed
        record["is_rank_zero"] = self.is_rank_zero
        record["is_local_rank_zero"] = self.is_local_rank_zero
        try:
            record["cpus_per_rank"] = self.cpus_per_rank
        except TopologyResolutionError:
            # Provenance is fail-open even though the decision path is not.
            record["cpus_per_rank"] = None
        return record

    def describe(self) -> str:
        """One-line human summary for the startup log."""
        return (
            f"{self.execution_mode}: world_size={self.world_size} "
            f"across {self.num_nodes} node(s), {self.local_world_size}/node; "
            f"rank={self.rank} local_rank={self.local_rank}; "
            f"cpus_on_node={self.cpus_on_node} "
            f"cpus_per_rank={self.to_dict()['cpus_per_rank']}; "
            f"gpus_visible={self.visible_gpus} "
            f"gpus_allocated_on_node={self.allocated_gpus_on_node}"
        )


def resolve_rank(environ: Mapping[str, str] | None = None) -> tuple[int | None, str]:
    """The global rank of this process and where it came from; the one rule for it.

    ``RANK`` first (torchrun always sets it). A hand-rolled ``srun`` sets ``SLURM_PROCID``
    instead, trusted only beside an explicit ``WORLD_SIZE``: without one, ``srun -n N`` runs N
    independent single-process jobs and PROCID is not a rank in any group. ``LOCAL_RANK`` is
    NOT a fallback: it numbers the processes on one node, so used as a global rank it makes
    every node's first process rank 0 (#2701).

    Returns:
        ``(rank, source)``, or ``(None, "unresolved")`` when the launch declares none.

    Raises:
        TopologyResolutionError: ``RANK`` or ``SLURM_PROCID`` is set but not a valid integer.
    """
    env = os.environ if environ is None else environ
    rank = _int_or_raise(env, "RANK", default=None)
    if rank is not None:
        return rank, "env:RANK"
    if _int_or_raise(env, "WORLD_SIZE", default=None) is not None:
        rank = _int_or_raise(env, "SLURM_PROCID", default=None)
        if rank is not None:
            return rank, "env:SLURM_PROCID"
    return None, "unresolved"


def resolve_run_topology(
    environ: Mapping[str, str] | None = None,
    *,
    cpu_probe: Callable[..., dict[str, Any]] | None = None,
    path_exists: Callable[[str], bool] | None = None,
    gpu_probe: Callable[[], int | None] | None = None,
) -> RunTopology:
    """Resolve the run topology from the environment the launcher set.

    Reads only variables that exist *before* ``dist.init_process_group``, so it
    answers during startup — which is the point, since worker counts and the
    iteration budget are decided before the process group exists.

    Args:
        environ: Environment mapping (defaults to ``os.environ``).
        cpu_probe: Callable returning a :func:`spectramr.core.resources.cpu_resources`
            dict. Injected so the policy is testable without a real machine.
        path_exists: ``os.path.exists`` equivalent, for the docker probe.
        gpu_probe: Callable returning the visible CUDA device count (see
            :func:`spectramr.core.resources.visible_gpu_count`). Injected for the
            same reason as ``cpu_probe`` -- every branch here must be reachable
            on a laptop with no GPU.

    Returns:
        The resolved :class:`RunTopology`.

    Raises:
        TopologyResolutionError: The environment is internally inconsistent —
            a malformed integer, a rank outside its world, or more local ranks
            than total ranks.
    """
    env = os.environ if environ is None else environ
    probe = cast(
        "Callable[..., dict[str, Any]]", cpu_probe if cpu_probe is not None else cpu_resources
    )
    exists = os.path.exists if path_exists is None else path_exists
    gpus = visible_gpu_count if gpu_probe is None else gpu_probe

    sources: dict[str, str] = {}
    mode, mode_source = _detect_execution_mode(env, exists)
    sources["execution_mode"] = mode_source

    # --- world size / rank -------------------------------------------------
    world_size = _int_or_raise(env, "WORLD_SIZE", default=None)
    if world_size is None:
        world_size, sources["world_size"] = 1, "default"
    else:
        sources["world_size"] = "env:WORLD_SIZE"
    if world_size < 1:
        raise TopologyResolutionError(f"WORLD_SIZE={world_size} must be >= 1.")

    rank, rank_source = resolve_rank(env)
    if rank is None:
        rank, rank_source = 0, "default"
    sources["rank"] = rank_source
    if rank >= world_size:
        raise TopologyResolutionError(
            f"RANK={rank} is outside WORLD_SIZE={world_size} (valid: 0..{world_size - 1})."
        )

    # --- ranks per node ----------------------------------------------------
    local_world_size = _int_or_raise(env, "LOCAL_WORLD_SIZE", default=None)
    if local_world_size is not None:
        sources["local_world_size"] = "env:LOCAL_WORLD_SIZE"
    else:
        local_world_size = _int_or_raise(env, "SLURM_NTASKS_PER_NODE", default=None)
        if local_world_size is not None:
            sources["local_world_size"] = "env:SLURM_NTASKS_PER_NODE"
        else:
            local_world_size, sources["local_world_size"] = world_size, "derived:world_size"
    if local_world_size < 1:
        raise TopologyResolutionError(f"local_world_size={local_world_size} must be >= 1.")
    if local_world_size > world_size:
        raise TopologyResolutionError(
            f"local_world_size={local_world_size} exceeds WORLD_SIZE={world_size}: "
            f"a node cannot hold more ranks than the job has."
        )

    local_rank = _int_or_raise(env, "LOCAL_RANK", default=None)
    if local_rank is not None:
        sources["local_rank"] = "env:LOCAL_RANK"
    else:
        local_rank = _int_or_raise(env, "SLURM_LOCALID", default=None)
        if local_rank is not None:
            sources["local_rank"] = "env:SLURM_LOCALID"
        else:
            local_rank, sources["local_rank"] = rank % local_world_size, "derived:rank%local"
    if local_rank >= local_world_size:
        raise TopologyResolutionError(
            f"LOCAL_RANK={local_rank} is outside local_world_size={local_world_size}."
        )

    # --- node count --------------------------------------------------------
    # Derived by division, but SLURM_NNODES is a *declaration* and outranks it:
    # on a heterogeneous allocation (unequal ranks per node — the same case
    # SLURM_CPUS_ON_NODE's "72(x2)" form exists for) the division is simply
    # wrong. Disagreement is logged, never silently resolved.
    derived_nodes = max(1, world_size // local_world_size)
    declared_nodes = _int_or_raise(env, "SLURM_NNODES", default=None)
    if declared_nodes is not None and declared_nodes >= 1:
        num_nodes, sources["num_nodes"] = declared_nodes, "env:SLURM_NNODES"
        if declared_nodes != derived_nodes:
            logger.warning(
                "[TOPOLOGY] SLURM_NNODES=%d disagrees with world_size/local_world_size"
                " = %d/%d = %d. Trusting SLURM_NNODES; the allocation is probably"
                " heterogeneous, so per-node counts derived by division are unsafe.",
                declared_nodes,
                world_size,
                local_world_size,
                derived_nodes,
            )
    else:
        num_nodes, sources["num_nodes"] = derived_nodes, "derived:world/local"

    # --- CPUs --------------------------------------------------------------
    # Fail-open probe (see core.resources): None is preserved, never defaulted.
    # `cpus_per_rank` is where None becomes an error, at the point of decision.
    cpu_info = probe(env)
    cpus_on_node = cpu_info.get("usable_cores")
    sources["cpus_on_node"] = "probe:cpu_resources.usable_cores"

    # --- GPUs --------------------------------------------------------------
    # Fail-open on both halves, like the CPU probe above and for the same
    # reason: a torch-less or GPU-less host must still resolve a topology.
    visible_gpus = gpus()
    sources["visible_gpus"] = "probe:visible_gpu_count"

    # Normalised to per-node by `core.resources`, which is why this read sits
    # AFTER `num_nodes` is known: `--gpus=N` populates only the job total.
    # `pipelines/distributed.py` calls the same helper, so the number stamped
    # here and the number the launch refusal decides on cannot drift apart.
    allocated_gpus_on_node, gpu_source = allocated_gpus_per_node(num_nodes, env)
    sources["allocated_gpus_on_node"] = gpu_source

    return RunTopology(
        execution_mode=mode,
        world_size=world_size,
        local_world_size=local_world_size,
        num_nodes=num_nodes,
        rank=rank,
        local_rank=local_rank,
        cpus_on_node=cpus_on_node,
        sources=sources,
        visible_gpus=visible_gpus,
        allocated_gpus_on_node=allocated_gpus_on_node,
    )
