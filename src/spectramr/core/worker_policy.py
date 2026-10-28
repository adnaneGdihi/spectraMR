"""Dataloader worker policy — one CPU-share rule, shared by every loader.

``num_workers`` was a flat read of the YAML with no topology term anywhere on
the path, so N ranks on one node each spawned the full declared count:
``num_workers: 8`` on a 4-GPU node with ``--cpus-per-task=16`` meant four
trainers plus **32** decoder processes competing for 16 cores. Step time went up
rather than down -- one of the reasons a multi-GPU run could come out slower than
the single-GPU baseline it was meant to beat.

This module owns the rule; :mod:`spectramr.core.topology` owns the facts it reads.
The separation matters because the rule is a *policy* with a compatibility
contract (see :func:`clamp_worker_count` on why it is a ceiling), while the
topology is an observed fact about the process.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

from spectramr.core.topology import RunTopology

logger = logging.getLogger(__name__)

__all__ = ["WorkerDecision", "clamp_worker_count"]


@dataclass(frozen=True)
class WorkerDecision:
    """A dataloader worker count, and why it is that number (pitfall #15c)."""

    workers: int  # what the loader will actually be built with
    declared: int  # what the YAML asked for
    cpus_per_rank: int | None  # the share this rank may use, when known
    clamped: bool  # True only when `workers < declared`
    reason: str
    reserved_cores: int  # cores withheld from the share for the main process
    persistent: bool = False  # whether the DataLoader keeps its workers alive

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clamp_worker_count(
    declared: int,
    topology: RunTopology,
    *,
    role: str = "train",
    log: bool = True,
    reserved_cores: int,
) -> WorkerDecision:
    """Clamp a declared worker count DOWN to this rank's CPU share.

    The declared value is a **ceiling, never a floor**. This function can only
    ever return a number ``<= declared``, which is what makes it safe to apply
    everywhere: an arm that already fits its allocation is untouched, so the
    ``# OOM fix`` arms that deliberately lowered ``num_workers`` keep the value
    they chose, and a single-rank run whose workers fit its cores minus
    ``reserved_cores`` is unchanged.

    What it fixes: ``num_workers`` was a flat read of the YAML with no topology
    term, so N ranks on one node each spawned the full count — ``num_workers: 8``
    on a 4-GPU node meant 32 decoder processes competing for 16 cores, and the
    step time went up rather than down.

    Args:
        declared: The configured ``num_workers``.
        topology: The resolved run topology.
        role: ``"train"`` / ``"val"`` / ``"inference"``, for the log line.
        log: Emit the clamp / unknown-CPU log lines. ``False`` for the
            provenance re-derivation, which asks the same question a second
            time in order to record the answer and must not narrate it twice.
        reserved_cores: Cores per rank kept back for the main process and its
            helper threads (``data.loader.reserved_cores``); the share is
            ``max(1, cpus_per_rank - reserved_cores)``. Negative raises.

    Returns:
        The :class:`WorkerDecision`. ``workers`` is ``0`` iff ``declared`` is
        ``0`` — a declared ``0`` means "load in the main process", a deliberate
        choice (and ``validation.loader``'s default), so it passes through
        untouched. The floor is otherwise ``1``, which is what makes the
        ``num_workers=0 + persistent_workers=True`` combination that torch
        raises on unreachable by construction.
    """
    if reserved_cores < 0:
        raise ValueError(f"reserved_cores must be >= 0, got {reserved_cores}")
    declared = int(declared or 0)
    if declared <= 0:
        return WorkerDecision(
            workers=0,
            declared=declared,
            cpus_per_rank=None,
            clamped=False,
            reason="declared-serial",
            reserved_cores=reserved_cores,
        )

    if topology.cpus_on_node is None:
        # Deliberately NOT a raise, and deliberately NOT a substituted default.
        # We cannot learn this rank's CPU share, so we decline to *reduce* the
        # declared value and say so loudly. That is different in kind from the
        # silent fallback non-negotiable 3 forbids: nothing is standing in for a
        # value we failed to read, and the outcome is exactly today's behaviour
        # rather than a guess. `RunTopology.cpus_per_rank` still raises for any
        # caller that genuinely needs the number.
        if log:
            logger.warning(
                "[TOPOLOGY] %s num_workers=%d left unclamped: this node's usable "
                "core count could not be probed, so a per-rank share is unknown. "
                "With world_size=%d this may oversubscribe the node.",
                role,
                declared,
                topology.world_size,
            )
        return WorkerDecision(
            workers=declared,
            declared=declared,
            cpus_per_rank=None,
            clamped=False,
            reason="cpus-unknown",
            reserved_cores=reserved_cores,
        )

    share = topology.cpus_per_rank
    # The reserve is per rank: the main process needs its own core whatever the
    # share, and `max(1, ...)` keeps a small allocation from clamping to zero.
    usable = max(1, share - reserved_cores)
    workers = max(1, min(declared, usable))
    if log and workers < declared and topology.is_local_rank_zero:
        # Log once per NODE: the condition is about that node's cores, and every
        # local rank would otherwise print the identical line.
        logger.warning(
            "[TOPOLOGY] %s num_workers %d -> %d: %s usable cores on this node "
            "shared by %d local rank(s) = %d per rank, minus %d reserved for the "
            "main process. The declared value is a ceiling; lower it in the "
            "config to make this permanent.",
            role,
            declared,
            workers,
            topology.cpus_on_node,
            topology.local_world_size,
            share,
            reserved_cores,
        )
    return WorkerDecision(
        workers=workers,
        declared=declared,
        cpus_per_rank=share,
        clamped=workers < declared,
        reason="clamped-to-cpu-share" if workers < declared else "fits-allocation",
        reserved_cores=reserved_cores,
    )
