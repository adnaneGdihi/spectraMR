"""The world size and backend a checkpoint was written at, and the resume guard over them."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from spectramr.infrastructure.checkpointing.payload import Topology
from spectramr.infrastructure.distributed.distributed_training import RankUtility

logger = logging.getLogger(__name__)


def topology_record(parallel: Any) -> Topology:
    """The topology this rank is running at; no runtime is the plain single-process path."""
    return Topology(
        world_size=RankUtility.get_world_size(),
        parallel_strategy=getattr(parallel, "strategy", None) or "none",
    )


def check_resume_topology(saved: Any, checkpoint_cfg: Any, parallel: Any, *, source: Path) -> None:
    """Refuse a resume at a different world size or sharded backend unless the arm opted in.

    ``max_iterations`` is per rank and the LR schedule is indexed by iteration, so a world
    size change silently changes what an iteration covers. A record that is not a mapping
    is accepted: unknown is not different.
    """
    if not isinstance(saved, dict) or "world_size" not in saved:
        return
    now = RankUtility.get_world_size()
    # No runtime (or no ``parallel`` block) is the plain, single-process path.
    saved_strategy = saved.get("parallel_strategy") or "none"
    now_strategy = getattr(parallel, "strategy", None) or "none"
    header = (
        f"{source} was written at world_size={saved['world_size']} "
        f"(parallel.strategy={saved_strategy!r}) and this run has world_size={now} "
        f"(parallel.strategy={now_strategy!r})."
    )

    if saved["world_size"] != now:
        if getattr(checkpoint_cfg, "allow_world_size_change", False) is not True:
            raise RuntimeError(
                header + " Resuming would change how many samples one iteration covers, "
                "and the LR schedule, data epoch and shuffle seed would no longer line up "
                "with what was saved. Resume at the original world size, or set "
                "checkpoint.allow_world_size_change: true to accept that."
            )
        logger.warning("%s Resuming anyway: checkpoint.allow_world_size_change is true.", header)

    # none, dp and ddp write interchangeable state; FSDP and DeepSpeed write formats a
    # plain optimizer (and each other) cannot load.
    sharded = {"fsdp", "deepspeed"}
    if saved_strategy != now_strategy and ({saved_strategy, now_strategy} & sharded):
        if getattr(checkpoint_cfg, "allow_parallel_strategy_change", False) is not True:
            raise RuntimeError(
                header + " An FSDP checkpoint carries unsharded optimizer state keyed by "
                "parameter name and a DeepSpeed one carries sharded per-rank state; "
                "neither loads into another backend's optimizer. Resume under the same "
                "parallel.strategy, or set checkpoint.allow_parallel_strategy_change: true "
                "to accept that (weights load; optimizer state may not)."
            )
        logger.warning(
            "%s Resuming anyway: checkpoint.allow_parallel_strategy_change is true.", header
        )


__all__ = ["check_resume_topology", "topology_record"]
