"""The data-parallel layout of the run: which group a collective belongs to.

A run with model-parallel axes (``parallel.mesh``) has ranks that hold the SAME data
shard. Any collective that reduces over data -- a metric, a sample count, an
all-gather of features -- must run over the data-parallel group only, or a sample is
counted once per model-parallel rank. Any collective that must agree across every
rank stays global.

The state lives here, in ``core``, so that metrics and models can read the group
without importing ``infrastructure`` (non-negotiable 5).
``spectramr.infrastructure.distributed.mesh`` builds the ``DeviceMesh`` and is the only
caller of :func:`activate_dp_layout`. With no layout active every helper returns the
global answer, which is exactly what a trivial mesh must reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "DPLayout",
    "activate_dp_layout",
    "current_dp_layout",
    "dp_group",
    "dp_rank",
    "dp_src_rank",
    "dp_world_size",
    "is_dp_writer",
    "reset_dp_layout",
]


@dataclass(frozen=True)
class DPLayout:
    """This rank's place in the data-parallel dimension of the mesh."""

    #: Index of this rank within its data-parallel group.
    rank: int
    #: Size of the data-parallel group.
    world_size: int
    #: Process group for data-parallel collectives. ``None`` means the default
    #: (world) group, which is what a trivial mesh uses.
    group: Any | None
    #: Global rank of data-parallel index 0 in this rank's group (a broadcast source).
    src_rank: int
    #: True on the one rank per model-parallel group whose data-parallel results are
    #: the ones that count (rank 0 of every non-dp coordinate).
    writer: bool
    dim_names: tuple[str, ...]
    shape: tuple[int, ...]


_ACTIVE: DPLayout | None = None


def activate_dp_layout(layout: DPLayout) -> None:
    """Install the run's layout. Called once, by the mesh builder, on every rank."""
    global _ACTIVE
    _ACTIVE = layout


def reset_dp_layout() -> None:
    """Forget the layout (tests, and a process that builds a second run)."""
    global _ACTIVE
    _ACTIVE = None


def current_dp_layout() -> DPLayout | None:
    return _ACTIVE


def _global_rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _global_world_size() -> int:
    import torch.distributed as dist

    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def dp_rank() -> int:
    """This rank's index among its data-parallel peers (the global rank when trivial)."""
    layout = _ACTIVE
    return layout.rank if layout is not None else _global_rank()


def dp_world_size() -> int:
    """The data-parallel group size (the world size when trivial)."""
    layout = _ACTIVE
    return layout.world_size if layout is not None else _global_world_size()


def dp_group() -> Any | None:
    """The process group for data-parallel collectives; ``None`` = the default group."""
    layout = _ACTIVE
    return layout.group if layout is not None else None


def dp_src_rank() -> int:
    """Global rank that broadcasts within this rank's data-parallel group."""
    layout = _ACTIVE
    return layout.src_rank if layout is not None else 0


def is_dp_writer() -> bool:
    """True on the rank whose data-parallel results are the ones to keep.

    Always True when no layout is active, so a single-process run is its own writer.
    """
    layout = _ACTIVE
    return layout.writer if layout is not None else True
