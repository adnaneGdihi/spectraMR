"""Halo exchange along depth for spatially split volumes, and its adjoint.

A volume ``[B, C, D, H, W]`` is split along D across the ranks of a spatial group:
rank ``r`` holds planes ``[r*Dl, (r+1)*Dl)``. A convolution needs ``lo`` planes from
the rank below and ``hi`` from the rank above (the *halo*); at the volume's own ends
the halo is filled as the unsplit layer would pad. The adjoint of that exchange is an
*overlap-add*, which a transposed convolution needs in its forward pass, so both
directions are written once here and each autograd function uses the other as its
backward.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

#: Global-end fills the exchange supports: what the unsplit layer pads with there.
EDGE_MODES = frozenset({"zeros", "replicate"})


def _neighbours(group: Any) -> tuple[int | None, int | None]:
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    below = dist.get_global_rank(group, rank - 1) if rank > 0 else None
    above = dist.get_global_rank(group, rank + 1) if rank < world - 1 else None
    return below, above


def _swap(
    to_below: torch.Tensor | None,
    to_above: torch.Tensor | None,
    from_below: torch.Tensor | None,
    from_above: torch.Tensor | None,
    group: Any,
) -> None:
    """Post every send and receive at once; empty slots are skipped symmetrically."""
    below, above = _neighbours(group)
    ops = []
    if below is not None:
        if to_below is not None:
            ops.append(dist.P2POp(dist.isend, to_below.contiguous(), below, group))
        if from_below is not None:
            ops.append(dist.P2POp(dist.irecv, from_below, below, group))
    if above is not None:
        if to_above is not None:
            ops.append(dist.P2POp(dist.isend, to_above.contiguous(), above, group))
        if from_above is not None:
            ops.append(dist.P2POp(dist.irecv, from_above, above, group))
    for request in dist.batch_isend_irecv(ops) if ops else ():
        request.wait()


def _planes(x: torch.Tensor, n: int) -> torch.Tensor:
    shape = list(x.shape)
    shape[2] = n
    return x.new_empty(shape)


def halo_forward(x: torch.Tensor, lo: int, hi: int, edge: str, group: Any) -> torch.Tensor:
    """``[lo planes from below | x | hi planes from above]``; global ends filled by ``edge``."""
    if max(lo, hi) > x.shape[2]:
        raise ValueError(f"halo ({lo}, {hi}) exceeds the {x.shape[2]} planes a rank holds")
    below, above = _neighbours(group)
    from_below = _planes(x, lo) if lo and below is not None else None
    from_above = _planes(x, hi) if hi and above is not None else None
    _swap(
        x[:, :, :hi] if hi else None,  # the rank below needs our first hi planes
        x[:, :, x.shape[2] - lo :] if lo else None,  # the rank above needs our last lo planes
        from_below,
        from_above,
        group,
    )
    parts = []
    if lo:
        parts.append(from_below if from_below is not None else _edge(x[:, :, :1], lo, edge))
    parts.append(x)
    if hi:
        parts.append(from_above if from_above is not None else _edge(x[:, :, -1:], hi, edge))
    return torch.cat(parts, dim=2) if len(parts) > 1 else x


def _edge(plane: torch.Tensor, n: int, edge: str) -> torch.Tensor:
    if edge == "zeros":
        return torch.zeros_like(plane).expand(-1, -1, n, -1, -1)
    return plane.expand(-1, -1, n, -1, -1)


def halo_adjoint(g: torch.Tensor, lo: int, hi: int, edge: str, group: Any) -> torch.Tensor:
    """The adjoint of :func:`halo_forward`: fold halo planes back onto their owners."""
    core = g.shape[2] - lo - hi
    out = g[:, :, lo : lo + core].clone()
    below, above = _neighbours(group)
    from_above = _planes(g, lo) if lo and above is not None else None
    from_below = _planes(g, hi) if hi and below is not None else None
    _swap(
        g[:, :, :lo] if lo else None,  # our lower halo came from the rank below's top
        g[:, :, lo + core :] if hi else None,  # our upper halo came from the rank above
        from_below,
        from_above,
        group,
    )
    if from_above is not None:
        out[:, :, core - lo :] += from_above
    if from_below is not None:
        out[:, :, :hi] += from_below
    if edge == "replicate":
        if lo and below is None:
            out[:, :, :1] += g[:, :, :lo].sum(dim=2, keepdim=True)
        if hi and above is None:
            out[:, :, -1:] += g[:, :, lo + core :].sum(dim=2, keepdim=True)
    return out


class HaloExchange(torch.autograd.Function):
    """Pad a depth slab with its neighbours' planes; backward folds the gradient back."""

    @staticmethod
    def forward(ctx, x, lo, hi, edge, group):  # type: ignore[override]
        ctx.lo, ctx.hi, ctx.edge, ctx.group = lo, hi, edge, group
        return halo_forward(x, lo, hi, edge, group)

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        return halo_adjoint(g, ctx.lo, ctx.hi, ctx.edge, ctx.group), None, None, None, None


class OverlapAdd(torch.autograd.Function):
    """Add the planes a slab produced for its neighbours into theirs (transposed conv)."""

    @staticmethod
    def forward(ctx, y, lo, hi, group):  # type: ignore[override]
        ctx.lo, ctx.hi, ctx.group = lo, hi, group
        return halo_adjoint(y, lo, hi, "zeros", group)

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        return halo_forward(g, ctx.lo, ctx.hi, "zeros", ctx.group), None, None, None


def exchange_halo(x: torch.Tensor, lo: int, hi: int, edge: str, group: Any) -> torch.Tensor:
    """Autograd-aware :func:`halo_forward`."""
    if edge not in EDGE_MODES:
        raise ValueError(f"unknown halo edge {edge!r}; expected one of {sorted(EDGE_MODES)}")
    return HaloExchange.apply(x, lo, hi, edge, group)


def overlap_add(y: torch.Tensor, lo: int, hi: int, group: Any) -> torch.Tensor:
    """Autograd-aware overlap-add of ``lo`` / ``hi`` spill planes onto the neighbours."""
    return OverlapAdd.apply(y, lo, hi, group)


def split_depth(x: torch.Tensor, group: Any) -> torch.Tensor:
    """This rank's slab of a full volume; depth must divide evenly across the group."""
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if x.shape[2] % world:
        raise ValueError(f"depth {x.shape[2]} does not split evenly across {world} ranks")
    size = x.shape[2] // world
    return x[:, :, rank * size : (rank + 1) * size]


def gather_depth(y: torch.Tensor, group: Any) -> torch.Tensor:
    """The full volume from every rank's slab, differentiable (all-gather)."""
    from torch.distributed.nn.functional import all_gather

    return torch.cat(all_gather(y.contiguous(), group=group), dim=2)


def reduce_spatial_gradients(model: torch.nn.Module, group: Any) -> None:
    """SUM parameter gradients over the spatial group: each slab holds a partial gradient.

    Weights are replicated across the group and every rank's gradient covers only its
    own planes, so the full gradient is the sum, not the mean a data-parallel reduce takes.
    """
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)


__all__ = [
    "EDGE_MODES",
    "exchange_halo",
    "gather_depth",
    "overlap_add",
    "reduce_spatial_gradients",
    "split_depth",
]
