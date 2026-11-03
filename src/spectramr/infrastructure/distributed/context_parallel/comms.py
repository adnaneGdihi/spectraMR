"""The two sequence collectives of context parallelism, as autograd ops.

Both gather ``[B, n, C]`` chunks along the sequence (dim 1) into ``[B, cp * n, C]``,
in cp-rank order. They differ in what the gradient means on the way back:

* :func:`gather_keys_values` feeds every rank's queries from every rank's keys, so the
  gradient of a key row arrives on every rank and is summed back onto the rank that
  owns the row (a reduce-scatter).
* :func:`gather_sequence` ends the split trunk. Every rank computes the same loss from
  the gathered sequence, so this rank's slice of the incoming gradient is already the
  true gradient of its chunk; it is multiplied by cp so that the dp x cp gradient
  average recovers the unsplit gradient for the parameters inside the trunk (the
  scheme of ``spatial/wrapper.py``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def _all_gather_seq(x: torch.Tensor, group: Any) -> torch.Tensor:
    world = dist.get_world_size(group)
    local = x.transpose(0, 1).contiguous()  # sequence-major, so each chunk is contiguous
    out = local.new_empty((world * local.shape[0], *local.shape[1:]))
    dist.all_gather_single(out, local, group=group)
    return out.transpose(0, 1)


def _local_rows(x: torch.Tensor, group: Any) -> torch.Tensor:
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    n = x.shape[1] // world
    return x.narrow(1, rank * n, n)


class _GatherKeysValues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):  # type: ignore[override]
        ctx.group = group
        return _all_gather_seq(x, group)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        world = dist.get_world_size(ctx.group)
        full = grad.transpose(0, 1).contiguous()
        out = full.new_empty((full.shape[0] // world, *full.shape[1:]))
        dist.reduce_scatter_single(out, full, op=dist.ReduceOp.SUM, group=ctx.group)
        return out.transpose(0, 1), None


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):  # type: ignore[override]
        ctx.group = group
        return _all_gather_seq(x, group)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        world = dist.get_world_size(ctx.group)
        return _local_rows(grad, ctx.group).mul(world), None


def gather_keys_values(x: torch.Tensor, group: Any) -> torch.Tensor:
    """All-gather along the sequence; backward reduce-scatters (sums) to the owner."""
    return _GatherKeysValues.apply(x, group)


def gather_sequence(x: torch.Tensor, group: Any) -> torch.Tensor:
    """All-gather along the sequence; backward is this rank's slice times cp."""
    return _GatherSequence.apply(x, group)


def split_sequence(x: torch.Tensor, group: Any) -> torch.Tensor:
    """This rank's contiguous chunk of the sequence (a view; autograd zero-pads it back)."""
    world = dist.get_world_size(group)
    if x.shape[1] % world:
        raise ValueError(
            f"context parallelism splits the token sequence across {world} ranks, but the "
            f"sequence has {x.shape[1]} tokens. Choose a matrix size and patch size whose "
            "token count (plus any extra token, such as U-ViT's time token) divides by "
            "parallel.mesh.context."
        )
    return _local_rows(x, group)


__all__ = ["gather_keys_values", "gather_sequence", "split_sequence"]
