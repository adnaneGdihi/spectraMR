"""The two tensor-parallel communication ops (Megatron's ``f`` and ``g``).

A column-then-row sharded pair (``Linear`` split by output rows, then ``Linear``
split by input columns) needs exactly two: the input is copied into the pair with an
all-reduce of its gradient on the way back, and the partial outputs are summed on the
way out with an identity gradient. Everything outside the pair is replicated and stays
identical on every tensor rank.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class _CopyToTensorParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):  # type: ignore[override]
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        g = g.contiguous()
        dist.all_reduce(g, group=ctx.group)
        return g, None


class _ReduceFromTensorParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):  # type: ignore[override]
        x = x.contiguous()
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        return g, None


def copy_to_tensor_parallel(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Identity forward; all-reduce of the gradient backward."""
    return _CopyToTensorParallel.apply(x, group)


def reduce_from_tensor_parallel(x: torch.Tensor, group: Any) -> torch.Tensor:
    """All-reduce (sum) of partial outputs forward; identity backward."""
    return _ReduceFromTensorParallel.apply(x, group)


def gather_along(t: torch.Tensor, dim: int, group: Any) -> torch.Tensor:
    """Concatenate every tensor rank's slice along ``dim`` (no autograd; for checkpoints)."""
    parts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, t.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


__all__ = ["copy_to_tensor_parallel", "gather_along", "reduce_from_tensor_parallel"]
