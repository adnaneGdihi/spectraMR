"""Normalisation and global pooling whose statistics span the whole, depth-split volume.

Each adopts the original module in place (parameters, buffers and state-dict keys
unchanged) and reduces its sums across the spatial group with a differentiable
all-reduce, so forward values and gradients equal the unsplit layer's. Statistics
are taken in two passes (mean, then centred second moment) for the same numerics as
the native kernels at float32.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _all_sum(t: torch.Tensor, group: Any) -> torch.Tensor:
    from torch.distributed.nn.functional import all_reduce

    return all_reduce(t, group=group)


def _group_stats(
    x: torch.Tensor, dims: tuple[int, ...], group: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    count = torch.tensor(
        [float(torch.tensor([x.shape[d] for d in dims]).prod())], device=x.device, dtype=x.dtype
    )
    count = _all_sum(count, group)
    mean = _all_sum(x.sum(dim=dims, keepdim=True), group) / count
    var = _all_sum((x - mean).square().sum(dim=dims, keepdim=True), group) / count
    return mean, var


class SpatialGroupNorm(nn.GroupNorm):
    """``GroupNorm`` over a depth-split ``[B, C, D, H, W]`` volume."""

    spatial_group: Any

    @classmethod
    def adopt(cls, norm: nn.GroupNorm, group: Any) -> SpatialGroupNorm:
        norm.__class__ = cls
        norm.spatial_group = group
        return norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        grouped = x.reshape(b, self.num_groups, c // self.num_groups, *x.shape[2:])
        mean, var = _group_stats(grouped, tuple(range(2, grouped.dim())), self.spatial_group)
        y = ((grouped - mean) / torch.sqrt(var + self.eps)).reshape_as(x)
        if self.affine:
            y = y * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return y


class SpatialInstanceNorm3d(nn.InstanceNorm3d):
    """``InstanceNorm3d`` (no running statistics) over a depth-split volume."""

    spatial_group: Any

    @classmethod
    def adopt(cls, norm: nn.InstanceNorm3d, group: Any) -> SpatialInstanceNorm3d:
        if norm.track_running_stats:
            raise ValueError("spatial InstanceNorm3d with track_running_stats is not supported")
        norm.__class__ = cls
        norm.spatial_group = group
        return norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = _group_stats(x, (2, 3, 4), self.spatial_group)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            y = y * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return y


class SpatialBatchNorm3d(nn.BatchNorm3d):
    """``BatchNorm3d`` whose training statistics span the spatial group.

    ``SyncBatchNorm`` does this on CUDA only; this runs on any device. In eval mode the
    running statistics are local and identical on every rank, so no collective runs.
    """

    spatial_group: Any

    @classmethod
    def adopt(cls, norm: nn.BatchNorm3d, group: Any) -> SpatialBatchNorm3d:
        norm.__class__ = cls
        norm.spatial_group = group
        return norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.training or not self.track_running_stats):
            return super().forward(x)
        mean, var = _group_stats(x, (0, 2, 3, 4), self.spatial_group)
        if self.training and self.track_running_stats:
            count = _all_sum(
                torch.tensor([float(x.numel() // x.shape[1])], device=x.device), self.spatial_group
            )
            with torch.no_grad():
                self.num_batches_tracked += 1
                momentum = (
                    self.momentum
                    if self.momentum is not None
                    else 1.0 / float(self.num_batches_tracked)
                )
                unbiased = var.flatten() * count / (count - 1).clamp_min(1)
                self.running_mean.lerp_(mean.flatten(), momentum)
                self.running_var.lerp_(unbiased, momentum)
        y = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            y = y * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return y


class SpatialGlobalAvgPool3d(nn.AdaptiveAvgPool3d):
    """``AdaptiveAvgPool3d(1)`` over a depth-split volume: one mean across every slab."""

    spatial_group: Any

    @classmethod
    def adopt(cls, pool: nn.AdaptiveAvgPool3d, group: Any) -> SpatialGlobalAvgPool3d:
        size = pool.output_size
        if (size if isinstance(size, int) else tuple(size)) not in (1, (1, 1, 1)):
            raise ValueError(f"spatial AdaptiveAvgPool3d supports output_size 1 only, not {size}")
        pool.__class__ = cls
        pool.spatial_group = group
        return pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        count = _all_sum(
            torch.tensor([float(x[0, 0].numel())], device=x.device, dtype=x.dtype),
            self.spatial_group,
        )
        return _all_sum(x.sum(dim=(2, 3, 4), keepdim=True), self.spatial_group) / count


def adaptive_pool_is_global(pool: nn.Module) -> bool:
    size = getattr(pool, "output_size", None)
    return (size if isinstance(size, int) else tuple(size or ())) in (1, (1, 1, 1))


__all__ = [
    "SpatialBatchNorm3d",
    "SpatialGlobalAvgPool3d",
    "SpatialGroupNorm",
    "SpatialInstanceNorm3d",
    "adaptive_pool_is_global",
]
