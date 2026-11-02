"""Normalisation and global pooling whose statistics span the whole, depth-split volume.

Each adopts the original module in place (parameters, buffers and state-dict keys
unchanged) and reduces its sums across the spatial group with a differentiable
all-reduce, so forward values and gradients equal the unsplit layer's. Statistics
are taken in two passes (mean, then centred second moment) for the same numerics as
the native kernels at float32. A half-precision slab is normalised in float32 (an fp16
sum over a volume overflows at 65504; autocast does the same for the native norms) and
returned in its own dtype, so an FSDP fp16 layer after it receives fp16.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from spectramr.infrastructure.distributed.spatial.halo import require_spatial_scope


def _all_sum(t: torch.Tensor, group: Any) -> torch.Tensor:
    from torch.distributed.nn.functional import all_reduce

    return all_reduce(t, group=group)


def _global_count(x: torch.Tensor, dims: tuple[int, ...], group: Any) -> float:
    # Slabs are equal (split_depth refuses an uneven split), so the global count is known
    # on the host: no tensor to copy to the device and no collective to wait on.
    return float(math.prod(x.shape[d] for d in dims) * dist.get_world_size(group))


def _group_stats(
    x: torch.Tensor, dims: tuple[int, ...], group: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    count = _global_count(x, dims, group)
    mean = _all_sum(x.sum(dim=dims, keepdim=True), group) / count
    var = _all_sum((x - mean).square().sum(dim=dims, keepdim=True), group) / count
    return mean, var


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def _affine(y: torch.Tensor, norm: nn.Module, dtype: torch.dtype) -> torch.Tensor:
    if norm.affine:
        y = y * norm.weight.view(1, -1, 1, 1, 1) + norm.bias.view(1, -1, 1, 1, 1)
    return y.to(dtype)


class SpatialGroupNorm(nn.GroupNorm):
    """``GroupNorm`` over a depth-split ``[B, C, D, H, W]`` volume."""

    spatial_group: Any

    @classmethod
    def adopt(cls, norm: nn.GroupNorm, group: Any) -> SpatialGroupNorm:
        norm.__class__ = cls
        norm.spatial_group = group
        return norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        require_spatial_scope("SpatialGroupNorm")
        b, c = x.shape[:2]
        grouped = _f32(x).reshape(b, self.num_groups, c // self.num_groups, *x.shape[2:])
        mean, var = _group_stats(grouped, tuple(range(2, grouped.dim())), self.spatial_group)
        y = ((grouped - mean) / torch.sqrt(var + self.eps)).reshape(x.shape)
        return _affine(y, self, x.dtype)


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
        require_spatial_scope("SpatialInstanceNorm3d")
        x32 = _f32(x)
        mean, var = _group_stats(x32, (2, 3, 4), self.spatial_group)
        return _affine((x32 - mean) / torch.sqrt(var + self.eps), self, x.dtype)


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
        require_spatial_scope("SpatialBatchNorm3d")
        if not (self.training or not self.track_running_stats):
            return super().forward(x)
        x32 = _f32(x)
        mean, var = _group_stats(x32, (0, 2, 3, 4), self.spatial_group)
        if self.training and self.track_running_stats:
            count = _global_count(x, (0, 2, 3, 4), self.spatial_group)
            with torch.no_grad():
                self.num_batches_tracked += 1
                momentum = (
                    self.momentum
                    if self.momentum is not None
                    else 1.0 / float(self.num_batches_tracked)
                )
                unbiased = var.flatten() * (count / max(count - 1, 1.0))
                self.running_mean.lerp_(mean.flatten().to(self.running_mean.dtype), momentum)
                self.running_var.lerp_(unbiased.to(self.running_var.dtype), momentum)
        return _affine((x32 - mean) / torch.sqrt(var + self.eps), self, x.dtype)


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
        require_spatial_scope("SpatialGlobalAvgPool3d")
        count = _global_count(x, (2, 3, 4), self.spatial_group)
        total = _all_sum(_f32(x).sum(dim=(2, 3, 4), keepdim=True), self.spatial_group)
        return (total / count).to(x.dtype)


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
