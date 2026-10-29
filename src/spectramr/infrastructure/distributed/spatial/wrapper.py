"""The depth-split model as a drop-in module: full volume in, full output out.

Every rank of a spatial group receives the identical volume (the data sampler routes by
data-parallel rank), computes its own depth slab, and gathers the slabs of each split
output so the loss sees the full output on every rank. The gather's backward keeps only
this rank's slab of the incoming gradient and multiplies it by the group size: every rank
computes the same loss, so the slab's partial gradient is ``1/sp`` of the total. The
data-parallel reduce then averages over ``dp x sp`` and recovers the unsplit gradient in
one collective, with no second reduction.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from spectramr.infrastructure.distributed.spatial.convert import (
    SpatialReport,
    check_spatial_equivalence,
    convert_to_spatial,
)
from spectramr.infrastructure.distributed.spatial.halo import spatial_scope, split_depth


class _GatherDepth(torch.autograd.Function):
    """All-gather a depth slab into the full volume; backward is this rank's slab x sp."""

    @staticmethod
    def forward(ctx, slab, group):  # type: ignore[override]
        ctx.group = group
        parts = [torch.empty_like(slab) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, slab.contiguous(), group=group)
        return torch.cat(parts, dim=2)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        world = dist.get_world_size(ctx.group)
        return split_depth(grad.contiguous(), ctx.group).mul(world), None


class SpatialModel(nn.Module):
    """Wrap a converted model so it takes and returns full volumes.

    ``self.module`` is the converted model, so state-dict keys are the unsplit model's
    (prefixed ``module.``, which every checkpoint path already strips). Only ``forward``
    is supported: a converted layer called on a full volume would compute a different
    network, so any other method of the inner model raises instead of passing through.
    """

    def __init__(
        self, module: nn.Module, group: Any, report: SpatialReport, probe_shape: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.module = module
        self._spatial_group = group
        self.split_mask = report.split_mask
        self.gate_report = {
            "spatial_size": dist.get_world_size(group),
            "probe_shape": list(probe_shape),
            "worst_relative_error": report.worst,
            "split_outputs": sum(report.split_mask),
            "replicated_outputs": len(report.split_mask) - sum(report.split_mask),
        }

    def forward(self, x: torch.Tensor) -> Any:
        if not torch.is_tensor(x) or x.dim() != 5:
            raise ValueError("SpatialModel takes one [B, C, D, H, W] volume")
        slab = split_depth(x, self._spatial_group)
        with spatial_scope():
            out = self.module(slab)
        leaves, spec = tree_flatten(out)
        if len(leaves) != len(self.split_mask):
            raise RuntimeError(
                f"the model returned {len(leaves)} tensors; the equivalence gate saw "
                f"{len(self.split_mask)}"
            )
        gathered = [
            _GatherDepth.apply(leaf, self._spatial_group) if split else leaf
            for leaf, split in zip(leaves, self.split_mask, strict=True)
        ]
        return tree_unflatten(gathered, spec)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_modules", {}).get("module")
            if inner is None:
                raise
            attr = getattr(inner, name)  # AttributeError propagates for unknown names
            if isinstance(attr, nn.Module):
                return attr
            if callable(attr):
                # Not AttributeError: getattr(model, name, None) must not turn this into None.
                raise RuntimeError(
                    f"{name!r} is a method of the depth-split model; a spatial model runs "
                    "only through forward(volume)"
                ) from None
            if _holds_tensor(attr):
                # A cached tensor (a posterior, an occupancy map) holds one slab, not the volume.
                raise RuntimeError(
                    f"{name!r} is a tensor cached by the depth-split model's forward and holds "
                    "only this rank's slab; it cannot be read from the wrapper"
                ) from None
            return attr


def _holds_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return True
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list | tuple):
        return any(_holds_tensor(v) for v in value)
    return False


def wrap_spatial(
    model: nn.Module, group: Any, sample_shape: tuple[int, ...], **tol: Any
) -> SpatialModel:
    """Convert ``model`` in place, prove it on a synthetic volume, and wrap it.

    Raises when the model has no 3D convolution (a 2-D model gains nothing from depth
    splitting), when a layer mixes the whole depth, or when the gate disagrees. The gate
    runs on every rank of the group, so a failure is raised everywhere alike.
    """
    if not any(isinstance(m, nn.Conv3d | nn.ConvTranspose3d) for m in model.modules()):
        raise ValueError(
            "parallel.mesh.spatial needs a 3D model: found no Conv3d or ConvTranspose3d"
        )
    reference = copy.deepcopy(model)
    convert_to_spatial(model, group)
    report = check_spatial_equivalence(reference, model, sample_shape, group, **tol)
    return SpatialModel(model, group, report, tuple(sample_shape))


__all__ = ["SpatialModel", "wrap_spatial"]
