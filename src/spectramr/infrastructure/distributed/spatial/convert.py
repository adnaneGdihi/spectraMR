"""Convert a 3D model to depth-split execution, and prove the conversion before trusting it.

:func:`convert_to_spatial` adopts every layer it knows how to split (exact type match:
a subclass may override ``forward``, and adopting it would replace that override).
A module list cannot see functional ops -- an FFT along depth, ``x.mean(dim=2)``,
``F.interpolate``, a reflect ``F.pad``, a ``flatten`` -- so the gate is
:func:`check_spatial_equivalence`: the converted model on its slab must reproduce the
unsplit model's output slab, input gradient and (summed) parameter gradients.
:func:`make_spatial` does both and raises on every rank alike.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from spectramr.infrastructure.distributed.spatial.conv import (
    SpatialConv3d,
    SpatialConvTranspose3d,
    SpatialUpsample,
)
from spectramr.infrastructure.distributed.spatial.halo import (
    reduce_spatial_gradients,
    split_depth,
)
from spectramr.infrastructure.distributed.spatial.norms import (
    SpatialBatchNorm3d,
    SpatialGlobalAvgPool3d,
    SpatialGroupNorm,
    SpatialInstanceNorm3d,
)

#: Exact layer type -> its in-place adopter.
ADOPTERS: dict[type[nn.Module], Callable[[Any, Any], nn.Module]] = {
    nn.Conv3d: SpatialConv3d.adopt,
    nn.ConvTranspose3d: SpatialConvTranspose3d.adopt,
    nn.Upsample: SpatialUpsample.adopt,
    nn.GroupNorm: SpatialGroupNorm.adopt,
    nn.InstanceNorm3d: SpatialInstanceNorm3d.adopt,
    nn.BatchNorm3d: SpatialBatchNorm3d.adopt,
    nn.AdaptiveAvgPool3d: SpatialGlobalAvgPool3d.adopt,
}

#: Layers that mix information across the whole depth and have no split form.
REFUSED: tuple[type[nn.Module], ...] = (
    nn.MultiheadAttention,
    nn.AdaptiveMaxPool3d,
    nn.LayerNorm,
)


class SpatialEquivalenceError(RuntimeError):
    """The depth-split model does not reproduce the unsplit one."""


def convert_to_spatial(model: nn.Module, group: Any) -> nn.Module:
    """Adopt every splittable layer in place; raise on a layer known to mix all of depth."""
    for name, module in model.named_modules():
        if isinstance(module, REFUSED):
            raise SpatialEquivalenceError(
                f"{name or type(model).__name__} ({type(module).__name__}) mixes the whole "
                "depth axis and has no depth-split form"
            )
        adopter = ADOPTERS.get(type(module))
        if adopter is not None:
            adopter(module, group)
    return model


def _slab_or_same(full: torch.Tensor, local: torch.Tensor, group: Any) -> torch.Tensor:
    """The part of a full-volume tensor this rank's output corresponds to."""
    if full.shape == local.shape:
        return full  # replicated (e.g. after a global pool)
    return split_depth(full, group)


def _replicas_differ(model: nn.Module, group: Any) -> list[str]:
    """Names of parameters/buffers that are not identical on every rank of the group.

    Depth-split ranks must hold the same weights: a rank-dependent initialisation
    would make each slab a different network while every local output still looks right.
    """
    names = []
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if not tensor.is_floating_point():
            continue
        lo, hi = tensor.detach().clone(), tensor.detach().clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
        if not torch.equal(lo, hi):
            names.append(name)
    return names


def _compare(
    reference: nn.Module,
    model: nn.Module,
    x: torch.Tensor,
    gen: torch.Generator,
    group: Any,
    atol: float,
    rtol: float,
    failures: list[str],
) -> float:
    x_ref = x.clone().requires_grad_(True)
    x_loc = split_depth(x, group).clone().requires_grad_(True)
    ref_out, loc_out = reference(x_ref), model(x_loc)
    weights = torch.randn(ref_out.shape, generator=gen).to(x.device)
    (ref_out * weights).sum().backward()
    scale = dist.get_world_size(group) if ref_out.shape == loc_out.shape else 1
    (loc_out * _slab_or_same(weights, loc_out, group)).sum().div(scale).backward()
    reduce_spatial_gradients(model, group)
    pairs = [(_slab_or_same(ref_out, loc_out, group), loc_out, "output")]
    pairs.append((split_depth(x_ref.grad, group), x_loc.grad, "input gradient"))
    ref_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        if param.grad is not None:
            pairs.append((ref_params[name].grad, param.grad, f"grad {name}"))
    worst = 0.0
    for expected, got, what in pairs:
        # Relative to the tensor's scale, not element-wise: gradients are sums of
        # per-slab partials, so their rounding differs in the last bits.
        diff, scale = float((got - expected).abs().max()), float(expected.abs().max())
        worst = max(worst, diff / (scale + 1e-12))
        if diff > atol + rtol * scale:
            failures.append(f"{what}: max |diff| {diff:.2e} at scale {scale:.2e}")
    return worst


def check_spatial_equivalence(
    reference: nn.Module,
    model: nn.Module,
    sample_shape: tuple[int, ...],
    group: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    seed: int = 0,
) -> float:
    """Return the worst relative error; raise :class:`SpatialEquivalenceError` on every rank.

    Runs in training mode (batch statistics included) with dropout disabled, and restores
    every buffer and mode afterwards so the check leaves no trace on running statistics.
    """
    tensors = [*model.parameters(), *model.buffers()]
    device = tensors[0].device if tensors else torch.device("cpu")
    saved = [{k: v.clone() for k, v in m.state_dict().items()} for m in (reference, model)]
    modes = [m.training for m in (reference, model)]
    try:
        for m in (reference, model):
            m.train()
            for sub in m.modules():
                if isinstance(sub, nn.modules.dropout._DropoutNd):
                    sub.eval()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(sample_shape, generator=gen).to(device)
        failures = [f"weights differ across ranks: {n}" for n in _replicas_differ(model, group)][:3]
        worst = 0.0
        try:
            worst = _compare(reference, model, x, gen, group, atol, rtol, failures)
        except Exception as exc:  # a shape error from a depth-mixing op is a failed check
            failures.append(f"{type(exc).__name__}: {exc}")
        flag = torch.tensor([float(bool(failures))], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        if flag.item():
            raise SpatialEquivalenceError(
                "the depth-split model does not reproduce the unsplit one"
                + (": " + "; ".join(failures[:5]) if failures else " on another rank")
            )
        return worst
    finally:
        for m, state, mode in zip((reference, model), saved, modes, strict=True):
            m.load_state_dict(state)
            m.train(mode)
            m.zero_grad(set_to_none=True)


def make_spatial(
    model: nn.Module, group: Any, sample_shape: tuple[int, ...], **tol: Any
) -> nn.Module:
    """Convert ``model`` in place and prove it on a synthetic volume of ``sample_shape``."""
    reference = copy.deepcopy(model)
    convert_to_spatial(model, group)
    check_spatial_equivalence(reference, model, sample_shape, group, **tol)
    return model


__all__ = [
    "ADOPTERS",
    "REFUSED",
    "SpatialEquivalenceError",
    "check_spatial_equivalence",
    "convert_to_spatial",
    "make_spatial",
]
