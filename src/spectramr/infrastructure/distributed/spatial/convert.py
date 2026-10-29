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
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch import nn
from torch.utils._pytree import tree_flatten

from spectramr.infrastructure.distributed.spatial.conv import (
    SpatialConv3d,
    SpatialConvTranspose3d,
    SpatialUpsample,
)
from spectramr.infrastructure.distributed.spatial.halo import (
    reduce_spatial_gradients,
    spatial_scope,
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


class SpatialReport(NamedTuple):
    """What the gate proved: the worst relative error, and which output leaves split.

    ``split_mask`` has one entry per output tensor, in ``tree_flatten`` order. A leaf
    is split when the local output's shape differs from the unsplit one (it carries
    depth); a replicated leaf (a global pool, say) is identical on every rank.
    """

    worst: float
    split_mask: tuple[bool, ...]


def _compare(
    reference: nn.Module,
    model: nn.Module,
    x: torch.Tensor,
    gen: torch.Generator,
    group: Any,
    atol: float,
    rtol: float,
    failures: list[str],
) -> SpatialReport:
    sp = dist.get_world_size(group)
    x_ref = x.clone().requires_grad_(True)
    x_loc = split_depth(x, group).clone().requires_grad_(True)
    ref_out = reference(x_ref)
    with spatial_scope():
        loc_out = model(x_loc)
    ref_leaves, _ = tree_flatten(ref_out)
    loc_leaves, _ = tree_flatten(loc_out)
    if len(ref_leaves) != len(loc_leaves) or not all(
        torch.is_tensor(t) for t in (*ref_leaves, *loc_leaves)
    ):
        failures.append("output structure differs from the unsplit model's")
        return SpatialReport(0.0, ())
    split = tuple(r.shape != o.shape for r, o in zip(ref_leaves, loc_leaves, strict=True))
    weights = [torch.randn(r.shape, generator=gen).to(x.device) for r in ref_leaves]
    # Reference: the unscaled loss. Sharded: a split leaf takes its slab of the weights;
    # a replicated leaf is identical on every rank, so its loss is divided by sp and the
    # SUM over the group below restores one copy.
    sum(((r * w).sum() for r, w in zip(ref_leaves, weights, strict=True))).backward()
    loc_terms = []
    for leaf, w, is_split in zip(loc_leaves, weights, split, strict=True):
        if is_split:
            loc_terms.append((leaf * split_depth(w, group)).sum())
        else:
            loc_terms.append((leaf * w).sum() / sp)
    sum(loc_terms).backward()
    reduce_spatial_gradients(model, group)
    pairs = []
    for ref, got, is_split in zip(ref_leaves, loc_leaves, split, strict=True):
        pairs.append((split_depth(ref, group) if is_split else ref, got, "output"))
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
    return SpatialReport(worst, split)


def check_spatial_equivalence(
    reference: nn.Module,
    model: nn.Module,
    sample_shape: tuple[int, ...],
    group: Any,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    seed: int = 0,
) -> SpatialReport:
    """Return the :class:`SpatialReport`; raise :class:`SpatialEquivalenceError` on every rank.

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
        report = SpatialReport(0.0, ())
        try:
            report = _compare(reference, model, x, gen, group, atol, rtol, failures)
        except Exception as exc:  # a shape error from a depth-mixing op is a failed check
            failures.append(f"{type(exc).__name__}: {exc}")
        flag = torch.tensor([float(bool(failures))], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        if flag.item():
            raise SpatialEquivalenceError(
                "the depth-split model does not reproduce the unsplit one"
                + (": " + "; ".join(failures[:5]) if failures else " on another rank")
            )
        return report
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
    "SpatialReport",
    "check_spatial_equivalence",
    "convert_to_spatial",
    "make_spatial",
]
