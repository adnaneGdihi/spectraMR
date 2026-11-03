"""The equivalence gate: a context-parallel backbone against its unsplit copy, before training.

Two float32 copies (the stem's FFT computes in complex64) of the backbone, with every zero-initialised parameter randomised
identically (DiT's adaLN-Zero gates are zero at build, which would make attention
contribute nothing and get no gradient, so a broken split would pass). One copy is split.

* Every adopted attention, on a chunk of a random sequence under a random full mask:
  its output rows and its input gradient must equal the unsplit attention's rows. This
  checks the key/value reduce-scatter and the mask row offset directly.
* The backbone on a synthetic field: equal output, and the cp-mean of the input gradient
  and of every parameter gradient equal to the unsplit gradients. The cp-mean is what
  the dp x cp gradient average computes, so this checks the gather's x cp as well.

The copies are discarded; the live backbone is split only after the gate passes.

Under tensor x context parallelism the backbone arrives already split by heads (and
gated against its unsplit blocks by the tensor-parallel plan), so both copies are
tensor-parallel and this gate proves the sequence split on top of it; the two proofs
compose to the unsplit model. The copies share the live model's process groups, which
cannot be copied.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from spectramr.infrastructure.distributed.context_parallel.attention import CP_ATTENTIONS

#: Absolute tolerance relative to the largest expected magnitude, and relative tolerance.
_ATOL, _RTOL = 1e-5, 1e-4


def _randomise_zero_parameters(module: nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point() and not p.any():
                p.copy_(0.2 * torch.randn(p.shape, generator=gen, dtype=p.dtype))


def _copy(backbone: nn.Module) -> nn.Module:
    groups = {id(m.tp_group): m.tp_group for m in backbone.modules() if hasattr(m, "tp_group")}
    clone = copy.deepcopy(backbone, groups).float().train()
    if hasattr(clone, "grad_checkpointing"):
        clone.grad_checkpointing = False
    _randomise_zero_parameters(clone, seed=0)
    return clone


def _relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.abs().max().clamp_min(1e-12)
    return float((actual - expected).abs().max() / scale)


def _cp_mean(t: torch.Tensor, group: Any) -> torch.Tensor:
    t = t.detach().clone()
    dist.all_reduce(t, group=group)
    return t / dist.get_world_size(group)


class _Report:
    def __init__(self) -> None:
        self.problems: list[str] = []
        self.worst = 0.0

    def compare(self, label: str, actual: Any, expected: Any) -> None:
        actual = actual.detach() if actual is not None else None
        expected = expected.detach() if expected is not None else None
        if actual is None or expected is None:
            if (actual is None) != (expected is None):
                self.problems.append(f"{label} (present on one side only)")
            return
        if actual.shape != expected.shape:
            self.problems.append(f"{label} (shape {tuple(actual.shape)})")
            return
        self.worst = max(self.worst, _relative(actual, expected))
        atol = _ATOL * float(expected.abs().max().clamp_min(1.0))
        if not torch.allclose(actual, expected, atol=atol, rtol=_RTOL):
            self.problems.append(label)


def _check_attention(
    name: str, reference: nn.Module, split: nn.Module, group: Any, report: _Report
) -> None:
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    width, local = split.proj.out_features, 3  # in_features is 1/tp of it under tp
    gen = torch.Generator().manual_seed(1)
    device = split.proj.weight.device
    x = torch.randn(2, local * world, width, generator=gen, dtype=torch.float32).to(device)
    mask = torch.randn(local * world, local * world, generator=gen, dtype=torch.float32)
    mask, rows = mask.to(device), slice(rank * local, (rank + 1) * local)
    weights = torch.randn(2, local * world, width, generator=gen, dtype=torch.float32).to(device)
    xr, xs = x.clone().requires_grad_(True), x[:, rows].clone().requires_grad_(True)
    out_r, out_s = reference(xr, attn_mask=mask), split(xs, attn_mask=mask)
    (out_r * weights).sum().backward()
    (out_s * weights[:, rows]).sum().backward()
    report.compare(f"{name} masked attention output", out_s, out_r[:, rows])
    report.compare(f"{name} attention input gradient", xs.grad, xr.grad[:, rows])
    for module in (reference, split):
        module.zero_grad(set_to_none=True)


def _probe_grid(extra_tokens: int, world: int) -> int:
    for grid in range(2, 65):
        if (grid * grid + extra_tokens) % world == 0:
            return grid
    raise ValueError(f"no probe grid up to 64 gives a token count divisible by cp={world}")


def verify_backbone(backbone: nn.Module, plan: Any, group: Any) -> tuple[list[str], float]:
    """Problems found (empty when equivalent) and the worst relative error seen.

    TF32 is off for the probe: split and unsplit convolve differently shaped gradients,
    and TF32 rounding (~1e-3 on Ada) would exceed the tolerance on a correct split.
    """
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    saved = (matmul.allow_tf32, cudnn.allow_tf32)
    matmul.allow_tf32, cudnn.allow_tf32 = False, False
    try:
        return _verify(backbone, plan, group)
    finally:
        matmul.allow_tf32, cudnn.allow_tf32 = saved


def _verify(backbone: nn.Module, plan: Any, group: Any) -> tuple[list[str], float]:
    from spectramr.infrastructure.distributed.context_parallel.plan import install

    reference, split = _copy(backbone), _copy(backbone)
    install(split, plan, group)
    report = _Report()
    for name, module in split.named_modules():
        if isinstance(module, CP_ATTENTIONS):
            _check_attention(name, reference.get_submodule(name), module, group, report)

    device = next(split.parameters()).device
    side = _probe_grid(plan.extra_tokens, dist.get_world_size(group)) * backbone.stem.patch_size
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(2, backbone.in_channels, side, side, generator=gen, dtype=torch.float32)
    t = torch.rand(2, generator=gen, dtype=torch.float32).to(device)
    x = x.to(device)
    xr, xs = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    out_r, out_s = reference(xr, t), split(xs, t)
    weights = torch.randn(out_r.shape, generator=gen, dtype=torch.float32).to(device)
    (out_r * weights).sum().backward()
    (out_s * weights).sum().backward()
    report.compare("output", out_s, out_r)
    report.compare("input gradient (cp mean)", _cp_mean(xs.grad, group), xr.grad)
    expected = dict(reference.named_parameters())
    for key, param in split.named_parameters():
        grad = None if param.grad is None else _cp_mean(param.grad, group)
        report.compare(f"gradient of {key} (cp mean)", grad, expected[key].grad)
    return report.problems, report.worst


__all__ = ["verify_backbone"]
