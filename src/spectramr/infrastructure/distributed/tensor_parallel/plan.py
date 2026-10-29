"""Which blocks are tensor-parallel, how they are split, and the proof before training.

Every ``TokenAttention`` is split by heads (DiT, UViT, HAT and the k-space AR/masked
backbones share it). MLPs are split where a registered block names them. A model with
nothing to split raises: a tensor mesh over a model that splits nothing would replicate
the whole network on every tensor rank and claim otherwise in provenance.

Each adopted block is checked against an unsplit copy before training starts: equal
output and input gradient, and after one SGD step on both, equal full weights. The
step reads the weights back through the block's state-dict hook and restores them
through its load hook, so the checkpoint path is exercised by the same check.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from spectramr.infrastructure.distributed.tensor_parallel.layers import (
    TensorParallelAttention,
    TensorParallelMLP,
)
from spectramr.models.generators.diffusion_transformer_stem import TokenAttention
from spectramr.models.generators.dit_backbone import DiTBlock
from spectramr.models.generators.uvit_backbone import UViTBlock

#: Block class -> attribute names of its ``Sequential(Linear, act, Linear)`` MLPs.
MLP_PLANS: dict[type[nn.Module], tuple[str, ...]] = {DiTBlock: ("mlp",), UViTBlock: ("mlp",)}


class TensorParallelError(RuntimeError):
    """The model cannot be, or was not correctly, split across the tensor group."""


def _targets(model: nn.Module) -> list[tuple[str, nn.Module, Any]]:
    found = []
    for name, module in model.named_modules():
        if type(module) is TokenAttention:
            found.append((name, module, TensorParallelAttention.adopt))
        for attr in MLP_PLANS.get(type(module), ()):
            found.append(
                (f"{name}.{attr}".lstrip("."), getattr(module, attr), TensorParallelMLP.adopt)
            )
    return found


def _identical_across(module: nn.Module, group: Any) -> bool:
    same = True
    for tensor in module.state_dict().values():
        if tensor.is_floating_point():
            lo, hi = tensor.detach().clone(), tensor.detach().clone()
            dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
            same = same and torch.equal(lo, hi)
    return same


def _verify(reference: nn.Module, split: nn.Module, width: int, group: Any) -> list[str]:
    """Output, input gradient and one-step weights of ``split`` against ``reference``."""
    device = next(split.parameters()).device
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(2, 8, width, generator=gen).to(device)
    xr, xs = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    weights = torch.randn(2, 8, width, generator=gen).to(device)
    out_r, out_s = reference(xr), split(xs)
    (out_r * weights).sum().backward()
    (out_s * weights).sum().backward()
    problems = []
    if not torch.allclose(out_s, out_r, atol=1e-5, rtol=1e-4):
        problems.append("output")
    if not torch.allclose(xs.grad, xr.grad, atol=1e-5, rtol=1e-4):
        problems.append("input gradient")
    # Cloned: unsharded entries share storage with the live parameters, and the
    # step below would otherwise rewrite the snapshot the restore reads.
    before = {k: v.clone() for k, v in split.state_dict().items()}
    for module in (reference, split):
        with torch.no_grad():
            for p in module.parameters():
                p -= 0.1 * p.grad
                p.grad = None
    stepped, expected = split.state_dict(), reference.state_dict()
    problems += [
        f"stepped {k}" for k in expected if not torch.allclose(stepped[k], expected[k], atol=1e-5)
    ]
    split.load_state_dict(before)
    return problems


def parallelize_tensor(model: nn.Module, group: Any) -> list[str]:
    """Split every planned block of ``model`` across ``group`` in place; return their names.

    Raises :class:`TensorParallelError` on every rank alike when nothing is planned,
    when replicas differ before the split, or when a split block disagrees with its
    unsplit copy.
    """
    targets = _targets(model)
    if not targets:
        raise TensorParallelError(
            f"{type(model).__name__} has no tensor-parallel plan: no TokenAttention and no "
            f"registered MLP block ({sorted(c.__name__ for c in MLP_PLANS)})."
        )
    failures = []
    for name, module, adopt in targets:
        if not _identical_across(module, group):
            failures.append(f"{name}: weights differ across tensor ranks")
            continue
        reference = copy.deepcopy(module)
        width = (
            module.proj.in_features if isinstance(module, TokenAttention) else module[0].in_features
        )
        adopt(module, group)
        failures += [f"{name}: {p}" for p in _verify(reference, module, width, group)]
    # On the model's device: NCCL reduces device tensors only.
    flag = torch.tensor([float(bool(failures))], device=next(model.parameters()).device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    if flag.item():
        raise TensorParallelError(
            "tensor-parallel split does not reproduce the unsplit blocks: "
            + ("; ".join(failures[:5]) if failures else "on another rank")
        )
    return [name for name, *_ in targets]


__all__ = ["MLP_PLANS", "TensorParallelError", "parallelize_tensor"]
