"""Which backbones run context-parallel, where their token sequence is split, and the proof.

A planned backbone embeds and position-encodes the full field on every rank, splits the
token sequence before its first transformer block, runs every block on its own chunk
(attention gathers keys and values), and all-gathers the sequence after its last block,
so the head sees the full sequence. Everything between the two points must be per-token
apart from attention, which is what the plan table asserts and the gate checks.

Only backbones listed in :data:`CP_PLANS` are split. HAT is not: its attention runs on
2-D windows interleaved with convolutions, so a sequence split has no meaning there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from spectramr.infrastructure.distributed.context_parallel.attention import (
    CP_ADOPTERS,
    TensorContextParallelAttention,
)
from spectramr.infrastructure.distributed.context_parallel.comms import (
    gather_sequence,
    split_sequence,
)
from spectramr.models.generators.dit_backbone import DiTBackbone
from spectramr.models.generators.uvit_backbone import UViTBackbone

#: Attribute on a split root holding its :class:`ContextParallelPlan`.
CP_PLAN_ATTR = "_spectramr_cp_plan"


@dataclass(frozen=True)
class TrunkPlan:
    """The block containers of a backbone's trunk, in execution order."""

    containers: tuple[str, ...]
    #: Tokens the trunk carries beyond the patch grid (U-ViT's time token).
    extra_tokens: int = 0


#: Backbone class -> its trunk. U-ViT's long skips stay inside the trunk, chunk-local.
CP_PLANS: dict[type[nn.Module], TrunkPlan] = {
    DiTBackbone: TrunkPlan(("blocks",)),
    UViTBackbone: TrunkPlan(("encoder", "middle", "decoder"), extra_tokens=1),
}


class ContextParallelError(RuntimeError):
    """The model cannot be, or was not correctly, split along its token sequence."""


@dataclass(frozen=True)
class ContextParallelPlan:
    """What :func:`parallelize_context` split: read by provenance."""

    group: Any
    size: int
    backbones: tuple[str, ...]
    attentions: int
    worst_relative_error: float
    #: Attentions that are also split by heads across the tensor group (tensor x context).
    tensor_split_attentions: int = 0


def _blocks(backbone: nn.Module, plan: TrunkPlan) -> list[nn.Module]:
    blocks: list[nn.Module] = []
    for name in plan.containers:
        container = backbone.get_submodule(name)
        blocks += list(container) if isinstance(container, nn.ModuleList) else [container]
    return blocks


def _require_identical(args: tuple, group: Any) -> None:
    """Every cp rank must enter the trunk with the same tokens and conditioning."""
    tensors = [a for a in args if torch.is_tensor(a) and a.is_floating_point()]
    stats = torch.stack([s for t in tensors for s in (t.double().sum(), t.double().square().sum())])
    lo, hi = stats.clone(), stats.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
    if not torch.allclose(lo, hi, rtol=1e-5, atol=1e-8):
        raise ContextParallelError(
            "context-parallel ranks entered the transformer trunk with different inputs. "
            "They must draw the same batch, timesteps and noise (keyed by data-parallel "
            "rank); a per-rank random stream splits one sample into unrelated chunks."
        )


class _TrunkHooks:
    """Split before the first block, gather after the last; checks inputs once."""

    def __init__(self, backbone: nn.Module, group: Any) -> None:
        self.backbone, self.group, self.checked = backbone, group, False

    def split(self, _module: nn.Module, args: tuple) -> tuple:
        if not args or not torch.is_tensor(args[0]):
            raise ContextParallelError("the first trunk block must take the tokens positionally")
        active = getattr(self.backbone, "_checkpointing_active", None)
        if active is not None and active():
            raise ContextParallelError(
                "gradient checkpointing would re-issue the context-parallel collectives "
                "during backward, which is not verified; disable it for this arm."
            )
        if not self.checked:
            _require_identical(args, self.group)
            self.checked = True
        return (split_sequence(args[0], self.group), *args[1:])

    def gather(self, _module: nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
        return gather_sequence(output, self.group)


def install(backbone: nn.Module, plan: TrunkPlan, group: Any) -> int:
    """Adopt the trunk's attentions and hook its ends, in place; return the attention count.

    A plain ``TokenAttention`` and a tensor-parallel one each have their own chunked form
    (:data:`CP_ADOPTERS`). A trunk with no attention to adopt raises: its hooks would
    split the sequence that no attention then gathers.
    """
    blocks = _blocks(backbone, plan)
    adopted = 0
    for block in blocks:
        for module in list(block.modules()):
            adopter = CP_ADOPTERS.get(type(module))
            if adopter is not None:
                adopter(module, group)
                adopted += 1
    if not adopted:
        raise ContextParallelError(
            f"{type(backbone).__name__}'s trunk has no attention context parallelism can "
            f"adopt ({sorted(c.__name__ for c in CP_ADOPTERS)})"
        )
    hooks = _TrunkHooks(backbone, group)
    blocks[0].register_forward_pre_hook(hooks.split)
    blocks[-1].register_forward_hook(hooks.gather)
    return adopted


def _targets(model: nn.Module) -> list[tuple[str, nn.Module, TrunkPlan]]:
    return [
        (name, module, CP_PLANS[type(module)])
        for name, module in model.named_modules()
        if type(module) in CP_PLANS
    ]


def parallelize_context(model: nn.Module, group: Any) -> ContextParallelPlan:
    """Split every planned backbone inside ``model`` along its token sequence, in place.

    Raises :class:`ContextParallelError` on every rank alike when nothing is planned,
    when replicas differ, or when a split backbone disagrees with its unsplit copy.
    """
    from spectramr.infrastructure.distributed.context_parallel.gate import verify_backbone
    from spectramr.infrastructure.distributed.tensor_parallel.plan import identical_across

    targets = _targets(model)
    if not targets:
        raise ContextParallelError(
            f"{type(model).__name__} has no context-parallel plan. Context parallelism "
            f"splits the token trunk of {sorted(c.__name__ for c in CP_PLANS)}; other "
            "models (HAT's windowed attention, convolutional backbones) have no sequence "
            "to split."
        )
    failures, worst = [], 0.0
    for name, backbone, plan in targets:
        if not identical_across(backbone, group):
            failures.append(f"{name or 'root'}: weights differ across context ranks")
            continue
        problems, error = verify_backbone(backbone, plan, group)
        failures += [f"{name or 'root'}: {p}" for p in problems]
        worst = max(worst, error)
    device = next(model.parameters()).device
    flag = torch.tensor([float(bool(failures))], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)  # world-wide, so every rank raises alike
    if flag.item():
        raise ContextParallelError(
            "context-parallel split does not reproduce the unsplit model: "
            + ("; ".join(failures[:5]) if failures else "on another rank")
        )
    attentions = sum(install(backbone, plan, group) for _, backbone, plan in targets)
    tensor_split = sum(
        type(m) is TensorContextParallelAttention for _, b, _ in targets for m in b.modules()
    )
    record = ContextParallelPlan(
        group=group,
        size=dist.get_world_size(group),
        backbones=tuple(name for name, *_ in targets),
        attentions=attentions,
        worst_relative_error=worst,
        tensor_split_attentions=tensor_split,
    )
    setattr(model, CP_PLAN_ATTR, record)
    return record


__all__ = [
    "CP_PLANS",
    "CP_PLAN_ATTR",
    "ContextParallelError",
    "ContextParallelPlan",
    "TrunkPlan",
    "install",
    "parallelize_context",
]
