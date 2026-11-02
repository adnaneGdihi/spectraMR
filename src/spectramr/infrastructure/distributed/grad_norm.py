"""The global gradient norm of a model, and clipping by it, under every parallel layout.

Replicated and FSDP2 models need nothing special: ``clip_grad_norm_`` over FSDP2's
DTensor gradients already reduces over the data-parallel shards. A tensor-parallel
model does: each tensor rank holds a different slice of the split parameters, so their
squared norms are summed over the tensor group, while the replicated parameters (norms,
adaLN, biases after a reduce) carry identical gradients on every tensor rank and are
counted once. Without that every tensor rank clips by its own partial norm, and the
replicated weights drift apart across the group.

A pipelined model holds one stage per pipeline rank, every parameter on exactly one
of them, so the squared stage norms are summed over the pipeline group; within a
stage the data-parallel reduction is the one above (FSDP2's, or none for replicas
whose gradients were already averaged).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from spectramr.core.module_utils import unwrap_model

__all__ = ["clip_grad_norm_global_"]


def _plain(norm: torch.Tensor) -> torch.Tensor:
    return norm.full_tensor() if hasattr(norm, "full_tensor") else norm


def _tensor_parallel_plan(model: Any) -> Any:
    root = unwrap_model(model)
    from spectramr.infrastructure.distributed.tensor_parallel.plan import TP_PLAN_ATTR

    return root, getattr(root, TP_PLAN_ATTR, None)


def clip_grad_norm_global_(model: torch.nn.Module, max_norm: float) -> torch.Tensor:
    """Clip ``model``'s gradients by their global 2-norm and return that norm.

    ``max_norm=inf`` measures without clipping. The norm is a plain device tensor,
    identical on every rank, and is never copied to the host here.
    """
    root, plan = _tensor_parallel_plan(model)
    from spectramr.infrastructure.distributed.pipeline_parallel.module import PP_GROUP_ATTR

    pipeline_group = getattr(root, PP_GROUP_ATTR, None)
    if pipeline_group is not None:
        # The schema refuses pipeline x tensor, so a pipelined model has no tensor plan.
        return _clip_over_stages(root, max_norm, pipeline_group)
    if plan is None:
        return _plain(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm))

    import torch.distributed as dist

    split_ids = {id(root.get_parameter(name)) for name in plan.sharded}
    params = [p for p in root.parameters() if p.grad is not None]
    split = [p.grad for p in params if id(p) in split_ids]
    whole = [p.grad for p in params if id(p) not in split_ids]
    split_sq = _plain(torch.nn.utils.get_total_norm(split)).square()
    whole_sq = _plain(torch.nn.utils.get_total_norm(whole)).square()
    dist.all_reduce(split_sq, group=plan.group)
    total = (split_sq + whole_sq.to(split_sq.device)).sqrt()
    if math.isfinite(max_norm):
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total)
    return total


def _clip_over_stages(root: torch.nn.Module, max_norm: float, group: Any) -> torch.Tensor:
    import torch.distributed as dist

    params = [p for p in root.parameters() if p.grad is not None]
    squared = _plain(torch.nn.utils.get_total_norm([p.grad for p in params])).square()
    dist.all_reduce(squared, group=group)
    total = squared.sqrt()
    if math.isfinite(max_norm):
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total)
    return total
