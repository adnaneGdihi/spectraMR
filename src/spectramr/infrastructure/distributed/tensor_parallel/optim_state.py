"""Optimizer state of a tensor-parallel model, in the unsplit model's shapes.

The model's state-dict hooks make a checkpoint's weights unsplit, but an optimizer's
per-parameter state (Adam's moments) is shaped like each rank's slice. Saved as-is,
rank 0 writes only its own tensor rank's moments and every tensor rank resumes from
them. These helpers gather each split parameter's state tensors over the tensor group
on save and slice them on load, with the same transforms as the weights.

``state`` maps a parameter's dotted name to its per-parameter state dict. Gathering is
collective: every rank of the tensor group calls it, in the same order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.distributed.tensor_parallel.plan import TP_PLAN_ATTR

__all__ = [
    "gather_split_optimizer_state",
    "has_tensor_parallel_plan",
    "rewrite_packed_optimizer_state",
    "slice_split_optimizer_state",
]


def has_tensor_parallel_plan(model: Any) -> bool:
    return getattr(unwrap_model(model), TP_PLAN_ATTR, None) is not None


def _split_entries(model: Any) -> list[tuple[str, Any, str, torch.Size]]:
    root = unwrap_model(model)
    plan = getattr(root, TP_PLAN_ATTR)
    entries = []
    for block in plan.blocks:
        module = root.get_submodule(block)
        for key in module.tp_sharding.slices:
            name = f"{block}.{key}".lstrip(".")
            entries.append((name, module.tp_sharding, key, module.get_parameter(key).shape))
    return entries


def _rewrite(model: Any, state: dict[str, dict], fn: Callable, *, local: bool) -> None:
    for name, sharding, key, local_shape in _split_entries(model):
        entry = state.get(name)
        if not entry:
            continue
        rewritten = dict(entry)  # never mutate the optimizer's live per-parameter dict
        for field, value in entry.items():
            # Per-parameter tensors only (``step`` is a scalar), in the shape being replaced.
            if torch.is_tensor(value) and value.dim() > 0 and (value.shape == local_shape) is local:
                rewritten[field] = fn(sharding, key, value)
        state[name] = rewritten


def gather_split_optimizer_state(model: Any, state: dict[str, dict]) -> None:
    """Replace each split parameter's slice-shaped state tensors by unsplit ones, in place."""
    _rewrite(model, state, lambda sharding, key, t: sharding.full(key, t), local=True)


def slice_split_optimizer_state(model: Any, state: dict[str, dict]) -> None:
    """Replace each split parameter's unsplit state tensors by this rank's slice, in place."""
    _rewrite(model, state, lambda sharding, key, t: sharding.local(key, t), local=False)


def rewrite_packed_optimizer_state(
    model: Any, optimizer: Any, packed: dict, *, gather: bool
) -> dict:
    """:func:`gather_split_optimizer_state` / slice for ``optimizer.state_dict()``'s form.

    That form keys ``state`` by parameter index; the index order is the order of
    ``optimizer.param_groups``. Returns a new dict; ``packed`` is left unchanged.
    """
    names = {id(p): n for n, p in unwrap_model(model).named_parameters()}
    params = [p for group in optimizer.param_groups for p in group["params"]]
    indices = [i for group in packed["param_groups"] for i in group["params"]]
    name_of = {i: names[id(p)] for i, p in zip(indices, params, strict=True) if id(p) in names}
    by_name = {name_of[i]: entry for i, entry in packed["state"].items() if i in name_of}
    (gather_split_optimizer_state if gather else slice_split_optimizer_state)(model, by_name)
    state = {i: by_name.get(name_of.get(i), entry) for i, entry in packed["state"].items()}
    return {**packed, "state": state}
