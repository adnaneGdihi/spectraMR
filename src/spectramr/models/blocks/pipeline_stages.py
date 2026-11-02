"""Splitting an unrolled cascade into contiguous pipeline stages.

A model that can be pipelined exposes ``pipeline_stages(n) -> list[CascadeStage]``.
Each stage is built from the model's own submodules under their original attribute
paths (``regularisers.3`` stays ``regularisers.3``), so the stages' state dicts are
disjoint subsets of the unsplit model's keys and a pipelined checkpoint loads into
the unsplit model unchanged.

Every call goes through ``forward`` so a sharding wrapper's hooks see it:

* ``stage("enter", x, **kwargs)`` -- first stage only, on the model's own forward
  arguments; returns the payload.
* ``stage("cascade", *payload)`` -- this stage's cascades; returns the payload.
* ``stage("exit", *payload)`` -- last stage only; returns the model's output.

The payload is a tuple whose entry 0 is the running estimate. Every other entry is
context every data-consistency step needs (measured k-space, mask, maps, an
embedding): cascades read it and pass it on by identity. :class:`CascadeStage`
enforces that, so a stage that drops or replaces the mask raises instead of
quietly running a cascade without data consistency.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn

from spectramr.models.blocks.grad_checkpointing import GradCheckpointingMixin

__all__ = [
    "CascadeStage",
    "Payload",
    "attach_submodules",
    "balanced_partition",
    "build_cascade_stages",
    "cascade_at",
    "run_pipeline_stages",
]

Payload = tuple[Any, ...]
_PHASES = ("enter", "cascade", "exit")


def balanced_partition(count: int, parts: int) -> list[range]:
    """``parts`` contiguous ranges over ``range(count)`` whose sizes differ by at most one."""
    if parts < 1 or parts > count:
        raise ValueError(
            f"cannot split {count} cascade(s) into {parts} pipeline stage(s); "
            f"parallel.mesh.pipeline must be between 1 and {count}."
        )
    base, extra = divmod(count, parts)
    bounds, start = [], 0
    for index in range(parts):
        stop = start + base + (1 if index < extra else 0)
        bounds.append(range(start, stop))
        start = stop
    return bounds


def attach_submodules(owner: nn.Module, model: nn.Module, paths: Iterable[str]) -> None:
    """Register ``model``'s submodule at each dotted ``path`` on ``owner`` at the same path.

    Intermediate containers are ``nn.ModuleDict``, so ``"blocks.3"`` lands as
    ``owner.blocks["3"]`` and its state-dict keys read ``blocks.3.<...>`` as in ``model``.
    """
    for path in paths:
        parts = path.split(".")
        node: nn.Module = owner
        for part in parts[:-1]:
            child = node._modules.get(part)
            if child is None:
                child = nn.ModuleDict()
                node.add_module(part, child)
            node = child
        node.add_module(parts[-1], model.get_submodule(path))


def cascade_at(container: nn.Module, index: int) -> nn.Module:
    """Cascade ``index`` of a ``ModuleList`` (the model) or a ``ModuleDict`` (a stage)."""
    return container[str(index)] if isinstance(container, nn.ModuleDict) else container[index]


class CascadeStage(GradCheckpointingMixin, nn.Module):
    """One contiguous run of cascades. Subclasses implement ``_enter``/``_cascade``/``_exit``."""

    def __init__(self, cascades: range, *, first: bool, last: bool, grad_checkpointing: bool):
        super().__init__()
        self.cascades = cascades
        self.is_first = first
        self.is_last = last
        self.grad_checkpointing = grad_checkpointing

    def forward(self, phase: str, *args: Any, **kwargs: Any) -> Any:
        if phase == "cascade":
            payload = tuple(args)
            out = tuple(self._cascade(payload))
            _check_context_passed_through(type(self).__name__, payload, out)
            return out
        if phase == "enter":
            if not self.is_first:
                raise RuntimeError(f"{type(self).__name__}: only the first stage can enter.")
            return tuple(self._enter(*args, **kwargs))
        if phase == "exit":
            if not self.is_last:
                raise RuntimeError(f"{type(self).__name__}: only the last stage can exit.")
            return self._exit(tuple(args))
        raise ValueError(f"unknown pipeline phase {phase!r}; expected one of {_PHASES}.")

    def batched_entries(self, payload: Payload, batch: int) -> tuple[bool, ...]:
        """Which payload entries carry the batch dimension (split into microbatches).

        Default: a tensor whose leading extent equals the batch. A model whose context
        can be unbatched with a coincidentally equal extent overrides this.
        """
        return tuple(
            isinstance(t, torch.Tensor) and t.ndim >= 1 and t.shape[0] == batch for t in payload
        )

    def _enter(self, *args: Any, **kwargs: Any) -> Payload:
        raise NotImplementedError

    def _cascade(self, payload: Payload) -> Payload:
        raise NotImplementedError

    def _exit(self, payload: Payload) -> torch.Tensor:
        return payload[0]


def _check_context_passed_through(name: str, before: Payload, after: Payload) -> None:
    same_arity = len(after) == len(before)
    if not same_arity or any(a is not b for a, b in zip(before[1:], after[1:], strict=True)):
        raise RuntimeError(
            f"{name} changed the pipeline context: every entry after the estimate (measured "
            "k-space, mask, maps, embedding) must pass through each stage unchanged, or the "
            "next stage runs its data consistency without it."
        )


def build_cascade_stages(
    model: nn.Module,
    parts: int,
    count: int,
    stage_cls: type[CascadeStage],
    paths_for: Any,
    **stage_kwargs: Any,
) -> list[CascadeStage]:
    """Partition ``count`` cascades into ``parts`` stages of ``stage_cls``.

    ``paths_for(cascades, first, last)`` names the submodules a stage owns. The union of
    the stages' state-dict keys must equal the model's, each key owned once.
    """
    stages: list[CascadeStage] = []
    ranges = balanced_partition(count, parts)
    for index, cascades in enumerate(ranges):
        first, last = index == 0, index == len(ranges) - 1
        stage = stage_cls(
            cascades,
            first=first,
            last=last,
            grad_checkpointing=bool(getattr(model, "grad_checkpointing", False)),
            **stage_kwargs,
        )
        attach_submodules(stage, model, paths_for(cascades, first, last))
        stages.append(stage)
    _check_keys_partition(model, stages)
    return stages


def _check_keys_partition(model: nn.Module, stages: Sequence[nn.Module]) -> None:
    owned: dict[str, int] = {}
    for index, stage in enumerate(stages):
        for key in stage.state_dict():
            if key in owned:
                raise RuntimeError(f"pipeline stages {owned[key]} and {index} both own {key!r}.")
            owned[key] = index
    missing = sorted(set(model.state_dict()) - set(owned))
    extra = sorted(set(owned) - set(model.state_dict()))
    if missing or extra:
        raise RuntimeError(
            f"{type(model).__name__}.pipeline_stages does not partition the model: "
            f"unowned {missing[:5]}, foreign {extra[:5]}."
        )


def run_pipeline_stages(stages: Sequence[nn.Module], x: Any, **kwargs: Any) -> Any:
    """The stages composed on one process; equals the unsplit forward."""
    payload = stages[0]("enter", x, **kwargs)
    for stage in stages:
        payload = stage("cascade", *payload)
    return stages[-1]("exit", *payload)
