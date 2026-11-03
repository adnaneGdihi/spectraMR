"""Checkpoints of a pipelined generator, in the unsplit model's keys.

Each stage's state-dict keys are the unsplit model's keys for the cascades it owns
(``pipeline_stages`` builds stages under the original paths), so consolidation is a
union. Every stage reads its full state (an FSDP2 stage gathers over its
data-parallel group first), the pipeline group gathers the pieces to its first
stage, and the writer (global rank 0) holds the whole model: ``spectramr infer``
loads the file into the unsplit model. Optimizer state is keyed by parameter name
(the distributed-checkpoint form), so it consolidates the same way.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Mapping
from typing import Any

import torch.distributed as dist

from spectramr.core.module_utils import strip_wrapper_prefixes
from spectramr.infrastructure.distributed.pipeline_parallel.module import (
    PipelinedGenerator,
    pipelined_of,
)

__all__ = ["PipelineCheckpointAdapter", "PipelineStateIO"]


def _options() -> Any:
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    # No cpu_offload: its rank-0-only gather would leave every stage but the first empty.
    return StateDictOptions(full_state_dict=True, cpu_offload=False)


def _to_cpu(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    return value.cpu() if hasattr(value, "cpu") else value


def _gather_to_first(pipelined: PipelinedGenerator, local: Any) -> list[Any] | None:
    pieces: list[Any] | None = [None] * len(pipelined.pp_ranks) if pipelined.is_first else None
    dist.gather_object(local, pieces, dst=pipelined.pp_ranks[0], group=pipelined.group)
    return pieces


class PipelineStateIO:
    """Wraps the strategy's own state IO; a model that is not pipelined passes through."""

    def __init__(self, inner: Any):
        self.inner = inner

    def model_state(self, model: Any) -> dict[str, Any]:
        pipelined = pipelined_of(model)
        if pipelined is None:
            return self.inner.model_state(model)
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        local = _to_cpu(get_model_state_dict(pipelined.stage, options=_options()))
        pieces = _gather_to_first(pipelined, local)
        return {k: v for piece in pieces for k, v in piece.items()} if pieces else {}

    def optimizer_state(self, model: Any, optimizer: Any) -> dict[str, Any]:
        pipelined = pipelined_of(model)
        if pipelined is None:
            return self.inner.optimizer_state(model, optimizer)
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        local = _to_cpu(get_optimizer_state_dict(pipelined.stage, optimizer, options=_options()))
        pieces = _gather_to_first(pipelined, local)
        if not pieces:
            return {}
        merged: dict[str, Any] = {"state": {}, "param_groups": []}
        for piece in pieces:
            merged["state"].update(piece["state"])
            for index, group in enumerate(piece["param_groups"]):
                if index == len(merged["param_groups"]):
                    merged["param_groups"].append({**group, "params": []})
                merged["param_groups"][index]["params"] += list(group["params"])
        return merged

    def load_model_state(self, model: Any, state: dict[str, Any]) -> None:
        pipelined = pipelined_of(model)
        if pipelined is None:
            return self.inner.load_model_state(model, state)
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        state = strip_wrapper_prefixes(state)
        own = set(pipelined.stage.state_dict())
        missing = sorted(own - set(state))
        if missing:
            raise KeyError(f"checkpoint lacks this pipeline stage's keys {missing[:5]}.")
        set_model_state_dict(
            pipelined.stage, {k: v for k, v in state.items() if k in own}, options=_options()
        )

    def load_optimizer_state(self, model: Any, optimizer: Any, state: dict[str, Any]) -> None:
        pipelined = pipelined_of(model)
        if pipelined is None:
            return self.inner.load_optimizer_state(model, optimizer, state)
        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

        own = {name for name, _ in pipelined.stage.named_parameters()}
        groups = [
            {**group, "params": [p for p in group["params"] if p in own]}
            for group in state["param_groups"]
        ]
        mine = {"state": {k: v for k, v in state["state"].items() if k in own},
                "param_groups": groups}  # fmt: skip
        set_optimizer_state_dict(pipelined.stage, optimizer, mine, options=_options())

    @contextlib.contextmanager
    def swap_in(self, model: Any, shadow: Any) -> Generator[None]:
        if pipelined_of(model) is not None:
            raise RuntimeError("EMA is refused under pipeline parallelism; nothing to swap in.")
        with self.inner.swap_in(model, shadow):
            yield


class PipelineCheckpointAdapter:
    """The strategy's adapter, with a state IO that consolidates the pipeline stages."""

    name = "pipeline"
    #: Gathering the stages is a collective over the pipeline group.
    requires_all_ranks = True
    writes_native_artifact = False

    def __init__(self, inner: Any):
        self.inner = inner
        self.state_io = PipelineStateIO(inner.state_io)

    def full_model_state(self, model: Any) -> dict[str, Any]:
        return self.state_io.model_state(model)

    def engine_owned_keys(self, models: Mapping[str, Any]) -> frozenset[str]:
        return self.inner.engine_owned_keys(models)

    def ema_is_whole(self, models: Mapping[str, Any]) -> bool:
        return self.inner.ema_is_whole(models)

    def save_native(self, **kwargs: Any) -> Any:
        return self.inner.save_native(**kwargs)

    def load_native(self, **kwargs: Any) -> Any:
        return self.inner.load_native(**kwargs)

    def native_tag_roots(self, **kwargs: Any) -> tuple[Any, ...]:
        return self.inner.native_tag_roots(**kwargs)
