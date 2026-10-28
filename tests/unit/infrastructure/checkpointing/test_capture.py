"""capture(): the live pipeline as one RunState, read the way the director read it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from spectramr.infrastructure.checkpointing import capture as capture_mod
from spectramr.infrastructure.checkpointing.capture import (
    collect_models,
    join_ema,
    resolve_scaler,
    split_ema,
)
from spectramr.infrastructure.checkpointing.payload import validate
from spectramr.infrastructure.distributed.checkpoint_adapters import DefaultCheckpointAdapter
from spectramr.infrastructure.distributed.state_io import PassthroughStateIO
from spectramr.infrastructure.optimization.ema import ModelEma
from tests.unit.infrastructure.checkpointing.conftest import FakeEarlyStopping, make_pipeline
from tests.unit.infrastructure.checkpointing.conftest import capture_with as _capture


class TestResolveScaler:
    def test_the_strategys_live_scaler_wins(self) -> None:
        live, passed = object(), object()
        strategy = SimpleNamespace(amp_helper=SimpleNamespace(scaler=live))
        assert resolve_scaler(SimpleNamespace(scaler=passed), strategy) is live

    def test_without_a_strategy_the_pipeline_scaler_is_used(self) -> None:
        passed = object()
        assert resolve_scaler(SimpleNamespace(scaler=passed), None) is passed

    def test_no_amp_means_no_scaler(self) -> None:
        assert resolve_scaler(SimpleNamespace(scaler=None), SimpleNamespace()) is None


def test_collect_models_passes_wrappers_through() -> None:
    wrapped = nn.DataParallel(nn.Linear(2, 2))
    models = collect_models(SimpleNamespace(generator=wrapped, discriminator=None))
    assert models == {"generator": wrapped}


def test_no_runtime_uses_the_default_adapter() -> None:
    resolved: list[object] = []
    real = capture_mod.resolve_checkpoint_adapter

    def spy(parallel):
        resolved.append(real(parallel))
        return resolved[-1]

    with patch.object(capture_mod, "resolve_checkpoint_adapter", side_effect=spy):
        _capture(make_pipeline())
    assert [type(a) for a in resolved] == [DefaultCheckpointAdapter]


def test_state_is_read_through_the_adapter_in_collective_order() -> None:
    calls: list[str] = []

    class Recording(DefaultCheckpointAdapter):
        def full_model_state(self, model):
            calls.append(f"model:{type(model).__name__}:{model.out_features}")
            return super().full_model_state(model)

    adapter = Recording()
    original = adapter.state_io.optimizer_state

    def optimizer_state(model, optimizer):
        calls.append(f"optim:{model.out_features}")
        return original(model, optimizer)

    adapter.state_io = SimpleNamespace(
        optimizer_state=optimizer_state, model_state=PassthroughStateIO().model_state
    )
    _capture(make_pipeline(), parallel=SimpleNamespace(strategy="ddp", checkpoint_adapter=adapter))
    assert calls == ["model:Linear:2", "model:Linear:1", "optim:2", "optim:1"]


class _StagedStrategy:
    """A strategy owning two stage models, declared out of order."""

    def __init__(self, **models: nn.Module) -> None:
        self.models = models

    def checkpoint_models(self) -> dict[str, nn.Module]:
        return dict(self.models)


def test_strategy_models_are_gathered_sorted_after_the_pipeline_models() -> None:
    calls: list[str] = []

    class Recording(DefaultCheckpointAdapter):
        def full_model_state(self, model):
            calls.append(f"model:{model.out_features}")
            return super().full_model_state(model)

    strategy = _StagedStrategy(**{"stage:z": nn.Linear(2, 7), "stage:a": nn.Linear(2, 5)})
    parallel = SimpleNamespace(strategy="ddp", checkpoint_adapter=Recording())
    state = _capture(make_pipeline(), strategy=strategy, parallel=parallel)
    assert calls == ["model:2", "model:1", "model:5", "model:7"]
    assert list(state["models"]) == ["generator", "discriminator", "stage:a", "stage:z"]


def test_a_strategy_model_shadowing_a_pipeline_model_raises() -> None:
    with pytest.raises(ValueError, match="collide"):
        _capture(make_pipeline(), strategy=_StagedStrategy(generator=nn.Linear(2, 2)))


def test_wrapped_models_are_saved_bare() -> None:
    pipe = make_pipeline()
    pipe.generator = nn.DataParallel(pipe.generator)
    state = _capture(pipe)
    assert all(not k.startswith("module.") for k in state["models"]["generator"])


def test_ema_counter_is_split_out_of_the_shadow() -> None:
    state = _capture(make_pipeline())
    assert ModelEma._NUM_UPDATES_KEY not in state["ema"]["generator"]
    assert state["ema"]["num_updates"] == 1


def test_ema_keys_are_bare() -> None:
    state = _capture(make_pipeline())
    assert set(state["ema"]["generator"]) == {"weight", "bias"}


def test_split_and_join_are_inverse() -> None:
    shadow = {"weight": torch.ones(1), ModelEma._NUM_UPDATES_KEY: torch.tensor(7)}
    assert join_ema(split_ema(dict(shadow)))[ModelEma._NUM_UPDATES_KEY].item() == 7


def test_rng_is_in_step_and_best_files() -> None:
    for role, reason, best in (
        ("step", "periodic", None),
        ("best", "best", {"name": "val_psnr", "value": 21.5}),
    ):
        state = _capture(make_pipeline(), role=role, reason=reason, best_metric=best)
        assert set(state["rng"]) >= {"torch", "numpy", "python"}


def test_scaler_state_is_saved_when_the_run_scales() -> None:
    state = _capture(make_pipeline(scaler=True))
    assert "scale" in state["scaler"]


def test_scaler_state_is_absent_without_a_scaler() -> None:
    assert "scaler" not in _capture(make_pipeline())


def test_early_stopping_state_is_captured() -> None:
    state = _capture(make_pipeline(), early_stopping=FakeEarlyStopping())
    assert state["early_stopping"]["wait_count"] == 1


def test_best_file_records_topology() -> None:
    state = _capture(
        make_pipeline(),
        role="best",
        reason="best",
        best_metric={"name": "val_psnr", "value": 21.5},
    )
    assert state["meta"]["topology"] == {"world_size": 1, "parallel_strategy": "none"}


def test_the_payload_validates_and_carries_both_optimizers() -> None:
    state = validate(_capture(make_pipeline()), source="test")
    assert set(state["optim"]) == {"generator", "discriminator"}
    assert set(state["models"]) == {"generator", "discriminator"}
    assert set(state["sched"]) == {"generator"}


class TestStrategyState:
    def test_owned_state_is_its_own_part(self) -> None:
        strategy = SimpleNamespace(strategy_state_dict=lambda: {"params": {"s": torch.ones(1)}})
        state = _capture(make_pipeline(), strategy=strategy)
        assert state["strategy"]["params"]["s"].item() == 1.0
        assert "s" not in state["models"]["generator"]

    def test_no_strategy_means_no_part(self) -> None:
        assert "strategy" not in _capture(make_pipeline())

    def test_a_strategy_owning_nothing_writes_no_empty_part(self) -> None:
        strategy = SimpleNamespace(strategy_state_dict=lambda: {"modules": {}, "params": {}})
        assert "strategy" not in _capture(make_pipeline(), strategy=strategy)


class TestEmaCounter:
    def test_split_without_a_counter_raises(self) -> None:
        with pytest.raises(KeyError):
            split_ema({"weight": torch.ones(1)})

    def test_join_without_a_counter_raises(self) -> None:
        with pytest.raises(KeyError):
            join_ema({"generator": {"weight": torch.ones(1)}})

    def test_split_leaves_the_callers_dict_alone(self) -> None:
        shadow = {"weight": torch.ones(1), ModelEma._NUM_UPDATES_KEY: torch.tensor(3)}
        assert split_ema(shadow)["num_updates"] == 3
        assert ModelEma._NUM_UPDATES_KEY in shadow
