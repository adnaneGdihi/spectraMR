"""The multi-stage strategy's ``pipeline`` checkpoint entry, read and refused in isolation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies import pipeline_stage_state as pss
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.pipeline_strategy import MultiTrainingStrategy


def _strategy() -> MultiTrainingStrategy:
    s = object.__new__(MultiTrainingStrategy)
    s.env = SimpleNamespace(opt_g=None)
    s.multi_stages = {"b": nn.Linear(2, 2), "a": nn.Linear(2, 2)}
    s._stage_optimizers = {"b": torch.optim.SGD(s.multi_stages["b"].parameters(), lr=0.1)}
    s._trainable_params = s._collect_trainable_params()
    return s


def _entry(**over) -> dict:
    return {
        "stage_optimizers": {"b": _strategy()._stage_optimizers["b"].state_dict()},
        "es_best": {"a": 0.5},
        "es_wait": {"a": 2},
        "frozen_by_early_stopping": [],
        "all_unfrozen": False,
    } | over


def test_the_base_strategy_owns_no_checkpoint_models() -> None:
    assert BaseTrainingStrategy.checkpoint_models(object.__new__(MultiTrainingStrategy)) == {}


def test_stage_models_are_named_and_sorted() -> None:
    s = _strategy()
    assert list(s.checkpoint_models()) == ["stage:a", "stage:b"]
    assert s.checkpoint_models()["stage:a"] is s.multi_stages["a"]


def test_a_strategy_before_its_first_epoch_end_captures_empty_early_stopping() -> None:
    entry = pss.capture_pipeline_state(_strategy())
    assert (entry["es_best"], entry["es_wait"], entry["frozen_by_early_stopping"]) == ({}, {}, [])
    assert entry["all_unfrozen"] is False


def test_a_strategy_state_without_the_pipeline_entry_is_refused() -> None:
    with pytest.raises(RuntimeError, match="#2962"):
        _strategy().load_strategy_state_dict({"modules": {}, "params": {}, "generators": {}})


@pytest.mark.parametrize("saved", [{}, {"a": {}, "b": {}}])
def test_a_stage_optimizer_set_that_differs_is_refused_before_any_change(saved) -> None:
    s = _strategy()
    with pytest.raises(RuntimeError, match="stage optimizers"):
        s.load_strategy_state_dict({"pipeline": _entry(stage_optimizers=saved, all_unfrozen=True)})
    assert not hasattr(s, "_es_best")
    assert not s._all_unfrozen


def test_an_early_stopped_stage_with_no_live_stage_is_refused() -> None:
    with pytest.raises(RuntimeError, match="'z'"):
        pss.check_pipeline_state(_strategy(), _entry(frozen_by_early_stopping=["z"]))


def test_restore_sets_the_counters_and_the_freeze() -> None:
    s = _strategy()
    pss.restore_pipeline_state(s, _entry(frozen_by_early_stopping=["a"]), load_optimizer=True)
    assert (s._es_best, s._es_wait, s._es_frozen) == ({"a": 0.5}, {"a": 2}, {"a"})
    assert not any(p.requires_grad for p in s.multi_stages["a"].parameters())
    assert {id(p) for p in s._trainable_params} == {id(p) for p in s.multi_stages["b"].parameters()}


def test_the_base_strategy_accepts_any_strategy_part() -> None:
    BaseTrainingStrategy.check_strategy_state(object.__new__(MultiTrainingStrategy), None)


def test_a_missing_strategy_part_is_refused_by_the_check() -> None:
    with pytest.raises(RuntimeError, match=r"'pipeline'.*#2962"):
        _strategy().check_strategy_state(None)


def test_the_optimizer_gate_off_keeps_the_stage_optimizer_fresh() -> None:
    s = _strategy()
    saved = {"state": {0: {"momentum_buffer": torch.ones(2, 2)}}, "param_groups": []}
    saved["param_groups"] = s._stage_optimizers["b"].state_dict()["param_groups"]
    pss.restore_pipeline_state(s, _entry(stage_optimizers={"b": saved}), load_optimizer=False)
    assert s._stage_optimizers["b"].state_dict()["state"] == {}
    assert s._es_wait == {"a": 2}
