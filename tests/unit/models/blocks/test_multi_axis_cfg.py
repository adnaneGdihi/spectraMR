"""Tests for MultiAxisCFGMixin's batched prediction hook."""

from __future__ import annotations

import torch

from spectramr.models.blocks.multi_axis_cfg import MultiAxisCFGMixin


class _Linear(MultiAxisCFGMixin):
    def __init__(self):
        super().__init__()
        self.sequential_calls = 0

    def _predict_noise(self, x, conditions):
        self.sequential_calls += 1
        return x + sum(v for v in conditions.values() if v is not None)


class _Batched(_Linear):
    def __init__(self):
        super().__init__()
        self.batched_calls = 0

    def _predict_noise_many(self, x, condition_sets):
        self.batched_calls += 1
        return super()._predict_noise_many(x, condition_sets)


def _conditions():
    return {"a": torch.full((2, 1), 1.0), "b": torch.full((2, 1), 4.0)}


def test_the_combination_is_unchanged_by_the_hook():
    x = torch.zeros(2, 1)
    out = _Linear().cfg_sample_step(x, _conditions(), {"a": 2.0, "b": 0.5})
    assert torch.allclose(out, torch.full_like(out, 2.0 * 1.0 + 0.5 * 4.0))


def test_all_predictions_go_through_one_hook_call():
    model = _Batched()
    model.cfg_sample_step(torch.zeros(2, 1), _conditions(), {"a": 2.0, "b": 0.5})
    assert model.batched_calls == 1
    assert model.sequential_calls == 3  # null + two axes, via the default loop


def test_zero_weight_and_missing_axes_are_not_evaluated():
    model = _Linear()
    conditions = {"a": torch.ones(2, 1), "b": None}
    model.cfg_sample_step(torch.zeros(2, 1), conditions, {"a": 0.0, "b": 1.0})
    assert model.sequential_calls == 1  # the null pass only
