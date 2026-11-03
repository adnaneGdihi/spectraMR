"""The census helper must turn red on a strategy that calls a model the hooks cannot see."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.utils.snapshot_census import records_a_model_stage, traced_step

pytestmark = pytest.mark.unit


def _strategy(models: dict) -> SimpleNamespace:
    return SimpleNamespace(env=SimpleNamespace(models=models), snapshot_session=None)


def test_a_model_called_through_env_models_is_seen() -> None:
    net = nn.Conv2d(1, 1, 1)
    s = _strategy({"generator": net})
    trace = traced_step(s, lambda: net(torch.ones(1, 1, 4, 4)))
    assert records_a_model_stage(trace)
    assert s.snapshot_session is None, "the census restores the strategy"


def test_planted_a_private_module_is_flagged() -> None:
    """The violation the gate exists for: the strategy calls a module it never registered."""
    hidden = nn.Conv2d(1, 1, 1)
    s = _strategy({"generator": nn.Conv2d(1, 1, 1)})
    trace = traced_step(s, lambda: hidden(torch.ones(1, 1, 4, 4)))
    assert not records_a_model_stage(trace)


def test_a_module_registered_through_capture_is_seen() -> None:
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

    own = nn.Conv2d(1, 1, 1)
    s = _strategy({})

    def run():
        with BaseTrainingStrategy._capture_model_output(s, module=own):
            own(torch.ones(1, 1, 4, 4))

    assert records_a_model_stage(traced_step(s, run))
