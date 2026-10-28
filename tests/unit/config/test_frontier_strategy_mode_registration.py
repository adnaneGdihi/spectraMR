"""Consistency test for the PR-4/5/7/9 strategy training-mode registration.

A new strategy key must be registered in STRATEGY_CLASS_PATHS (dispatch) and
VALID_TRAINING_MODES (reference list). This guards that the frontier strategies
stay consistent across both.
"""

from __future__ import annotations

import pytest

from spectramr.config.validation_constants import VALID_TRAINING_MODES
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

_FRONTIER_MODES = [
    "bloch_schrodinger_bridge",
    "i2sb",
    "vf_consistency_distillation",
    "universal_reconstruction",
    "flow_matching_pfode",
]


@pytest.mark.parametrize("mode", _FRONTIER_MODES)
def test_mode_in_strategy_class_paths(mode: str) -> None:
    assert mode in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


@pytest.mark.parametrize("mode", _FRONTIER_MODES)
def test_mode_in_valid_training_modes(mode: str) -> None:
    assert mode in VALID_TRAINING_MODES
