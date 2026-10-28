"""Unit tests for the equivariance-conformal schema (B-1 / Idea 1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.equivariance_conformal import (
    EquivarianceConformalConfig,
    TrainingConfigEquivarianceConformal,
)


def test_defaults():
    cfg = EquivarianceConformalConfig()
    assert cfg.alpha == 0.1
    assert cfg.num_rotations_K == 8
    assert cfg.group == "SO2"
    assert cfg.exchangeability_test == "ks_test"


def test_extra_forbidden():
    with pytest.raises(ValidationError):
        EquivarianceConformalConfig(bogus_field=1)


def test_frozen_immutable():
    cfg = EquivarianceConformalConfig()
    with pytest.raises(ValidationError):
        cfg.alpha = 0.2  # type: ignore[misc]


def test_alpha_bounds():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValidationError):
            EquivarianceConformalConfig(alpha=bad)


def test_group_literal_enforced():
    with pytest.raises(ValidationError):
        EquivarianceConformalConfig(group="SO3")


def test_num_rotations_lower_bound():
    with pytest.raises(ValidationError):
        EquivarianceConformalConfig(num_rotations_K=0)


def test_top_level_mounts_subblock():
    top = TrainingConfigEquivarianceConformal(equivariance_conformal={})
    assert top.equivariance_conformal.num_rotations_K == 8
    assert top.equivariance_conformal.group == "SO2"


def test_top_level_allows_extra_strategy_fields():
    # BaseTrainingConfigSchema is extra="allow"; arbitrary sibling keys are OK.
    top = TrainingConfigEquivarianceConformal(
        equivariance_conformal={},
        strategy_class="some.dotted.Path",
    )
    assert top.equivariance_conformal.alpha == 0.1
