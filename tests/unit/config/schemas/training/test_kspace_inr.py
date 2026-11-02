"""``training.kspace_inr``: the per-slice fit's knobs are bounded and closed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.config.schemas.training.kspace_inr import KSpaceINRTrainingConfigSchema


def test_the_defaults_fit_at_least_one_step() -> None:
    cfg = KSpaceINRTrainingConfigSchema()
    assert cfg.inner_steps >= 1 and cfg.inner_lr > 0 and cfg.first_order is True


@pytest.mark.parametrize(
    "bad", [{"inner_steps": 0}, {"inner_lr": -1.0}, {"lambda_dc": -0.1}, {"steps": 3}]
)
def test_an_out_of_range_or_unknown_knob_raises(bad: dict) -> None:
    with pytest.raises(ValidationError):
        KSpaceINRTrainingConfigSchema(**bad)


def test_it_is_mounted_on_the_training_block() -> None:
    field = TrainingStrategyConfigSchema.model_fields["kspace_inr"]
    assert field.default is None
