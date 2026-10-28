"""Tests for the TALL-SR training sub-schema (``training.tall_sr``)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.config.schemas.training.tall_sr import TALLSRTrainingConfigSchema


def test_default_locality_weight_is_zero() -> None:
    assert TALLSRTrainingConfigSchema().lambda_locality == 0.0


def test_rejects_negative_locality_weight() -> None:
    with pytest.raises(ValidationError):
        TALLSRTrainingConfigSchema(lambda_locality=-0.1)


def test_forbids_unknown_key() -> None:
    with pytest.raises(ValidationError):
        TALLSRTrainingConfigSchema(lambda_locality=0.01, lambda_typo=1.0)


def test_attached_to_training_strategy_schema() -> None:
    assert "tall_sr" in TrainingStrategyConfigSchema.model_fields
