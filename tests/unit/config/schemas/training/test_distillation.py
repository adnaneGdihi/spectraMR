import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.distillation import DistillationTrainingConfigSchema


def test_teacher_checkpoint_defaults_to_none() -> None:
    assert DistillationTrainingConfigSchema().teacher_checkpoint is None


def test_unknown_keys_are_refused() -> None:
    with pytest.raises(ValidationError):
        DistillationTrainingConfigSchema(teacher=1)


def test_it_mounts_at_training_distillation() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema

    assert "distillation" in TrainingStrategyConfigSchema.model_fields
