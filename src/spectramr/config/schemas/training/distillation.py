"""``training.distillation``: the frozen teacher a distillation arm learns from."""

from __future__ import annotations

from pydantic import Field

from ..strictness import StrictSchema


class DistillationTrainingConfigSchema(StrictSchema):
    """Read by ``ConcreteDistillationStrategy._load_teacher``."""

    teacher_checkpoint: str | None = Field(
        default=None,
        description="Checkpoint of the frozen teacher (a run's best or step file). Its "
        "generator part is loaded with weights(policy='overlap'); unset runs the arm "
        "without the distillation term.",
    )


__all__ = ["DistillationTrainingConfigSchema"]
