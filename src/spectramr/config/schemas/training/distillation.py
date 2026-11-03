"""``training.distillation``: the frozen teacher a distillation arm learns from."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..strictness import StrictSchema


class DistillationTrainingConfigSchema(StrictSchema):
    """Read by ``ConcreteDistillationStrategy`` (teacher load and ``validation_step``)."""

    teacher_checkpoint: str | None = Field(
        default=None,
        description="Checkpoint of the frozen teacher (a run's best or step file). Its "
        "generator part is loaded with weights(policy='overlap'); unset runs the arm "
        "without the distillation term.",
    )
    validation_input: Literal["digital_twin", "loaded"] = Field(
        default="digital_twin",
        description="What the student reads at validation. 'digital_twin' corrupts the loaded "
        "target with the twin and scores against that target; 'loaded' feeds the loaded input "
        "scan unchanged (real corruption, no twin) and scores against the loaded target, which "
        "must then be a different (clean) image or the step raises.",
    )


__all__ = ["DistillationTrainingConfigSchema"]
