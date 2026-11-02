"""Motion Training Configuration Schema.

This module defines the schema for Motion Meta-Learning experiments
(Stage 1 of motion correction pipeline).
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .base import BaseTrainingConfigSchema

VALID_MOTION_TYPES = ("smooth", "abrupt", "respiratory", "cardiac", "rigid", "elastic")


class MotionConfig(BaseModel):
    """Motion meta-training configuration."""

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    max_translation: float = Field(
        default=5.0,
        gt=0.0,
        le=128.0,
        description="Maximum translation (pixels) for random motion corruption (must be > 0)",
    )
    max_rotation: float = Field(
        default=0.05,
        gt=0.0,
        le=3.14159,
        description="Maximum rotation (radians) for random motion corruption (0, pi]",
    )
    motion_type: Literal["smooth", "abrupt", "respiratory", "cardiac", "rigid", "elastic"] = Field(
        default="smooth",
        description=f"Type of motion trajectory. One of: {VALID_MOTION_TYPES}",
    )
    fixed_trajectory_seed: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Single-task control: draw one motion trajectory from this seed and reuse it "
            "for every training batch. None draws a new random trajectory per batch "
            "(meta-training). Validation always draws random motion, so both arms are "
            "scored on the same distribution."
        ),
    )

    @field_validator("motion_type")
    @classmethod
    def _validate_motion_type(cls, v: str) -> str:
        if v not in VALID_MOTION_TYPES:
            raise ValueError(f"motion_type must be one of {VALID_MOTION_TYPES}, got {v!r}")
        return v


class TrainingConfigMotion(BaseTrainingConfigSchema):
    """Configuration for Motion Meta-Learning paradigm."""

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    training_mode: Literal["motion_meta", "motion", "meta_learning"] = Field(default="motion_meta")

    motion: MotionConfig = Field(
        default_factory=MotionConfig,
        description="Motion Meta-Training specific configuration",
    )


__all__ = ["MotionConfig", "TrainingConfigMotion"]
