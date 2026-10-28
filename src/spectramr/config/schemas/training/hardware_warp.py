"""Hardware-warp calibration schema (``training_mode: hardware_warp_calibration``).

A scanner's gradient-nonlinearity and static-field displacement form a harmonic-gradient warp
that repeat sessions of one head observe through the commutator of the head motion with the
warp (:mod:`spectramr.infrastructure.physics.hardware_warp_commutator`). The calibration is a
frozen-parameter least-squares fit, so this schema carries no optimizer knobs: it names the
harmonic band, the ball the basis is orthonormalized on, the identifiability floor the Tier-1
audit enforces, the pose set the fit consumes and the artifact it writes, and the scanner
identity the artifact is labelled with. A vendor pipeline may already apply a partial
gradient-nonlinearity correction, so the estimate is a residual that is only meaningful per
unit and software version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from .base import BaseTrainingConfigSchema


class TrainingConfigHardwareWarpCalibration(BaseTrainingConfigSchema):
    """Hardware-warp identification from repeat-session commutators."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    max_degree: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Highest solid-harmonic degree of the warp basis.",
    )
    min_degree: int = Field(
        default=2,
        ge=1,
        description=(
            "Lowest solid-harmonic degree of the warp basis. Degree 1 is the constant-field "
            "gauge of the scanner frame and is excluded by default."
        ),
    )
    ball_radius_mm: float = Field(
        default=100.0,
        gt=0,
        description="Radius of the ball the basis is orthonormalized on, in mm.",
    )
    basis_points_per_axis: int = Field(
        default=9,
        ge=5,
        le=41,
        description="Cubic-grid resolution per axis of the ball point cloud.",
    )
    identifiability_floor: float = Field(
        default=1e-4,
        gt=0,
        description=(
            "Smallest per-degree identifiability constant (omega^2 (1 - |P_l(cos theta)|) per "
            "Theorem 2.1) the pose set must reach; the Tier-1 audit fails below it."
        ),
    )
    pose_set_path: Path = Field(
        ...,
        description="JSON pose set: unit, software version, ball radius and the session pairs.",
    )
    artifact_path: Path = Field(
        ...,
        description="Where the calibration writes {coeffs, covariance, degrees, constants}.",
    )
    readout_term: Literal["off"] = Field(
        default="off",
        description=(
            "Readout-direction term handling. Only 'off' exists: the pose-set contract carries "
            "no within-session residuals to estimate the term from, so the strategy raises on "
            "anything else and no other value is advertised until one is implemented."
        ),
    )
    unit_id: str = Field(
        ..., min_length=1, description="Scanner unit the artifact is labelled with."
    )
    software_version: str = Field(
        ..., min_length=1, description="Vendor software version the residual warp belongs to."
    )

    @model_validator(mode="after")
    def _degree_band_is_ordered(self) -> TrainingConfigHardwareWarpCalibration:
        if self.min_degree > self.max_degree:
            raise ValueError(
                f"min_degree ({self.min_degree}) must not exceed max_degree ({self.max_degree})."
            )
        return self

    @classmethod
    def paradigm_field_names(cls) -> frozenset[str]:
        """The knobs this paradigm adds on top of the shared base schema."""
        return frozenset(cls.model_fields) - frozenset(BaseTrainingConfigSchema.model_fields)

    @classmethod
    def from_training_block(cls, training: Any) -> TrainingConfigHardwareWarpCalibration:
        """Validate the paradigm knobs carried as extras on a loaded ``training:`` block.

        ``TrainingStrategyConfigSchema`` is ``extra="allow"``, so a loaded config holds these
        knobs untyped; this is the one place they become typed and bounded. A ``None`` value
        counts as undeclared and takes the schema default. Raises ``ValidationError`` when a
        required knob is missing or a value is out of range.
        """
        declared = {
            name: getattr(training, name)
            for name in cls.paradigm_field_names()
            if getattr(training, name, None) is not None
        }
        return cls.model_validate(declared)


__all__ = ["TrainingConfigHardwareWarpCalibration"]
