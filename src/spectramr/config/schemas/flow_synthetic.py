"""``data.flow_synthetic``: the analytic 4D-flow phantom behind ``dataset_type: flow_synthetic``.

The phantom has exact velocity ground truth, which no open 4D-flow acquisition
provides, so it carries the supervised training and every ground-truth metric of
the ``mri_flow`` regime. Velocity encoding (``venc``, scheme) is read from
``data.phase_contrast``, the one owner of the acquisition's encoding; this block
holds only what a real acquisition would not need.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FlowSyntheticConfigSchema(BaseModel):
    """Grid, physics and sampling of the synthetic phase-contrast cine."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(default=False, description="Master toggle for the synthetic phantom.")
    train_subjects: int = Field(
        default=8, gt=0, description="Distinct phantoms in the training split."
    )
    val_subjects: int = Field(
        default=2, gt=0, description="Distinct phantoms in the validation split."
    )
    grid: tuple[int, int, int] = Field(
        default=(32, 48, 40),
        description="(readout X, phase-encode Y, phase-encode Z) matrix; one item per X.",
    )
    voxel_size_mm: tuple[float, float, float] = Field(
        default=(2.0, 2.0, 2.0), description="Voxel size in (X, Y, Z) order, mm."
    )
    frames: int = Field(default=12, ge=2, description="Cardiac frames per cycle.")
    coils: int = Field(default=4, ge=1, description="Receive coils.")
    vessels: int = Field(default=2, ge=1, le=3, description="Tubes per phantom.")
    peak_velocity: float = Field(
        default=1.0,
        gt=0,
        description="Peak speed (m/s) of the fastest tube; above venc it aliases.",
    )
    noise_std: float = Field(
        default=0.02,
        ge=0,
        description="Complex k-space noise std, relative to tissue magnitude 0.5.",
    )
    acceleration: float = Field(
        default=6.0, ge=1, description="k-t undersampling factor per frame."
    )
    center_fraction: float = Field(
        default=0.08,
        ge=0,
        lt=1,
        description="Fully sampled centre box, fraction per phase-encode axis.",
    )
    sampling_pattern: Literal["kt_vd_gaussian"] = Field(
        default="kt_vd_gaussian",
        description="k-t pattern, resolved by kt_sampling.kt_sampling_pattern.",
    )
    sampling_sigma: float = Field(
        default=0.3, gt=0, description="Gaussian density width in normalised k-space coordinates."
    )
    background_order: int = Field(
        default=1, ge=0, le=3, description="Polynomial degree of the per-encoding background phase."
    )
    background_amplitude: float = Field(
        default=0.3,
        ge=0,
        description="Std (rad) of the background-phase coefficients; 0 disables it.",
    )
    seed: int = Field(default=0, description="Phantoms, masks and noise are deterministic in it.")

    @field_validator("grid")
    @classmethod
    def _grid_is_resolvable(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if min(value) < 8:
            raise ValueError(f"every grid axis needs >= 8 voxels; got {value}")
        return value

    @field_validator("voxel_size_mm")
    @classmethod
    def _voxels_are_positive(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if min(value) <= 0:
            raise ValueError(f"voxel sizes must be positive; got {value}")
        return value


__all__ = ["FlowSyntheticConfigSchema"]
