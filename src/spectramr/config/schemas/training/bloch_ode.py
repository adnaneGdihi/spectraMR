"""Schema for the VF spin-history block (``training.bloch_ode``)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class BlochODEConfig(BaseModel):
    """Knobs of :class:`spectramr.infrastructure.physics.spin_history.SpinHistoryStage`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(default=False, description="Corrupt the twin input with spin history.")
    dt: float = Field(default=10.0, gt=0.0, description="Integration step (ms).")
    slice_thickness: float = Field(
        default=5.0,
        gt=0.0,
        description="Through-plane displacement (voxels) beyond which a voxel resets.",
    )
    n_steps: int = Field(
        default=150, ge=1, description="Integration steps; n_steps * dt must cover 2 TR."
    )
    tr: float = Field(default=500.0, gt=0.0, description="Repetition time (ms).")
    t1_ms: float = Field(
        default=400.0, gt=0.0, description="Parenchyma T1 (ms), uniform over the slice."
    )
    t2_ms: float = Field(
        default=100.0, gt=0.0, description="Parenchyma T2 (ms), uniform over the slice."
    )
    amplitude: float = Field(
        default=8.0, gt=0.0, description="Peak through-plane displacement (voxels)."
    )
    period_ms: float = Field(
        default=1500.0, gt=0.0, description="Period of the through-plane motion (ms)."
    )
