"""Schema for the VF phase-steganography block (``training.phase_steganography``)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PhaseSteganographyConfig(BaseModel):
    """Knobs of :class:`spectramr.infrastructure.physics.phase_steganography.PhaseSteganographicMarker`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False, description="Inject the phase grid before the twin corrupts."
    )
    spatial_freq_x: float = Field(default=8.0, gt=0.0, description="Grid cycles per FOV along x.")
    spatial_freq_y: float = Field(default=8.0, gt=0.0, description="Grid cycles per FOV along y.")
    amplitude: float = Field(default=0.5, gt=0.0, description="Peak phase modulation (rad).")
    learnable: bool = Field(
        default=False,
        description="Trainable grid parameters. Not wired to an optimizer, so true is rejected.",
    )
