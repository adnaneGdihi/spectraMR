"""Schema for the VF null-space marker erasure block (``training.null_space_erasure``)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NullSpaceErasureConfig(BaseModel):
    """Knobs of :class:`spectramr.infrastructure.physics.null_space_erasure.NullSpaceMarkerEraser`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False, description="Project the twin marker out of the reconstruction."
    )
    epsilon: float = Field(
        default=1.0e-6, gt=0.0, description="Tikhonov term of the pseudo-inverse."
    )
    apply_at_inference: bool = Field(
        default=True,
        description="Erase in the validation/inference pass. The only mode implemented; "
        "false would erase nowhere, so enabled + false is rejected at build.",
    )
