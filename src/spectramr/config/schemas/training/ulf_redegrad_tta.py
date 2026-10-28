"""Config for UlfReDegradationTTAStrategy (MICCAI MRIxFields2026, B-2.10).

Read by ``UlfReDegradationTTAStrategy`` via ``config.training.ulf_redegrad_tta``.
"""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class UlfReDegradationTTAConfig(StrictSchema):
    """Re-degradation-consistency test-time-adaptation knobs (B-2.10)."""

    adaptation_steps: int = Field(
        default=8,
        ge=0,
        description=(
            "Per-sample test-time gradient steps adapting the MODEL on the "
            "self-supervised re-degradation loss ||A(G(src)) - src||. 0 disables TTA "
            "(plain feed-forward) — the one-knob ablation control."
        ),
    )
    adaptation_lr: float = Field(
        default=1.0e-4,
        gt=0.0,
        description="Inner-loop (test-time) Adam learning rate for the adaptation.",
    )
    consistency_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight of the re-degradation-consistency loss in the inner loop.",
    )
