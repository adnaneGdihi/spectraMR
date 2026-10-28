"""TALL-SR training sub-section schema.

Located at: ``training.tall_sr`` in v6.0 YAML configs. Read by
``ReconstructionTrainingStrategy`` when the model is ``tall_sr``.
"""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class TALLSRTrainingConfigSchema(StrictSchema):
    """TALL-SR paradigm-specific training parameters."""

    lambda_locality: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of the differentiable TALL locality regulariser (Spec §9.3) added to "
            "g_total. 0.0 disables it (no compute, no loss key). Requires model_type tall_sr."
        ),
    )


__all__ = ["TALLSRTrainingConfigSchema"]
