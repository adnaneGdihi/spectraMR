"""Config schema for the field-dependent spectral Wiener strategy (B-2.7)."""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class FieldWienerConfig(StrictSchema):
    """Knobs for the field-dependent spectral Wiener restorer (B-2.7).

    The field-dependent noise PSD is structural on the model (``model_kwargs.use_field_noise`` —
    the one-knob ablation); this block carries the loss weight only.
    """

    source_field: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Source field (Tesla) used when the batch carries no per-sample field_strength "
            "(the paired ULF route). Absent both, the strategy raises."
        ),
    )
