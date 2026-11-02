"""Post-training model-compression export (``compression:``).

Not to be confused with ``physics.coil_processing.compression``, which reduces the coil axis of
the data. This block compresses the trained *network*: after the run finishes, the weights the
validation grades are pruned and/or quantized, the validation pass is re-run on them, and the
fp32 and compressed numbers are written side by side to ``compression_report.json``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spectramr.config.schemas.enums import CompressionPruning, CompressionQuantization


class CompressionQuantizationConfig(BaseModel):
    """Weight-only symmetric per-tensor quantization of Conv/Linear weights."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: CompressionQuantization = Field(
        default=CompressionQuantization.NONE,
        description="none | int8_weight | int4_weight. Biases and activations stay fp32.",
    )


class CompressionPruningConfig(BaseModel):
    """Post-training weight pruning of Conv/Linear weights."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: CompressionPruning = Field(
        default=CompressionPruning.NONE,
        description=(
            "none | l1_global (global magnitude) | l2_filter (per-layer filter L2) | "
            "random (unstructured, seeded from run.seed)."
        ),
    )
    amount: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Fraction of prunable weights (l1_global, random) or filters (l2_filter) zeroed.",
    )

    @model_validator(mode="after")
    def _amount_matches_method(self) -> CompressionPruningConfig:
        if self.method is CompressionPruning.NONE and self.amount > 0.0:
            raise ValueError(
                f"compression.pruning.amount={self.amount} is unread while method is 'none'; "
                "declare a pruning method or remove the amount."
            )
        if self.method is not CompressionPruning.NONE and self.amount <= 0.0:
            raise ValueError(
                f"compression.pruning.method={self.method.value!r} needs amount in (0, 1)."
            )
        return self


class CompressionExportConfigSchema(BaseModel):
    """Top-level ``compression:`` block: the post-training export the run performs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = Field(default=False, description="Run the post-training compression export.")
    quantization: CompressionQuantizationConfig = Field(
        default_factory=CompressionQuantizationConfig
    )
    pruning: CompressionPruningConfig = Field(default_factory=CompressionPruningConfig)

    @model_validator(mode="after")
    def _enabled_needs_a_method(self) -> CompressionExportConfigSchema:
        no_method = (
            self.quantization.method is CompressionQuantization.NONE
            and self.pruning.method is CompressionPruning.NONE
        )
        if self.enabled and no_method:
            raise ValueError(
                "compression.enabled is true but neither quantization.method nor pruning.method "
                "is set; the export would re-score the unchanged fp32 weights."
            )
        return self


__all__ = [
    "CompressionExportConfigSchema",
    "CompressionPruningConfig",
    "CompressionQuantizationConfig",
]
