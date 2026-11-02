r"""Schema for the ``mcgi_encoder`` ``model_kwargs`` (MCGI, bundle M2).

The single owner of the encoder's knobs: ``MCGIEncoder.__init__`` validates what
the arm declares under ``model.model_kwargs`` through this model, and the
``mcgi_invariance_declared`` audit check reads the same model, so the defaults and
the legal values are defined once. MCGI rides the reconstruction strategy; the
invariance is structural (the rank transform in front of the backbone), not
something the training loop enforces.

* ``rank_transform`` -- the per-image midrank :math:`R` in front of the backbone.
  ``false`` is the one-knob non-invariant control: the same backbone on raw
  intensities, so the pair isolates the rank transform.
* ``symmetrize_order_reversal`` -- adjoins :math:`\sigma:\mathbf x\mapsto-\mathbf x`,
  extending invariance from :math:`G_+` to :math:`G_+\rtimes\mathbb Z_2`.
* ``hard_rank_eval`` -- must be ``true``: the exact midrank runs in training and
  inference (#2409 retired the soft rank).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MCGIConfig(BaseModel):
    """Validated ``model_kwargs`` of the monotone-contrast-group-invariant encoder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden_channels: int = Field(default=16, ge=1, description="Backbone hidden width.")
    depth: int = Field(default=3, ge=1, description="Conv layers in the backbone.")
    rank_transform: bool = Field(
        default=True,
        description=(
            "Apply the per-image midrank before the backbone. False is the "
            "non-invariant one-knob control."
        ),
    )
    symmetrize_order_reversal: bool = Field(
        default=True,
        description=(
            "Pool over x -> -x, extending invariance to G_+ x Z_2 (needed when "
            "the cohort mixes order-reversing contrasts such as T1w and T2w)."
        ),
    )
    hard_rank_eval: bool = Field(
        default=True,
        description="Exact midrank in every mode; false is rejected (#2409).",
    )

    @field_validator("hard_rank_eval")
    @classmethod
    def _exact_rank_only(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "mcgi_encoder runs the exact midrank in every mode; hard_rank_eval=false "
                "is not supported (#2409)."
            )
        return value


__all__ = ["MCGIConfig"]
