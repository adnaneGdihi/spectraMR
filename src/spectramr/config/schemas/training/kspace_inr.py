"""k-space INR training sub-schema (``training_mode: kspace_inr``).

The strategy fits the SIREN coordinate field to each slice's sampled k-space in
an inner loop, starting from the network's weights, so the outer optimiser learns
the initialisation the per-slice fit starts from (Tancik et al. 2021).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class KSpaceINRTrainingConfigSchema(BaseModel):
    """Inner-loop knobs of the per-slice k-space INR fit."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    inner_steps: int = Field(
        default=3,
        ge=1,
        description=(
            "Gradient steps of the per-slice fit on the sampled k-space points. At least 1: "
            "with none, every slice gets the same coordinate field."
        ),
    )
    inner_lr: float = Field(
        default=1e-3,
        gt=0.0,
        description=(
            "Step size of the per-slice fit (plain gradient descent on the DC term of k-space "
            "scaled to unit RMS on the sampled points)."
        ),
    )
    lambda_dc: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight of the fitted field's sampled-point k-space MSE, relative to the slice's "
            "sampled power, in the outer objective; the image-domain term is the declared "
            "losses.image_losses on RSS(ifft2c(k))."
        ),
    )
    first_order: bool = Field(
        default=True,
        description=(
            "True drops second-order terms of the inner loop from the outer gradient "
            "(first-order MAML); False backpropagates through every inner step."
        ),
    )


__all__ = ["KSpaceINRTrainingConfigSchema"]
