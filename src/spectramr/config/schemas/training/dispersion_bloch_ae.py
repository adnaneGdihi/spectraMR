r"""Schema for DL-BAE: Dispersion-Latent Bloch Autoencoder (M4).

Defines the ``training.dispersion_bloch_ae`` sub-block and the top-level
``TrainingConfigDispersionBlochAE`` that mounts it. The strategy
(:class:`~spectramr.infrastructure.training.strategies.dispersion_bloch_ae_strategy.DispersionBlochAEStrategy`)
trains :class:`~spectramr.models.physics_ae.disp_bloch_ae.DispersionBlochAutoencoder`:
an encoder maps a multi-field image stack to a *field-invariant* tissue latent
:math:`\xi=(\rho,\{b_k,\tau_{c,k}\},a_0,c_0)`, and the decoder is the
Bloembergen-Purcell-Pound dispersion law
(:mod:`spectramr.infrastructure.physics.dispersion`) followed by the existing Bloch
render.

Identifiability is a *hard* constraint, not a preference: a :math:`P`-pool BPP
model has :math:`2P+1` free constants per rate, so recovering them needs
measurements at :math:`M \ge 2P+1` distinct fields. The Tier-1
``dispersion_identifiability`` check enforces
``2*num_pools + 1 <= len(fields_present)`` as an ERROR -- an under-determined arm
would otherwise train happily to a meaningless latent (pitfall #9: no silent
degradation).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..renames import reject_renamed_keys
from .base import BaseTrainingConfigSchema


class DispersionBlochAEConfig(BaseModel):
    """Hyperparameters for the dispersion-latent Bloch autoencoder (DL-BAE)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training.dispersion_bloch_ae"))  # type: ignore[arg-type]
    )

    num_pools: int = Field(
        default=1,
        ge=1,
        le=4,
        description=(
            "Number of BPP relaxation pools P. Needs M >= 2P+1 distinct fields "
            "to be identifiable (Tier-1 dispersion_identifiability)."
        ),
    )
    fields_present: tuple[float, ...] = Field(
        ...,
        description=(
            "Distinct B0 field strengths (Tesla) present in the training data, "
            "in the channel order the encoder receives them."
        ),
    )
    init_tau_c: float = Field(
        default=1e-8,
        gt=0.0,
        description="Initial rotational correlation time tau_c (s); ~1e-8 for brain tissue.",
    )
    monotonicity_weight: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight on the dT1/dB0 >= 0 monotonicity penalty. Zero disables the "
            "projection (Tier-1 dispersion_monotonicity_weight_positive warns)."
        ),
    )
    data_consistency_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight on the multi-field data-consistency reconstruction term.",
    )
    tau_c_bounds: tuple[float, float] = Field(
        default=(1e-11, 1e-6),
        description="Physiological (min, max) bounds for tau_c (s); the encoder is clamped here.",
    )

    @field_validator("fields_present")
    @classmethod
    def _fields_distinct_and_positive(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Reject non-positive or duplicated fields -- both break identifiability."""
        if any(b <= 0.0 for b in value):
            raise ValueError(
                f"fields_present must be positive field strengths in Tesla; got {value!r}."
            )
        if len(set(value)) != len(value):
            raise ValueError(
                "fields_present must be DISTINCT: repeated fields add no rank to the "
                f"dispersion fit and inflate the apparent M. Got {value!r}."
            )
        return value

    @field_validator("tau_c_bounds")
    @classmethod
    def _bounds_ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Reject an inverted or non-positive tau_c interval."""
        low, high = value
        if not (0.0 < low < high):
            raise ValueError(f"tau_c_bounds must satisfy 0 < min < max; got {value!r}.")
        return value


class TrainingConfigDispersionBlochAE(BaseTrainingConfigSchema):
    """Top-level training config wrapper that mounts the DL-BAE sub-block."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    dispersion_bloch_ae: DispersionBlochAEConfig


__all__ = [
    "DispersionBlochAEConfig",
    "TrainingConfigDispersionBlochAE",
]
