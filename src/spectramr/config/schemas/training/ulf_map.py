"""Config for UlfMapStrategy (MICCAI MRIxFields2026, B-2.1 physics-noise MAP).

Read by ``UlfMapStrategy`` via ``config.training.ulf_map``. The ULF→HF restoration
is a MAP / half-quadratic-splitting solve whose data term is weighted by the
physically-calibrated Hoult noise level ``σ ∝ B0^{-7/4}`` (lower field ⇒ noisier
measurement ⇒ weaker data pull ⇒ the learned prior fills more). The denoiser /
ADMM iterations / penalty are inherited from ``config.training.pnp``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spectramr.config.schemas.renames import reject_renamed_keys
from spectramr.config.schemas.strictness import StrictSchema


class UlfMapConfig(StrictSchema):
    """Ultra-low-field physics-noise MAP knobs (B-2.1)."""

    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training.ulf_map"))  # type: ignore[arg-type]
    )

    solve_mode: Literal["map", "prior_only"] = Field(
        default="map",
        description=(
            "'map' runs the HQS data-consistency solve. 'prior_only' is the single-pass "
            "control: the validation output is the denoiser applied once to the input, "
            "with no data term."
        ),
    )
    data_weight_mode: Literal["hoult", "normalised"] = Field(
        default="hoult",
        description=(
            "'hoult': w = 1/sigma^2 (clamped), absolute scale; at 64 mT this is ~6e-4 of "
            "rho and the data term is inert. 'normalised': w = relative_data_weight * rho * "
            "(source_field / calibration_field)^3.5, i.e. the Hoult field law (sigma ~ "
            "B0^-1.75) fixes only the field dependence, and the absolute scale is set "
            "against rho."
        ),
    )
    relative_data_weight: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Normalised mode only: w / rho at the calibration field. 1.0 gives the data "
            "term equal weight to the prior penalty at the measured field."
        ),
    )
    calibration_field: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Normalised mode only: the true measurement field (Tesla) at which "
            "w = relative_data_weight * rho. A mis-set source_field then changes w by the "
            "Hoult field law, with this anchor and the scale held fixed."
        ),
    )

    @model_validator(mode="after")
    def _check_weight_mode(self) -> UlfMapConfig:
        if self.data_weight_mode == "normalised":
            missing = [
                n for n in ("relative_data_weight", "calibration_field") if getattr(self, n) is None
            ]
            if missing:
                raise ValueError(f"data_weight_mode='normalised' requires {missing}.")
        elif self.relative_data_weight is not None or self.calibration_field is not None:
            raise ValueError(
                "relative_data_weight / calibration_field are only read in "
                "data_weight_mode='normalised'; they would be ignored under 'hoult'."
            )
        return self

    source_field: float = Field(
        default=0.1,
        gt=0.0,
        description=(
            "Ultra-low-field measurement field strength (Tesla). Sets the Hoult "
            "noise level via σ = sigma_floor / (source_field/3)^1.75."
        ),
    )
    sigma_floor: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "High-field reference noise std (the 3T baseline) used to derive the "
            "ULF noise level through the Hoult B0^{7/4} scaling."
        ),
    )
    data_weight_max: float = Field(
        default=1.0e3,
        gt=0.0,
        description=(
            "Upper clamp on the data-consistency weight 1/σ² (numerical stability "
            "at high field where σ is tiny)."
        ),
    )
