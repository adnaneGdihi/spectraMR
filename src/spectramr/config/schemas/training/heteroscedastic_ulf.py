"""Config for HeteroscedasticULFStrategy (MICCAI MRIxFields2026, B-2.9).

Read by ``HeteroscedasticULFStrategy`` via ``config.training.heteroscedastic_ulf``.
The model predicts a per-voxel mean and log-variance; the Gaussian NLL is
regularised by a prior pulling the predicted variance toward the physically
calibrated ULF noise level :math:`\\sigma(B_0)^2` (Hoult), so the uncertainty map
is grounded in the spatially-varying acquisition noise rather than free-floating.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from spectramr.config.schemas.strictness import StrictSchema


class HeteroscedasticULFConfig(StrictSchema):
    """Heteroscedastic ULF restoration knobs (B-2.9)."""

    sigma_floor: float = Field(
        default=0.05,
        gt=0.0,
        description=(
            "High-field (3T) reference noise std; the per-sample ULF target variance "
            "is (sigma_floor / (B0/3)^1.75)^2 via the Hoult scaling."
        ),
    )
    source_field: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "ULF source field (Tesla) for the Hoult anchor when the batch carries no "
            "per-sample field_strength (the paired ULF route). Required in that case: "
            "absent both, the strategy raises rather than anchoring to the 3T floor."
        ),
    )
    lambda_var_prior: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Weight of the variance-prior term pulling the predicted variance toward "
            "the physical Hoult ULF noise level (0 disables it = plain Gaussian NLL)."
        ),
    )
    max_sigma: float = Field(
        default=0.2,
        gt=0.0,
        le=1.0,
        description=(
            "Clamp on the target noise std on the [0,1] data scale. The raw Hoult "
            "amplification is an absolute pre-normalisation factor (~370x at 0.1T) "
            "that is meaningless on renormalised [0,1] images; this keeps the variance "
            "anchor physical (a [0,1] residual variance cannot exceed ~1)."
        ),
    )
    lambda_mean: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of a direct L1 anchor |mean - target| on the mean channel. Guards the "
            "DEGENERATE hedge (#20) where the Gaussian NLL is minimised by inflating the "
            "variance (shrinking exp(-logvar)*(y-mean)^2) and letting the mean go lazy; the "
            "anchor keeps the mean grounded regardless of the predicted variance. 0 = bare "
            "NLL (pre-guard behaviour)."
        ),
    )
    logvar_min: float = Field(
        default=-10.0,
        description=(
            "Lower clamp on the predicted log-variance before the NLL. Prevents a spuriously "
            "over-confident (near-zero-variance) collapse. For [0,1] data logvar=-10 => "
            "sigma ~ 0.007."
        ),
    )
    logvar_max: float = Field(
        default=2.0,
        description=(
            "Upper clamp on the predicted log-variance before the NLL. Prevents a "
            "measurement-independent variance blow-up (the 'I don't know' hedge). For [0,1] "
            "data logvar=2 => sigma ~ 2.7. Must exceed logvar_min (validated at load)."
        ),
    )

    @model_validator(mode="after")
    def _check_logvar_band(self) -> HeteroscedasticULFConfig:
        if self.logvar_max <= self.logvar_min:
            raise ValueError(
                f"logvar_max ({self.logvar_max}) must exceed logvar_min ({self.logvar_min})."
            )
        return self
