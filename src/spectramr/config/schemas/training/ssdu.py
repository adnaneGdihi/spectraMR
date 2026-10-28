"""SSDU (self-supervised reconstruction) training sub-schema.

Declarative config for ``training_mode: self_supervised_reconstruction`` and the
``robust_ssdu`` (Noisier2Noise) variant. Frozen Pydantic v2 model,
``extra="forbid"``, mounted on ``TrainingStrategyConfigSchema`` as an optional
sub-block (mirrors the v6.1 pattern used by GeoMambaULFTrainingConfigSchema).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SSDUTrainingConfigSchema(BaseModel):
    """Parameters for the SSDU Lambda/Theta k-space split (+ Robust SSDU)."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    theta_fraction: float = Field(
        default=0.4,
        gt=0.0,
        lt=1.0,
        description="Fraction of acquired k-space points held out into Theta (loss set).",
    )
    num_masks: int = Field(
        default=1,
        ge=1,
        description="Number of independent (Lambda,Theta) splits per example (multi-mask SSDU).",
    )
    split_seed: int = Field(
        default=0,
        ge=0,
        description="RNG seed for reproducible Lambda/Theta splits.",
    )

    # --- Robust SSDU (Noisier2Noise, Millard & Chiew 2024) -----------------
    noisier2noise_correction: bool = Field(
        default=False,
        description=(
            "Enable Robust SSDU (Millard & Chiew): corrupt the network input with "
            "noise of std alpha * sigma_n before the Lambda split, score the "
            "prediction on ALL acquired lines with the weighted loss, and apply the "
            "Noisier2Noise correction at validation. Selected by the 'robust_ssdu' "
            "strategy key."
        ),
    )
    noise_std_estimate: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "k-space noise σ for the Noisier2Noise correction. Required when "
            "noisier2noise_correction is true (pitfall #15). At σ→0 the correction "
            "reduces to vanilla SSDU."
        ),
    )
    noise_std_source: Literal["declared", "self_calibrated"] = Field(
        default="declared",
        description=(
            "Where the Noisier2Noise sigma comes from. 'declared' (default) uses "
            "noise_std_estimate verbatim. 'self_calibrated' measures it per slice "
            "from the coil null space, where a physical image is rank one and the "
            "residual is pure noise, and ignores noise_std_estimate. "
            "A constant cannot be right for every scan once the pipeline "
            "normalizes per subject: normalization removes SCALE but turns "
            "receiver-gain and SNR differences into sigma differences. Measured "
            "on synthetic 4-coil data through this arm's own image-domain "
            "quantile normalization, the correct sigma runs 0.018 at high SNR to "
            "0.268 at low -- a 15x span, around a declared 0.02. M4Raw is 0.3T, "
            "where the low end is the design point, so a high-SNR constant leaves "
            "the correction under-powered by 3-13x and by a different factor per "
            "subject. Self-calibration costs 2-4 ms per batch and reuses the coil "
            "maps already in it; it reads 1.5-7% HIGH with ESPIRiT maps."
        ),
    )
    alpha: float = Field(
        default=0.75,
        gt=0.0,
        description=(
            "Robust SSDU's noise ratio: the network input is corrupted with "
            "complex-Gaussian noise of std alpha * sigma_n on the acquired support "
            "(arXiv:2210.01696 Eq. 25), where sigma_n is noise_std_estimate or the "
            "per-slice null-space fit. It also sets the weight (1+alpha^2)/alpha^2 "
            "on the input lines of the loss (Eq. 35) and the Noisier2Noise "
            "correction applied at validation (Eq. 36). 0.75 is the paper's value "
            "for the weighted loss. Read only under noisier2noise_correction."
        ),
    )
    noise_model: Literal["gaussian_kspace", "ncchi_magnitude"] = Field(
        default="gaussian_kspace",
        description=(
            "Dataset-driven noise model shared with the T1 keystone "
            "(complex-Gaussian per-coil k-space vs non-central-χ RSS magnitude). "
            "Never hardcoded to 0.3T."
        ),
    )
    noise_covariance: Literal["white", "committed_m4raw"] = Field(
        default="white",
        description=(
            "Coil covariance of the injected noise. The Noisier2Noise correction "
            "needs the injected noise distributed as alpha times the measurement "
            "noise (arXiv:2210.01696 Eq. 29); M4Raw's coil noise is correlated "
            "(normalised eigenvalues 0.62-1.39), so 'white' leaves the correction "
            "approximate. 'committed_m4raw' draws with the shape of the committed "
            "M4Raw covariance (infrastructure/physics/m4raw_noise.py), scaled so "
            "the mean per-coil power is (alpha*sigma)^2. Read only under "
            "noisier2noise_correction; needs n_coils equal to that matrix's size."
        ),
    )
    n_coils: int = Field(
        default=1,
        ge=1,
        description=(
            "Coils the measured k-space carries. The strategy refuses a batch with "
            "a different count, because the loss, the noise injection and the "
            "self-calibrated sigma are all per-coil."
        ),
    )

    score_structural_zeros: bool = Field(
        default=False,
        description=(
            "Add the columns outside the acquired phase-encode support to every "
            "scored Theta. y is zero there by construction, so the reconstruction is "
            "pulled to the acquisition's band without any reference. Needs "
            "`support_mask` in the batch (`complementary_rep_masking` emits it). "
            "Vanilla SSDU only: Robust SSDU scores all of Omega with its own weights."
        ),
    )

    @model_validator(mode="after")
    def _validate_robust_requires_noise(self) -> SSDUTrainingConfigSchema:
        if (
            self.noisier2noise_correction
            and self.noise_std_estimate is None
            and self.noise_std_source == "declared"
        ):
            raise ValueError(
                "ssdu.noisier2noise_correction=true requires noise_std_estimate "
                "(the k-space noise sigma); none was set (pitfall #15: the knob "
                "must be wired). Set ssdu.noise_std_source: self_calibrated to "
                "measure it per slice from the coil null space instead."
            )
        if "alpha" in self.model_fields_set and not self.noisier2noise_correction:
            raise ValueError(
                "ssdu.alpha is read only by the Robust SSDU correction, which is off "
                "(noisier2noise_correction: false). Remove it, or turn the correction on."
            )
        if self.noisier2noise_correction and self.noise_model != "gaussian_kspace":
            raise ValueError(
                "the Noisier2Noise correction is derived for complex-Gaussian k-space "
                f"noise; ssdu.noise_model={self.noise_model!r} has no such correction."
            )
        if "noise_covariance" in self.model_fields_set and not self.noisier2noise_correction:
            raise ValueError(
                "ssdu.noise_covariance shapes the Robust SSDU injection, which is off "
                "(noisier2noise_correction: false). Remove it, or turn the correction on."
            )
        if self.noise_covariance == "committed_m4raw" and self.n_coils != 4:
            raise ValueError(
                "ssdu.noise_covariance: committed_m4raw is the 4x4 M4Raw matrix; "
                f"n_coils={self.n_coils} cannot use it."
            )
        if self.noisier2noise_correction and self.num_masks != 1:
            raise ValueError(
                "Robust SSDU draws one (Lambda, noise) corruption per step and scores "
                f"all of Omega; num_masks={self.num_masks} has no Robust SSDU form here."
            )
        if self.score_structural_zeros and self.noisier2noise_correction:
            raise ValueError(
                "ssdu.score_structural_zeros extends the vanilla Theta; Robust SSDU "
                "scores all of Omega with Eq. 35 weights and never reads it."
            )
        if self.noise_std_source == "self_calibrated" and self.noise_std_estimate is not None:
            raise ValueError(
                "ssdu.noise_std_source: self_calibrated IGNORES noise_std_estimate, "
                f"but it is declared as {self.noise_std_estimate}. Leaving it would "
                "put a number in the config that nothing reads and that a reader "
                "would take for the sigma actually used (pitfall #15). Remove it, "
                "or set noise_std_source: declared."
            )
        return self
