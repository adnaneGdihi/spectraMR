"""Config for FieldColdDiffusionStrategy (MICCAI MRIxFields2026, B-2.4).

Cold diffusion (Bansal et al. 2022) with a *deterministic field down-conversion*
degradation instead of Gaussian noising: the degradation path runs from the
high-field image (t=0, near-identity) to a ULF-like low-resolution image (t=T, a
strong low-pass). A restorer is trained to invert the path and the image is
recovered by the improved cold-diffusion sampling rule. Read via
``config.training.field_cold_diffusion``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spectramr.config.schemas.strictness import StrictSchema

_LOWPASS_KNOBS = ("max_cutoff", "min_cutoff")
_GAUSSIAN_KNOBS = ("noise_sigma_min", "noise_sigma_max")


class FieldColdDiffusionConfig(StrictSchema):
    """Field-degradation cold-diffusion knobs (B-2.4)."""

    degradation: Literal["field_lowpass", "gaussian"] = Field(
        default="field_lowpass",
        description=(
            "Degradation D_t of the forward path. 'field_lowpass' is the field "
            "down-conversion (max_cutoff -> min_cutoff). 'gaussian' is isotropic "
            "Gaussian noising x_t = x_0 + sigma_t * eps (noise_sigma_min -> "
            "noise_sigma_max); its sampler starts from the ULF image plus "
            "noise_sigma_max noise. Unknown values are rejected at load."
        ),
    )

    n_steps: int = Field(
        default=10,
        ge=1,
        description="Sampler step count (reverse cold-diffusion trajectory length).",
    )
    train_steps: int = Field(
        default=10,
        ge=1,
        description=(
            "Training curriculum resolution (t ~ U{1..train_steps}); INDEPENDENT of "
            "the sampler n_steps so the single-step ablation collapses only the "
            "sampler, not the training degradation distribution (pitfall #17)."
        ),
    )
    max_cutoff: float = Field(
        default=5.0,
        gt=0.0,
        description="Low-pass cutoff at t=0 (near-identity high-field end; large=sharp).",
    )
    min_cutoff: float = Field(
        default=0.25,
        gt=0.0,
        description="Low-pass cutoff at t=T (ULF end; small=heavy resolution loss).",
    )

    noise_sigma_min: float | None = Field(
        default=None,
        gt=0.0,
        description="Gaussian noise std at t=1 (degradation='gaussian' only; required there).",
    )
    noise_sigma_max: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Gaussian noise std at t=T, also the noise added to the ULF start of the "
            "sampler (degradation='gaussian' only; required there)."
        ),
    )
    sampler_seed: int = Field(
        default=0,
        description=(
            "Seed of the start-noise draw, re-seeded on every validation call so "
            "passes stay comparable (degradation='gaussian' only)."
        ),
    )

    @model_validator(mode="after")
    def _knobs_match_degradation(self) -> FieldColdDiffusionConfig:
        """Each degradation reads its own schedule; a knob of the other one is unread."""
        if self.degradation == "gaussian":
            missing = [k for k in _GAUSSIAN_KNOBS if getattr(self, k) is None]
            if missing:
                raise ValueError(f"degradation='gaussian' requires {missing}.")
            if self.noise_sigma_min > self.noise_sigma_max:  # type: ignore[operator]
                raise ValueError("noise_sigma_min must not exceed noise_sigma_max.")
            unread = [k for k in _LOWPASS_KNOBS if k in self.model_fields_set]
        else:
            unread = [k for k in _GAUSSIAN_KNOBS if getattr(self, k) is not None]
            if "sampler_seed" in self.model_fields_set:
                unread.append("sampler_seed")
        if unread:
            raise ValueError(
                f"{unread} are not read under degradation={self.degradation!r}; remove them."
            )
        return self
