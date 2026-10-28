"""Conditional alpha-stable denoising training sub-section schema.

Located at: ``training.levy_denoising``. Read by ``LevyDenoisingStrategy``,
which trains ``D(y, x_t, sigma_t) -> x_0`` on ``x_t = x_0 + sigma_t * S_alpha``
and samples at validation with a deterministic DLIM step or a stochastic
reverse SDE with Poisson resetting.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spectramr.config.schemas.strictness import StrictSchema

#: training_mode keys routed to LevyDenoisingStrategy, the only reader of this block.
LEVY_DENOISING_MODES: tuple[str, ...] = ("levy_diffusion", "resetting_diffusion")


def reads_levy_denoising(strategy_class: str | None, training_mode: str | None) -> bool:
    """Whether the declared strategy is the one that reads ``training.levy_denoising``."""
    strategy = strategy_class or ""
    return strategy.endswith("LevyDenoisingStrategy") or (strategy or training_mode or "") in (
        LEVY_DENOISING_MODES
    )


class LevyDenoisingTrainingConfigSchema(StrictSchema):
    """Noise law, noise-level range and validation sampler of the conditional denoiser."""

    alpha: float = Field(
        ...,
        gt=0.0,
        le=2.0,
        description=(
            "Stability index of the symmetric alpha-stable corruption. 2.0 is Gaussian "
            "(variance 2 sigma^2 in this parameterisation) and is the control; < 2 is heavy-tailed."
        ),
    )
    sigma_min: float = Field(
        default=0.01,
        gt=0.0,
        description="Smallest noise scale (training draw and last sampler step).",
    )
    sigma_max: float = Field(
        default=2.0, gt=0.0, description="Largest noise scale (training draw and sampler start)."
    )
    noise_clip: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Clamp on the unit-scale stable draw before it is multiplied by sigma_t; "
            "None leaves the tails untouched."
        ),
    )
    sampler: Literal["dlim", "resetting_sde"] = Field(
        default="dlim",
        description=(
            "Validation sampler: 'dlim' is the deterministic x0-prediction step; "
            "'resetting_sde' is the Euler-Maruyama reverse SDE with Poisson resetting "
            "(Gaussian score, so alpha must be 2.0)."
        ),
    )
    reset_rate: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "resetting_sde only: resets per unit of log-sigma time; a step from sigma_i to "
            "sigma_i+1 resets with probability 1 - (sigma_i+1 / sigma_i)^rate. 0 is the plain SDE."
        ),
    )
    reset_sigma: float | None = Field(
        default=None,
        gt=0.0,
        description="resetting_sde only: noise level a reset re-noises the sample up to.",
    )
    max_resets: int = Field(
        default=0,
        ge=0,
        description="resetting_sde only: cap on resets per sample (bounds the NFE).",
    )
    sampling_steps: int = Field(
        default=10, ge=1, description="Denoiser evaluations per validation sample."
    )

    @staticmethod
    def denoiser_in_channels(cond_channels: int, out_channels: int) -> int:
        """Width of the network input ``[y, x_t, log sigma_t]``."""
        return cond_channels + out_channels + 1

    @model_validator(mode="after")
    def _ordered_noise_range(self) -> LevyDenoisingTrainingConfigSchema:
        if self.sigma_min >= self.sigma_max:
            raise ValueError(
                f"training.levy_denoising: sigma_min ({self.sigma_min}) must be < "
                f"sigma_max ({self.sigma_max})."
            )
        self._resetting_knobs_are_read()
        return self

    def _resetting_knobs_are_read(self) -> None:
        tag = "training.levy_denoising"
        if self.sampler != "resetting_sde":
            if self.reset_rate or self.reset_sigma is not None or self.max_resets:
                raise ValueError(
                    f"{tag}: reset_rate / reset_sigma / max_resets are read only by "
                    f"sampler 'resetting_sde', not {self.sampler!r}."
                )
            return
        if self.alpha != 2.0:
            raise ValueError(
                f"{tag}: sampler 'resetting_sde' uses the Tweedie score, exact only at "
                f"alpha=2.0; got alpha={self.alpha}."
            )
        if self.reset_rate == 0.0:
            if self.reset_sigma is not None or self.max_resets:
                raise ValueError(f"{tag}: reset_sigma / max_resets are unread at reset_rate=0.")
            return
        if self.reset_sigma is None or not self.sigma_min < self.reset_sigma <= self.sigma_max:
            raise ValueError(
                f"{tag}: reset_rate > 0 needs reset_sigma in (sigma_min, sigma_max]; "
                f"got {self.reset_sigma!r}."
            )
        if self.max_resets < 1:
            raise ValueError(f"{tag}: reset_rate > 0 with max_resets=0 never resets.")


__all__ = ["LEVY_DENOISING_MODES", "LevyDenoisingTrainingConfigSchema", "reads_levy_denoising"]
