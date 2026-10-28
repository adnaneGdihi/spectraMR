"""Config for UlfDpsStrategy (MICCAI MRIxFields2026, B-2.2).

Read by ``UlfDpsStrategy`` via ``config.training.ulf_dps``; absence falls back to
defaults.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class UlfDpsConfig(StrictSchema):
    """ULF-consistent diffusion-posterior-sampling knobs (B-2.2)."""

    timesteps: int = Field(
        default=1000,
        ge=1,
        description="Number of DDPM diffusion steps for the learned high-field prior.",
    )
    beta_schedule: Literal["cosine", "linear"] = Field(
        default="cosine",
        description="Noise schedule for alphas_cumprod ('cosine' or 'linear').",
    )
    sampling_steps: int = Field(
        default=50,
        ge=1,
        description="DPS reverse-sampling steps at validation (<= timesteps).",
    )
    likelihood_weight: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "DPS measurement-likelihood guidance weight (the data-consistency step "
            "size against the ULF observation). 0.0 = unconstrained prior sampling "
            "(no ULF consistency) — the one-knob ablation control."
        ),
    )
