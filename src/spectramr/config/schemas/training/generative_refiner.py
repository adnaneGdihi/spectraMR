"""Config for GenerativeRefinerStrategy (MRIxFields2026 saturation work, Track A).

Read by ``GenerativeRefinerStrategy`` via ``config.training.generative_refiner``.

The strategy is the shared *generative-iterative refiner* chassis, generalized from
the b22 ``ulf_dps`` recipe (the only cohort arm that broke the saturation ceiling):
train a field-conditioned diffusion **prior** with a non-saturating eps-matching
objective, and at inference run a DDIM/DPS reverse loop whose per-step data-consistency
guidance is driven by a **pluggable forward operator** ``A`` (the arm's distinctive
mechanism). ``guidance_weight = 0`` recovers the unconditional prior and is the one-knob
ablation, exactly as ``likelihood_weight = 0`` is for b22.

``forward_operator`` is a closed ``Literal`` so an unknown/unimplemented mechanism is
rejected at config-load time (pitfall #9, no silent fallback) rather than degrading to a
default at runtime.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spectramr.config.schemas.strictness import StrictSchema


class GenerativeRefinerConfig(StrictSchema):
    """Generative-iterative refiner knobs (Track A shared chassis)."""

    timesteps: int = Field(
        default=1000, gt=0, description="DDPM diffusion timesteps for the prior schedule."
    )
    beta_schedule: Literal["cosine", "linear"] = Field(
        default="cosine", description="Noise schedule for the eps-matching prior."
    )
    sampling_steps: int = Field(
        default=50, gt=0, description="Reverse (DDIM/DPS) sampling steps at inference."
    )
    guidance_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="DPS data-consistency step size (zeta). 0.0 disables the guidance "
        "(unconditional prior) and IS the one-knob ablation against the mechanism.",
    )
    forward_operator: Literal["field_lowpass", "wiener_band"] = Field(
        default="field_lowpass",
        description="The arm's distinctive mechanism, expressed as the DPS forward "
        "operator A(x). 'field_lowpass' is the field-dependent low-pass degradation "
        "(the b22 recipe); 'wiener_band' projects onto the b27 Wiener recoverable band "
        "(the field-dependent SNR crossover). Unknown values raise at load time.",
    )
    source_field: float = Field(
        default=0.1,
        gt=0.0,
        le=7.0,
        description="Source (measurement) field strength in Tesla for the 'wiener_band' "
        "operator; sets the Wiener noise ratio N(b) and hence the recoverable-band radius "
        "(lower field => larger N => narrower band). Ignored by 'field_lowpass'.",
    )

    @model_validator(mode="after")
    def _check(self) -> GenerativeRefinerConfig:
        if self.sampling_steps > self.timesteps:
            raise ValueError(
                f"sampling_steps ({self.sampling_steps}) must be <= timesteps "
                f"({self.timesteps}); the reverse loop cannot take more steps than the "
                "forward schedule has."
            )
        return self
