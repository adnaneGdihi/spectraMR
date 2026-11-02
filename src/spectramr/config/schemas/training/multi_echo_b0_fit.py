"""Config schema for the multi-echo B0 field-fit strategy (audit 2026-07 I1).

Unlike ``B0MappingStrategy`` (deformable registration, NOT B0 estimation), the
``multi_echo_b0_fit`` strategy runs the differentiable analytic B0 fit
(``infrastructure.physics.b0_mapping.map_b0``: conjugate-product phase
difference + Tikhonov CG) on the generator's refined complex echoes and grades
the recovered field in Hz against a reference ``b0_map`` with ``B0FieldRMSE``.
"""

from __future__ import annotations

from pydantic import Field

from spectramr.config.schemas.strictness import StrictSchema


class MultiEchoB0FitConfig(StrictSchema):
    """Knobs for the physics-in-the-loop multi-echo B0 fit (I1).

    ``t_shift`` is the readout time-shift between the two echoes used for the
    phase-difference fit (seconds, i.e. ΔTE). It is load-bearing: the recovered
    field scales as ``ΔB0 = +angle(E2·conj(E1)) / (2π · t_shift)``, so it MUST match the
    acquisition. ``gamma`` follows Koolstra et al. (2021): 2e-10 simulation,
    2e-8 phantom, 4e-8 in vivo.
    """

    t_shift: float = Field(
        gt=0.0,
        description="Readout time-shift ΔTE between the two echoes (seconds). "
        "Load-bearing: ΔB0 = Δφ/(2π·t_shift).",
    )
    gamma: float = Field(
        default=2e-10,
        gt=0.0,
        description="Tikhonov smoothness weight for the field solve "
        "(2e-10 sim / 2e-8 phantom / 4e-8 in vivo).",
    )
    max_cg_iterations: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Max CG iterations for the Tikhonov field solve.",
    )
    lambda_field: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight on the Hz field-map MSE vs the reference b0_map.",
    )
    lambda_echo: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight on an optional echo-consistency L1 (refined echoes "
        "vs batch['target_echoes'], when provided). 0 disables it.",
    )
