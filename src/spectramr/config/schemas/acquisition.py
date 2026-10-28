"""Acquisition / sampling design configuration.

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-11 (M3),
      §PR-25 (Part-I G).

Two top-level sub-blocks:

- ``codesign``  — continuous-trajectory PILOT-style acquisition with
                  hardware-feasibility constraints (slew rate, max
                  gradient).
- ``active``    — closed-loop BALD-style active acquisition, where the
                  network selects the next k-space line based on its
                  current posterior.

Each is opt-in (``enabled=False`` by default), so adding this schema
is non-breaking for existing configs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class HardwareConstraints(BaseModel):
    """Slew-rate / max-gradient feasibility envelope for PILOT."""

    model_config = {"extra": "ignore", "frozen": True}

    slew_rate_max_T_per_m_per_s: float = Field(
        default=200.0,
        gt=0.0,
        description="Hardware slew-rate ceiling in T/m/s.",
    )
    gradient_max_mT_per_m: float = Field(
        default=40.0,
        gt=0.0,
        description="Peak gradient amplitude in mT/m.",
    )
    gradient_dwell_us: float = Field(
        default=4.0,
        gt=0.0,
        description="Gradient sampling dwell time in microseconds.",
    )


class CodesignConfig(BaseModel):
    """Continuous-trajectory PILOT-style co-design (image + sampling)."""

    model_config = {"extra": "ignore", "frozen": True}

    enabled: bool = Field(default=False)
    constraints: HardwareConstraints = Field(default_factory=HardwareConstraints)
    feasibility_lambda: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight on the hardware-feasibility penalty term.",
    )
    field_of_view_m: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Field of view in metres along the phase-encode/readout plane. Required when "
            "codesign is enabled: it converts the trajectory to cycles per metre for the "
            "T/m and T/m/s limits."
        ),
    )


class ActiveAcquisitionConfig(BaseModel):
    """Closed-loop active acquisition (BALD / max-variance / random)."""

    model_config = {"extra": "ignore", "frozen": True}

    enabled: bool = Field(default=False)
    target_rate: float = Field(
        default=0.125,
        gt=0.0,
        le=1.0,
        description="Total sampling-rate budget (1/R).",
    )
    seed_mask_lf_fraction: float = Field(
        default=0.04,
        ge=0.0,
        le=1.0,
        description="Initial low-frequency seed fraction before any active queries.",
    )
    query_strategy: str = Field(
        default="bald",
        description="Acquisition function: bald | max_variance | max_entropy | random.",
    )
    query_batch: int = Field(
        default=4,
        ge=1,
        description="Number of lines acquired per active-acquisition step.",
    )
    retrain_every: int = Field(
        default=100,
        ge=1,
        description="Gradient steps between successive query rounds.",
    )


class AcquisitionConfigSchema(BaseModel):
    """Top-level acquisition design block (top-level under TrainingSettings)."""

    model_config = {"extra": "ignore", "frozen": True}

    codesign: CodesignConfig = Field(default_factory=CodesignConfig)
    active: ActiveAcquisitionConfig = Field(default_factory=ActiveAcquisitionConfig)


__all__ = [
    "AcquisitionConfigSchema",
    "ActiveAcquisitionConfig",
    "CodesignConfig",
    "HardwareConstraints",
]
