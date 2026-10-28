r"""Schema for Bloch-Manifold-Constrained Diffusion Posterior Sampling (B-2).

Idea 2 of ``TODO/VF_CAMPAIGN_IMPLEMENTATION_PLAN.md``. Extends the
diffusion training schema with a ``training.bloch_manifold_dps`` block that
parameterises the manifold-projection DPS sampler. Training itself is
standard DDPM (ε-prediction); the block only configures the *inference*
novelty — the per-voxel Bloch projection and the DPS likelihood guidance.

Mounted on the ``training_mode`` discriminated union as
``"bloch_manifold_dps"`` (see the wiring note in the module docstring of
``spectramr.config.schemas.training``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..renames import reject_renamed_keys
from .diffusion import TrainingConfigDiffusion

# Pulse sequences the projector / Bloch forward map understands. Kept in
# sync with BlochManifoldProjector (only steady-state SPGR/GRE is wired to
# the differentiable Bloch layer today). Listing the broader MRF family
# here lets the schema validate intent while the model raises loudly if an
# unwired sequence is actually instantiated (no silent fallback).
_SUPPORTED_SEQUENCES = (
    "gre",
    "spgr",
    "mprage",
    "bssfp",
    "mrf_fisp",
    "mrf_truefisp",
)


class BlochManifoldDPSConfig(BaseModel):
    """Hyperparameters for the ``training.bloch_manifold_dps`` block."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pulse_sequence: Literal["gre", "spgr", "mprage", "bssfp", "mrf_fisp", "mrf_truefisp"] = Field(
        default="spgr",
        description="Pulse sequence whose Bloch manifold constrains sampling.",
    )
    # mixedCase field names mirror the physics symbols (T1/T2/M0/B1) and the
    # YAML keys verbatim; N815 is intentional here.
    param_bounds_T1_ms: tuple[float, float] = Field(  # noqa: N815
        default=(100.0, 5000.0),
        description="(lo, hi) longitudinal relaxation bound, ms.",
    )
    param_bounds_T2_ms: tuple[float, float] = Field(  # noqa: N815
        default=(10.0, 2000.0),
        description="(lo, hi) transverse relaxation bound, ms.",
    )
    param_bounds_M0: tuple[float, float] = Field(  # noqa: N815
        default=(0.0, 2.0),
        description="(lo, hi) proton-density / M0 bound (arbitrary units).",
    )
    param_bounds_B1: tuple[float, float] = Field(  # noqa: N815
        default=(0.7, 1.3),
        description="(lo, hi) B1 transmit scaling bound (reserved for B1 fits).",
    )
    projection_inner_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Gauss-Newton inner iterations per manifold projection.",
    )
    projection_gn_damping: float = Field(
        default=1e-4,
        gt=0.0,
        description="Levenberg-Marquardt damping for the per-voxel solve.",
    )
    projection_every: int = Field(
        default=1,
        ge=1,
        description="Apply the manifold projection every k reverse steps.",
    )
    likelihood_weight_zeta: float = Field(
        default=1.0,
        ge=0.0,
        description="DPS measurement-likelihood guidance scale (0 = prior-only).",
    )
    sampling_steps: int = Field(
        default=1000,
        ge=1,
        description="Number of reverse diffusion steps at inference.",
    )
    sampler_type: Literal["ddim", "ddpm"] = Field(
        default="ddim",
        description="Reverse sampler. ddim is deterministic, ddpm stochastic.",
    )
    prediction_type: Literal["epsilon", "x0"] = Field(
        default="epsilon",
        description="Whether the score net predicts noise or the clean image.",
    )

    # Retired prior-checkpoint field (#647); the refusal names the removal record.
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training.bloch_manifold_dps"))  # type: ignore[arg-type]
    )

    @model_validator(mode="after")
    def _bounds_physical(self) -> BlochManifoldDPSConfig:
        """Reject unphysical relaxation/M0/B1 bounds (Tier-1 intent)."""
        for name, (lo, hi) in (
            ("param_bounds_T1_ms", self.param_bounds_T1_ms),
            ("param_bounds_T2_ms", self.param_bounds_T2_ms),
            ("param_bounds_M0", self.param_bounds_M0),
            ("param_bounds_B1", self.param_bounds_B1),
        ):
            if lo >= hi:
                raise ValueError(f"bloch_manifold_dps.{name}: lo ({lo}) must be < hi ({hi}).")
            if lo < 0.0:
                raise ValueError(f"bloch_manifold_dps.{name}: lower bound must be >= 0.")
        # T2 <= T1 must be feasible somewhere in the boxes (the relaxation
        # ordering constraint); otherwise the manifold is empty.
        if self.param_bounds_T2_ms[0] > self.param_bounds_T1_ms[1]:
            raise ValueError(
                "bloch_manifold_dps: T2 lower bound exceeds T1 upper bound — "
                "the Bloch manifold {T2 <= T1} would be empty."
            )
        if self.param_bounds_M0[0] >= self.param_bounds_M0[1] or self.param_bounds_M0[1] <= 0.0:
            raise ValueError("bloch_manifold_dps.param_bounds_M0 must be a positive (lo<hi) range.")
        return self


class TrainingConfigBlochManifoldDPS(TrainingConfigDiffusion):
    """Top-level training config when ``training_mode == 'bloch_manifold_dps'``.

    Inherits the full diffusion schema (timesteps, noise_schedule,
    prediction_type, sampler, …) and adds the Bloch-manifold projection
    block. ``extra='allow'`` mirrors the sibling twin-DPS schema so VF
    arms can carry their extended physics blocks.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    bloch_manifold_dps: BlochManifoldDPSConfig


__all__ = [
    "_SUPPORTED_SEQUENCES",
    "BlochManifoldDPSConfig",
    "TrainingConfigBlochManifoldDPS",
]
