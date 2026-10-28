r"""Schema for PR-CC: Physics-Residual Conformal hallucination certificate.

Defines the ``training.phys_residual_conformal`` sub-block and the top-level
``TrainingConfigPhysResidualConformal`` that mounts it. The strategy
(:class:`~spectramr.infrastructure.training.strategies.phys_residual_conformal_strategy.PhysResidualConformalStrategy`)
is a degenerate, no-gradient calibration pass: it loads a frozen tissue-parameter
estimator, computes the physics-residual nonconformity score over a held-out
calibration split (re-rendering the observed contrast through the Bloch forward),
and emits the split-conformal quantile + empirical coverage, optionally with
Mondrian (per-:math:`B_0`) field stratification.

Mounting (mirrors B-1 equivariance_conformal): the typed wrapper is registered in
``spectramr/config/schemas/training/__init__.py``::

    from .phys_residual_conformal import TrainingConfigPhysResidualConformal
    _MODE_DISPATCH["phys_residual_conformal"] = "TrainingConfigPhysResidualConformal"

The sub-block is also typed as an optional field of ``TrainingStrategyConfigSchema`` so a
retired key is refused when the config loads, not when the strategy is built.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..renames import reject_renamed_keys
from .base import BaseTrainingConfigSchema


class PhysResidualConformalConfig(BaseModel):
    """Hyperparameters for the physics-residual conformal calibration (PR-CC)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = Field(default=0.1, gt=0.0, lt=1.0, description="Mis-coverage rate (0.1 -> 90%).")
    tr_ms: float = Field(..., gt=0.0, description="Repetition time TR (ms) for re-render.")
    te_ms: float = Field(..., gt=0.0, description="Echo time TE (ms) for re-render.")
    ti_ms: float = Field(
        default=0.0, ge=0.0, description="Inversion time TI (ms); used for IR/FLAIR."
    )
    contrast_type: Literal[
        "spin_echo", "se", "t2", "inversion_recovery", "ir", "flair", "gradient_echo"
    ] = Field(default="spin_echo", description="Contrast routed to the Bloch forward.")
    stratify_by: Literal["none", "b0"] = Field(
        default="none",
        description="'b0' enables Mondrian per-field calibration (residual-shift remedy).",
    )
    calibration_fraction: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Fraction of the val loader used as the calibration split (rest = test).",
    )
    output_artefact: str | None = Field(
        default=None, description="Optional JSON path for the certificate report."
    )

    # Retired prior-checkpoint field (#647); the refusal names the removal record.
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training.phys_residual_conformal"))  # type: ignore[arg-type]
    )


class TrainingConfigPhysResidualConformal(BaseTrainingConfigSchema):
    """Top-level training config wrapper that mounts the PR-CC sub-block."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    phys_residual_conformal: PhysResidualConformalConfig


__all__ = [
    "PhysResidualConformalConfig",
    "TrainingConfigPhysResidualConformal",
]
