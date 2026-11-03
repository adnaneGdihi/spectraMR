"""The fMRI Riemannian DFC arm declares no k-space physics and no image objective.

It generates SPD functional-connectivity matrices, so there are no coils, no measurements
and no undersampling. A stale `coils.processing_mode: rss` made the audit treat it as a
k-space reconstruction with inert physics (health:physics_config).
"""

from __future__ import annotations

from pathlib import Path

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.strategies.fmri_surface_strategies import (
    RiemannianDFCDiffusionStrategy,
)
from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker

ARM = (
    Path(__file__).resolve().parents[3]
    / "experiments/inprogress/fmri_2026/idea_4_riemannian_dfc_diffusion.yaml"
)


def _load() -> TrainingSettings:
    return TrainingSettings.from_yaml(str(ARM))


def test_the_arm_is_not_treated_as_a_kspace_reconstruction() -> None:
    settings = _load()
    assert settings.data.coils.processing_mode not in ("rss", "svd", "flatten")
    assert ConfigHealthChecker().check_physics_config(settings) == []


def test_the_empty_image_loss_list_matches_the_strategy_objective() -> None:
    settings = _load()
    assert settings.losses.image_losses == []
    assert RiemannianDFCDiffusionStrategy.inline_losses == frozenset()
    assert RiemannianDFCDiffusionStrategy.folds_image_losses is False
