"""The build set (models/losses/build_set.py) is derived from the weight table (#1966)."""

from __future__ import annotations

import pytest

from spectramr.config.schemas.loss import (
    LossComponentConfig,
    LossConfigSchema,
    ReconstructionLossesConfig,
)
from spectramr.models.losses.build_set import declared_build_set, strategy_resolved
from spectramr.models.losses.weights import build_loss_weight_table


def test_a_declared_list_term_is_built() -> None:
    cfg = LossConfigSchema(
        output_domain="image",
        image_losses=[LossComponentConfig(name="ssim", weight=0.5)],
    )
    assert declared_build_set(cfg) == {"ssim": 0.5}


def test_a_disabled_list_entry_is_not_built() -> None:
    cfg = LossConfigSchema(
        output_domain="image",
        image_losses=[LossComponentConfig(name="ssim", weight=0.5, enabled=False)],
    )
    assert declared_build_set(cfg) == {}


def test_a_reconstruction_lambda_is_declared_but_computed_inline() -> None:
    cfg = LossConfigSchema(reconstruction=ReconstructionLossesConfig(enable_l1=True, lambda_l1=2.0))
    assert declared_build_set(cfg) == {}
    assert build_loss_weight_table(cfg)["l1"].weight == 2.0


def test_enable_false_switches_a_written_term_off() -> None:
    cfg = LossConfigSchema(
        reconstruction=ReconstructionLossesConfig(enable_ssim=False, lambda_ssim=0.3)
    )
    assert "ssim" not in declared_build_set(cfg)


def test_an_enable_flag_alone_declares_the_schema_default() -> None:
    cfg = LossConfigSchema(reconstruction=ReconstructionLossesConfig(enable_l1=True))
    assert build_loss_weight_table(cfg)["l1"].weight == pytest.approx(10.0)


def test_policy_exclusion_removes_a_declared_term() -> None:
    cfg = LossConfigSchema(
        output_domain="image",
        image_losses=[LossComponentConfig(name="ssim", weight=0.5)],
        policy={"exclude_defaults": ["ssim"]},
    )
    assert declared_build_set(cfg) == {}


def test_a_composed_component_is_declared() -> None:
    cfg = LossConfigSchema.model_validate(
        {"composed": {"components": [{"name": "ssim", "weight": 0.4, "enabled": True}]}}
    )
    assert declared_build_set(cfg) == {"ssim": 0.4}


def test_strategy_managed_names_are_not_built() -> None:
    cfg = LossConfigSchema(
        output_domain="image",
        image_losses=[LossComponentConfig(name="ssim", weight=0.5)],
    )
    spec = build_loss_weight_table(cfg)["ssim"]
    assert not strategy_resolved(spec)
    assert declared_build_set(cfg) == {"ssim": 0.5}


def test_stargan_diversity_is_computed_inline_not_built() -> None:
    # stargan_v2_strategy.py:382 computes the diversity term from losses.gan.lambda_diversity.
    from pathlib import Path

    from spectramr.config.settings import TrainingSettings

    root = Path(__file__).resolve().parents[4]
    arm = root / "experiments/inprogress/mrixfields2026/baselines/baseline_stargan_v2_task3.yaml"
    cfg = TrainingSettings.from_yaml(str(arm))
    assert build_loss_weight_table(cfg.losses)["diversity"].weight == 1.0
    assert "diversity" not in declared_build_set(cfg.losses)
