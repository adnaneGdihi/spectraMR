"""Tests for CertifiedRobustnessStrategy (A-8.3 CertRob, component C)."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from spectramr.config.schemas.training.certified_robustness import CertifiedRobustnessConfig
from spectramr.infrastructure.training.strategies.certified_robustness_strategy import (
    CertifiedRobustnessStrategy,
)
from spectramr.models.losses.lipschitz_penalty import LipschitzSpectralPenalty


def _high_gain_model() -> nn.Sequential:
    # weight = 4 * I -> sigma_max is EXACTLY 4.0 (so the witness bound is exact).
    lin = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        lin.weight.copy_(4.0 * torch.eye(8))
    return nn.Sequential(lin)


def _strategy_with(lam: float) -> CertifiedRobustnessStrategy:
    strat = object.__new__(CertifiedRobustnessStrategy)
    strat.env = types.SimpleNamespace(generator=_high_gain_model())
    strat._lambda_lipschitz = lam
    strat._lip_penalty = LipschitzSpectralPenalty(target_sigma=1.0, n_power_iterations=3)
    return strat


# --------------------------------------------------------------------------- #
# Penalty injection into the training loss
# --------------------------------------------------------------------------- #


def test_compute_losses_adds_penalty_to_g_total_loss_when_lambda_positive(monkeypatch) -> None:
    # The penalty must attach to the CANONICAL grad-carrying key g_total_loss (the
    # key base.py reads for backward) — not loss_total, which the parent never emits.
    import spectramr.infrastructure.training.strategies.certified_robustness_strategy as mod

    base = torch.tensor(1.0, requires_grad=True)
    monkeypatch.setattr(
        mod.AdversarialRobustnessStrategy,
        "_compute_losses_impl",
        lambda self, **k: {"g_total_loss": base},
    )
    strat = _strategy_with(lam=0.5)
    out = strat._compute_losses_impl(input_batch=None, target_batch=None, epoch=0)
    assert "lipschitz_penalty" in out
    assert float(out["lipschitz_penalty"]) > 0.0  # high-gain model is penalised
    assert float(out["g_total_loss"]) > 1.0  # penalty added on top of the base loss
    assert out["g_total_loss"].requires_grad  # still differentiable


def test_compute_losses_no_penalty_when_total_not_grad_carrying(monkeypatch) -> None:
    # If the parent returns a detached / non-grad total, the penalty must NOT attach
    # (adding to a non-leaf detached tensor would not train the weights -> inert #16).
    import spectramr.infrastructure.training.strategies.certified_robustness_strategy as mod

    monkeypatch.setattr(
        mod.AdversarialRobustnessStrategy,
        "_compute_losses_impl",
        lambda self, **k: {"g_total_loss": torch.tensor(1.0)},  # requires_grad=False
    )
    strat = _strategy_with(lam=0.5)
    out = strat._compute_losses_impl(input_batch=None, target_batch=None, epoch=0)
    assert "lipschitz_penalty" not in out


def test_compute_losses_no_penalty_when_lambda_zero(monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.certified_robustness_strategy as mod

    base = torch.tensor(1.0, requires_grad=True)
    monkeypatch.setattr(
        mod.AdversarialRobustnessStrategy,
        "_compute_losses_impl",
        lambda self, **k: {"g_total_loss": base},
    )
    strat = _strategy_with(lam=0.0)  # the control arm
    out = strat._compute_losses_impl(input_batch=None, target_batch=None, epoch=0)
    assert "lipschitz_penalty" not in out
    assert float(out["g_total_loss"]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Validation emits the global certified-bound witnesses
# --------------------------------------------------------------------------- #


def test_validation_step_emits_bound_witnesses(monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.certified_robustness_strategy as mod
    from spectramr.core.metrics.lipschitz_bound_metrics import (
        ComposedSpectralNormBound,
        MaxLayerSpectralNorm,
    )

    monkeypatch.setattr(
        mod.AdversarialRobustnessStrategy,
        "validation_step",
        lambda self, *a, **k: {"val_psnr": 30.0},
    )
    strat = object.__new__(CertifiedRobustnessStrategy)
    strat.env = types.SimpleNamespace(generator=_high_gain_model())
    strat._bound_metric = ComposedSpectralNormBound()
    strat._max_layer_metric = MaxLayerSpectralNorm()
    out = strat.validation_step(None, None)
    assert out["val_psnr"] == 30.0  # parent metrics preserved
    assert out["val_composed_spectral_norm_bound"] == pytest.approx(4.0, rel=0.05)
    assert out["val_max_layer_spectral_norm"] == pytest.approx(4.0, rel=0.05)


# --------------------------------------------------------------------------- #
# Setup reads the config block (knobs not hardcoded; #15)
# --------------------------------------------------------------------------- #


def test_setup_reads_config_block(monkeypatch) -> None:
    import spectramr.infrastructure.training.strategies.certified_robustness_strategy as mod

    monkeypatch.setattr(
        mod.AdversarialRobustnessStrategy.__mro__[1],
        "_setup_strategy_specific_components",
        lambda self: None,  # skip the heavy reconstruction setup; the knob read stays live
    )
    cfg_block = CertifiedRobustnessConfig(
        lambda_lipschitz=0.25, target_sigma=2.0, n_power_iterations=4, epsilon=0.05, attack_steps=7
    )
    strat = object.__new__(CertifiedRobustnessStrategy)
    strat.config = types.SimpleNamespace(training=types.SimpleNamespace(certified_robustness=cfg_block))
    strat.env = types.SimpleNamespace(generator=_high_gain_model())
    strat.logging_service = None
    strat._setup_strategy_specific_components()
    assert strat._lambda_lipschitz == 0.25
    assert strat._target_sigma == 2.0
    assert strat._n_power_iters == 4
    assert strat.epsilon == 0.05  # adversarial knob read from config, not hardcoded
    assert strat.attack_steps == 7
    assert isinstance(strat._lip_penalty, LipschitzSpectralPenalty)


# --------------------------------------------------------------------------- #
# Registration + schema mount + schema validation
# --------------------------------------------------------------------------- #


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "certified_robustness" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "certified_robustness" in TrainingStrategyConfigSchema.model_fields


def test_schema_rejects_unknown_attack_norm() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CertifiedRobustnessConfig(attack_norm="garbage")
