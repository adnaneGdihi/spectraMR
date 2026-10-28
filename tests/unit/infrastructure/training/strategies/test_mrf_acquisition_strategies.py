"""``CRLBMRFPulseDesignStrategy`` validation emits the CRLB objective it declares (#2828).

The strategy's only Fisher/CRLB quantity was written to the training loss dict
alone. Validation went through ``MetricsMixin.validation_step``, which returns
registry metrics only, so ``validation.scoring.compute: [crlb_objective]`` named
a column no validation pass produced. These tests call ``validation_step`` the
way ``pipelines/train.py`` does, under ``torch.no_grad``, with a stub generator.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.config.schemas.validation import ValidationConfigSchema
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.infrastructure.training.strategies.mrf_acquisition_strategies import (
    CRLBMRFPulseDesignStrategy,
)


def _strategy(compute: list[str], generator: nn.Module) -> CRLBMRFPulseDesignStrategy:
    """A strategy carrying only what ``validation_step`` reads; no model is built."""
    strategy = object.__new__(CRLBMRFPulseDesignStrategy)
    strategy.env = SimpleNamespace(generator=generator)
    strategy.device = torch.device("cpu")
    strategy.config = SimpleNamespace(
        validation=ValidationConfigSchema(
            scoring={"compute": compute, "primary": "crlb_objective"}
        )
    )
    return strategy


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    pulse_in = torch.rand(3, 6, 2) + 0.1  # [B, T, (alpha, TR)]
    prior = torch.rand(3, 5) + 0.5  # [B, 5] Bloch-parameter prior samples
    return pulse_in, prior


def test_validation_returns_every_declared_emitted_metric() -> None:
    declared = CRLBMRFPulseDesignStrategy.capabilities.emitted_metrics
    assert "crlb_objective" in declared
    strategy = _strategy(["crlb_objective"], nn.Linear(2, 2))
    with torch.no_grad():
        metrics = strategy.validation_step(*_batch(), batch_idx=0)
    assert declared <= metrics.keys()
    assert math.isfinite(metrics["crlb_objective"])


def test_validation_objective_is_the_training_objective() -> None:
    generator = nn.Linear(2, 2)
    pulse_in, prior = _batch()
    strategy = _strategy(["crlb_objective"], generator)
    with torch.no_grad():
        metrics = strategy.validation_step(pulse_in, prior)
    pulse = generator(pulse_in)
    expected = -CRLBMRFPulseDesignStrategy._toy_fisher_trace(
        pulse, prior.clone().requires_grad_(True)
    )
    assert metrics["crlb_objective"] == pytest.approx(float(expected.detach()), rel=1e-5)


def test_generator_mode_is_restored() -> None:
    generator = nn.Linear(2, 2).train()
    with torch.no_grad():
        _strategy(["crlb_objective"], generator).validation_step(*_batch())
    assert generator.training


def test_registry_metrics_still_run_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_registry_step(self, input_batch, target_batch, **kwargs):
        calls.append(1)
        return {"psnr": 1.0}

    monkeypatch.setattr(MetricsMixin, "validation_step", fake_registry_step)
    strategy = _strategy(["psnr", "crlb_objective"], nn.Linear(2, 2))
    with torch.no_grad():
        metrics = strategy.validation_step(*_batch())
    assert calls == [1]
    assert {"psnr", "crlb_objective"} <= metrics.keys()


def test_no_registry_metric_is_scored_on_the_pulse() -> None:
    """The real ``MetricsMixin`` path, not a stub: an all-emitted list adds nothing (#2839).

    This used to hold only because the strategy skipped the mixin; the computer
    itself substituted PSNR/SSIM/MSE for the empty registry list.
    """
    strategy = _strategy(["crlb_objective"], nn.Linear(2, 2))
    with torch.no_grad():
        assert set(strategy.validation_step(*_batch())) == {"crlb_objective"}


def test_unowned_validation_name_raises() -> None:
    strategy = _strategy(["crlb_objective", "not_a_metric"], nn.Linear(2, 2))
    with torch.no_grad(), pytest.raises(ValueError, match="not_a_metric"):
        strategy.validation_step(*_batch())
