"""Tests for ``GenerativeTrainingStrategy`` (Phase-3 maximum-likelihood loop).

The strategy is model-agnostic: it computes a likelihood in priority order
(``log_prob`` -> ``training_loss`` -> ``(z, log_det)`` -> reconstruction
fallback). We exercise each path by calling ``_compute_losses_impl`` directly
with a fake model, bypassing the heavy ``BaseTrainingStrategy.__init__`` via
``__new__`` (the method only touches ``kwargs['model']`` here).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies.generative import (  # noqa: E402
    GenerativeTrainingStrategy,
)


def _strategy() -> GenerativeTrainingStrategy:
    # Skip __init__ (needs a full TrainingEnvironment); we only call the
    # pure loss method with an explicit model.
    return GenerativeTrainingStrategy.__new__(GenerativeTrainingStrategy)


class _LogProbModel(torch.nn.Module):
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), -5.0)


class _TrainingLossModel(torch.nn.Module):
    def training_loss(self, x: torch.Tensor) -> torch.Tensor:
        return (x**2).mean()


class _FlowModel(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        z = torch.zeros(x.shape[0], 4)
        log_det = torch.zeros(x.shape[0])
        return z, log_det


class _PlainModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


def test_log_prob_path() -> None:
    s = _strategy()
    x = torch.randn(3, 1, 8, 8)
    out = s._compute_losses_impl(x, x, epoch=0, model=_LogProbModel())
    assert "g_total_loss" in out
    assert torch.isclose(out["g_total_loss"], torch.tensor(5.0), atol=1e-5)


def test_training_loss_path() -> None:
    s = _strategy()
    x = torch.ones(2, 1, 4, 4)
    out = s._compute_losses_impl(x, x, epoch=0, model=_TrainingLossModel())
    assert torch.isclose(out["g_total_loss"], torch.tensor(1.0), atol=1e-6)


def test_flow_z_logdet_path() -> None:
    s = _strategy()
    x = torch.randn(2, 1, 4, 4)
    out = s._compute_losses_impl(x, x, epoch=0, model=_FlowModel())
    # z=0, log_det=0 -> log p = -0.5*d*log(2pi); NLL = +0.5*d*log(2pi)
    expected = 0.5 * 4 * math.log(2 * math.pi)
    assert torch.isclose(out["g_total_loss"], torch.tensor(expected), atol=1e-4)


def test_reconstruction_fallback_path() -> None:
    s = _strategy()
    x = torch.randn(2, 1, 4, 4)
    out = s._compute_losses_impl(x, x, epoch=0, model=_PlainModel())
    # pred == x -> MSE 0
    assert torch.isclose(out["g_total_loss"], torch.tensor(0.0), atol=1e-6)
    assert "loss_recon" in out


def test_explicit_model_does_not_touch_env() -> None:
    """Passing ``model=`` must not evaluate ``self.generator_model`` (→ ``self.env``).

    Regression for the eager ``dict.get`` default: ``kwargs.get('model',
    self.generator_model)`` accessed ``self.env.generator`` even when the caller
    supplied ``model=``, crashing every explicit-model call on a bare strategy.
    The bare ``__new__`` instance here has no ``env`` attribute at all.
    """
    s = _strategy()
    assert not hasattr(s, "env")  # bare __new__ — no environment attached
    out = s._compute_losses_impl(
        torch.randn(2, 1, 4, 4), torch.randn(2, 1, 4, 4), epoch=0, model=_PlainModel()
    )
    assert "g_total_loss" in out


def test_standard_normal_nll_helper() -> None:
    z = torch.zeros(5, 16)
    log_det = torch.zeros(5)
    nll = GenerativeTrainingStrategy._standard_normal_nll(z, log_det)
    expected = 0.5 * 16 * math.log(2 * math.pi)
    assert torch.isclose(nll, torch.tensor(expected), atol=1e-4)


class _ConditionalLogProbModel(torch.nn.Module):
    conditions_on_input = True

    def log_prob(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        assert condition is not None
        return -((x - condition) ** 2).flatten(1).sum(dim=1)


def test_conditional_log_prob_receives_the_input() -> None:
    s = _strategy()
    inp, tgt = torch.zeros(2, 1, 4, 4), torch.ones(2, 1, 4, 4)
    out = s._compute_losses_impl(inp, tgt, epoch=0, model=_ConditionalLogProbModel())
    assert torch.isclose(out["loss_nll"], torch.tensor(16.0))


def test_conditional_log_prob_without_input_raises() -> None:
    s = _strategy()
    with pytest.raises(ValueError, match="conditions on the input"):
        s._compute_losses_impl(None, torch.ones(1, 1, 2, 2), epoch=0, model=_ConditionalLogProbModel())


# ── validation: the arm's config scores log p(x), never the latent ─────────────

_ARMS = ("glow_brain", "equivariant_flow_so2")


class _KnownLogProb(torch.nn.Module):
    """log p(x) = -c_b per sample; ``forward`` raises so the latent is never scored."""

    def __init__(self, per_sample: torch.Tensor) -> None:
        super().__init__()
        self.per_sample = per_sample

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.training, "validation must score the model in eval mode"
        return self.per_sample.to(x.dtype)

    def forward(self, x: torch.Tensor):  # pragma: no cover - must not run
        raise AssertionError("validation scored the flow latent instead of log_prob")


def _arm_strategy(arm: str, model: torch.nn.Module):
    from pathlib import Path
    from types import SimpleNamespace

    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    path = (
        Path(__file__).resolve().parents[5] / f"experiments/inprogress/normalizing_flows/{arm}.yaml"
    )
    settings = TrainingSettings.from_yaml(str(path))
    cls = TrainingStrategyFactory().get_strategy_class(settings)
    assert cls is GenerativeTrainingStrategy
    s = cls.__new__(cls)
    s.config = settings
    s.env = SimpleNamespace(generator=model)
    s.device = torch.device("cpu")
    return s, settings


@pytest.mark.parametrize("arm", _ARMS)
def test_validation_scores_bits_per_dim_from_log_prob(arm: str) -> None:
    from spectramr.infrastructure.services.metric_keys import early_stop_monitor_candidates

    log_px = torch.tensor([-100.0, -300.0])
    model = _KnownLogProb(log_px)
    model.train()
    s, settings = _arm_strategy(arm, model)
    x = torch.randn(2, 1, 8, 8)

    metrics = s.validation_step(torch.zeros_like(x), x, batch_idx=0)

    d = x[0].numel()
    expected_bpd = float((-log_px / (d * math.log(2.0))).mean())
    assert metrics["nll_bits_per_dim"] == pytest.approx(expected_bpd, rel=1e-6)
    assert metrics["val_loss"] == pytest.approx(200.0, rel=1e-6)
    assert model.training, "validation restores the train mode it found"
    # Selection reads this very key.
    monitor = settings.early_stopping.metric
    assert settings.early_stopping.enabled
    assert any(k in metrics for k in early_stop_monitor_candidates(monitor))


def test_validation_flow_contract_without_log_prob() -> None:
    s, _ = _arm_strategy("glow_brain", _FlowModel())
    x = torch.randn(3, 1, 2, 2)  # D = 4 = the fake latent width
    metrics = s.validation_step(x, x)
    expected_bpd = 0.5 * math.log(2 * math.pi) / math.log(2.0)  # z=0, log_det=0
    assert metrics["nll_bits_per_dim"] == pytest.approx(expected_bpd, rel=1e-5)


def test_validation_raises_without_a_likelihood() -> None:
    s, _ = _arm_strategy("glow_brain", _PlainModel())
    x = torch.randn(2, 1, 4, 4)
    with pytest.raises(TypeError, match="exact likelihood"):
        s.validation_step(x, x)


def test_validation_rejects_pixel_metrics() -> None:
    from spectramr.core.metrics.computer import create_validation_metrics_computer

    s, _ = _arm_strategy("glow_brain", _KnownLogProb(torch.zeros(2)))
    s._validation_computer = create_validation_metrics_computer(["nll_bits_per_dim", "psnr"])
    x = torch.randn(2, 1, 4, 4)
    with pytest.raises(ValueError, match="psnr"):
        s.validation_step(x, x)
