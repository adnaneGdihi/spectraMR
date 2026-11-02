"""``CRLBMRFPulseDesignStrategy``: the expected-CRLB pulse design of MRF §4.

Validation emits the CRLB objective it declares (#2828) on a fixed prior grid, the
training step projects the design onto the arm's ``mrf`` SAR/slew bound, and the real
arm resolves through the strategy factory and the model registry to a design whose
CRLB falls under optimisation.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.config.schemas.validation import ValidationConfigSchema
from spectramr.infrastructure.physics.acquisition.mrf_acquisition_manifold import (
    AcquisitionBound,
)
from spectramr.infrastructure.training.strategies import base as strategy_base
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.infrastructure.training.strategies.mrf_acquisition_strategies import (
    CRLBMRFPulseDesignStrategy,
)
from spectramr.infrastructure.training.strategies.rng_streams import rank_seeded_generator
from spectramr.models.generators.mrf_pulse_design import MRFPulseDesign

_ARM = (
    Path(__file__).resolve().parents[5]
    / "experiments/inprogress/mrf_2026/idea_4_crlb_pulse_design.yaml"
)


def _strategy(
    compute: list[str], generator: nn.Module, bound: AcquisitionBound | None = None
) -> CRLBMRFPulseDesignStrategy:
    """A strategy carrying only what its steps read; no environment is built."""
    strategy = object.__new__(CRLBMRFPulseDesignStrategy)
    strategy.env = SimpleNamespace(generator=generator)
    strategy.device = torch.device("cpu")
    strategy.config = SimpleNamespace(
        validation=ValidationConfigSchema(scoring={"compute": compute, "primary": "crlb_objective"})
    )
    strategy.lambda_beltrami = 1e-3
    strategy.t1_range_ms, strategy.t2_range_ms = (100.0, 3000.0), (20.0, 500.0)
    strategy.prior_samples, strategy.te_ms = 16, 2.0
    strategy.bound = bound or AcquisitionBound(4.0, 200.0)
    strategy._prior_rng = rank_seeded_generator(0)
    return strategy


def _design() -> MRFPulseDesign:
    return MRFPulseDesign(n_tr=12, alpha_init_rad=1.0, tr_init_ms=10.0)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(3, 1, 8), torch.rand(3, 1, 8)


def test_validation_returns_every_declared_emitted_metric() -> None:
    declared = CRLBMRFPulseDesignStrategy.capabilities.emitted_metrics
    strategy = _strategy(["crlb_objective"], _design())
    with torch.no_grad():
        metrics = strategy.validation_step(*_batch(), batch_idx=0)
    assert declared <= metrics.keys()
    assert all(math.isfinite(metrics[k]) for k in declared)


def test_validation_objective_is_the_training_objective_on_the_grid() -> None:
    from spectramr.infrastructure.physics.acquisition.mrf_crlb import log_grid_prior

    generator = _design()
    strategy = _strategy(["crlb_objective"], generator)
    with torch.no_grad():
        metrics = strategy.validation_step(*_batch())
    t1, t2 = log_grid_prior(16, strategy.t1_range_ms, strategy.t2_range_ms)
    with torch.no_grad():
        expected = strategy._crlb_objective(generator.design(), t1, t2)
    assert metrics["crlb_objective"] == pytest.approx(float(expected), rel=1e-5)


def test_generator_mode_is_restored() -> None:
    generator = _design().train()
    with torch.no_grad():
        _strategy(["crlb_objective"], generator).validation_step(*_batch())
    assert generator.training


def test_registry_metrics_still_run_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_registry_step(self, input_batch, target_batch, **kwargs):
        calls.append(1)
        return {"psnr": 1.0}

    monkeypatch.setattr(MetricsMixin, "validation_step", fake_registry_step)
    strategy = _strategy(["psnr", "crlb_objective"], _design())
    with torch.no_grad():
        metrics = strategy.validation_step(*_batch())
    assert calls == [1]
    assert {"psnr", "crlb_objective"} <= metrics.keys()


def test_no_registry_metric_is_scored_on_the_pulse() -> None:
    """The real ``MetricsMixin`` path: an all-emitted list adds nothing (#2839)."""
    strategy = _strategy(["crlb_objective"], _design())
    with torch.no_grad():
        keys = set(strategy.validation_step(*_batch()))
    assert keys == set(CRLBMRFPulseDesignStrategy.capabilities.emitted_metrics)


def test_unowned_validation_name_raises() -> None:
    strategy = _strategy(["crlb_objective", "not_a_metric"], _design())
    with torch.no_grad(), pytest.raises(ValueError, match="not_a_metric"):
        strategy.validation_step(*_batch())


def test_a_non_pulse_generator_raises() -> None:
    strategy = _strategy(["crlb_objective"], nn.Identity())
    with pytest.raises(ValueError, match="mrf_pulse_design"):
        strategy._compute_losses_impl(torch.rand(2, 8), torch.rand(2, 8), epoch=0)


def test_missing_mrf_bound_raises() -> None:
    with pytest.raises(ValueError, match="sar_max_w_per_kg"):
        CRLBMRFPulseDesignStrategy._resolve_bound(SimpleNamespace(sar_max_w_per_kg=4.0))


def test_a_binding_sar_bound_changes_the_objective() -> None:
    """The bound reaches training: a binding ceiling moves the CRLB, a slack one does not."""
    generator = _design()
    x, y = _batch()
    slack = _strategy(["crlb_objective"], generator, AcquisitionBound(4.0, 200.0))
    same = _strategy(["crlb_objective"], generator, AcquisitionBound(8.0, 400.0))
    tight = _strategy(["crlb_objective"], generator, AcquisitionBound(0.1, 200.0))
    objs = [s._compute_losses_impl(x, y, epoch=0)["crlb_objective"] for s in (slack, same, tight)]
    assert objs[0] == pytest.approx(float(objs[1]))
    assert float(objs[2]) != pytest.approx(float(objs[0]))


def test_arm_resolves_and_its_design_lowers_the_crlb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reachability: YAML -> strategy factory -> model registry -> CRLB falls."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
    from spectramr.models.factories.model_factory import get_model_factory

    config = TrainingSettings.from_yaml(str(_ARM))
    strategy_cls = TrainingStrategyFactory().get_strategy_class(config)
    assert strategy_cls is CRLBMRFPulseDesignStrategy
    kwargs = {**config.model.model_kwargs, "n_tr": 24}  # the arm's model, shortened for CPU
    generator = get_model_factory().create_generator(config.model.model_type, **kwargs)
    assert isinstance(generator, MRFPulseDesign)

    def _light_init(self, env, **_):
        self.env, self.config, self.device = env, env.config, torch.device("cpu")

    monkeypatch.setattr(strategy_base.BaseTrainingStrategy, "__init__", _light_init)
    strategy = strategy_cls(env=SimpleNamespace(generator=generator, config=config))
    assert strategy.bound == AcquisitionBound(4.0, 200.0)
    assert strategy.prior_samples == 64 and strategy.te_ms == 2.0

    opt = torch.optim.Adam(generator.parameters(), lr=2e-2)
    x, y = torch.rand(2, 1, 8), torch.rand(2, 1, 8)
    with torch.no_grad():
        before = strategy.validation_step(x, y)
    for _ in range(25):
        loss = strategy._compute_losses_impl(x, y, epoch=0)["g_total_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = strategy.validation_step(x, y)
    assert before["crlb_objective"] == pytest.approx(before["crlb_objective_initial"])
    assert after["crlb_objective"] < before["crlb_objective"]


class _Wrapper(nn.Module):
    """A DDP-shaped wrapper: the design lives under ``.module``."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.module = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


def test_initial_crlb_survives_a_training_wrapper() -> None:
    strategy = _strategy(["crlb_objective"], _Wrapper(_design()))
    with torch.no_grad():
        metrics = strategy.validation_step(*_batch())
    assert math.isfinite(metrics["crlb_objective_initial"])


def test_initial_crlb_without_the_design_generator_raises() -> None:
    class _Pulse(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.rand(x.shape[0], 6, 2) + 0.5

    strategy = _strategy(["crlb_objective"], _Pulse())
    with torch.no_grad(), pytest.raises(TypeError, match="mrf_pulse_design"):
        strategy.validation_step(*_batch())


def test_the_objective_is_a_declared_train_column() -> None:
    keys = object.__new__(CRLBMRFPulseDesignStrategy).declared_metric_keys()
    assert "crlb_objective" in keys
    assert CRLBMRFPulseDesignStrategy.reports_train_metrics is False


def test_te_at_or_above_the_reachable_tr_floor_raises() -> None:
    design = _design()  # tr_init 10 ms, k_max 0.9 -> floor 5.5 ms
    CRLBMRFPulseDesignStrategy._check_te_below_tr(design, 2.0)
    with pytest.raises(ValueError, match="shortest reachable TR"):
        CRLBMRFPulseDesignStrategy._check_te_below_tr(design, 5.5)
    with pytest.raises(TypeError, match="mrf_pulse_design"):
        CRLBMRFPulseDesignStrategy._check_te_below_tr(nn.Identity(), 2.0)
