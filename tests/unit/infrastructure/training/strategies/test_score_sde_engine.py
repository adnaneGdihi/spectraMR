"""Score-SDE engine: built from experiment_72's own config, reached through DiffusionTrainingStrategy.

The strategy is a namespace carrying the real ``DiffusionTrainingStrategy`` methods the engine
and the dispatch hooks call, so the test drives the production seams without building a run.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from spectramr.config.settings import TrainingSettings
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from spectramr.infrastructure.training.strategies.score_sde_engine import (
    ScoreSDEEngine,
    build_score_sde_engine,
)
from spectramr.models.diffusion.samplers.pc_sampler_consistency import PCSamplerConsistency
from spectramr.models.factories.model_factory import ModelFactory
from spectramr.models.generators.score_based_diffusion_generator import (
    ScoreBasedDiffusionGenerator,
)

pytestmark = pytest.mark.unit

ARM = Path("experiments/inprogress/diffusion/experiment_72_score_based_diffusion_sde.yaml")


class _Denoiser(nn.Module):
    """Records the time it is conditioned on."""

    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Conv2d(1, 1, 3, padding=1)
        self.seen_t: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.seen_t.append(t)
        return self.c(x)


class _Strategy(types.SimpleNamespace):
    """Carries the strategy's loss-ownership declaration, which the fold reads off the class."""

    inline_losses = DiffusionTrainingStrategy.inline_losses
    folds_image_losses = DiffusionTrainingStrategy.folds_image_losses
    _compute_losses_impl = DiffusionTrainingStrategy._compute_losses_impl
    validation_step = DiffusionTrainingStrategy.validation_step
    _generate_validation_prediction = DiffusionTrainingStrategy._generate_validation_prediction


@pytest.fixture(scope="module")
def config() -> TrainingSettings:
    return TrainingSettings.from_yaml(ARM)


def _strategy(config: TrainingSettings) -> types.SimpleNamespace:
    kw = config.model.model_kwargs
    gen = ScoreBasedDiffusionGenerator(
        denoising_model=_Denoiser(),
        device="cpu",
        sde_type=kw["sde_type"],
        sigma_min=kw["sigma_min"],
        sigma_max=kw["sigma_max"],
    )
    s = _Strategy(
        config=config,
        generator_model=gen,
        num_timesteps=config.training.diffusion.timesteps,
        env=types.SimpleNamespace(losses={"mse": nn.MSELoss()}),
        declared=[],
        metrics_calls=[],
    )
    for name in (
        "_declared_prediction_type",
        "_conditions_on_input",
        "_resolve_validation_sampling_steps",
        "_apply_builder_image_losses",
    ):
        setattr(s, name, types.MethodType(getattr(DiffusionTrainingStrategy, name), s))
    s._declare_model_input = lambda tensors, **kw: s.declared.append(kw["extra"])
    s._prepare_validation_data = lambda *a: None
    s._is_cold_diffusion = lambda: False
    s._compute_validation_metrics = lambda pred, *a, **k: (
        s.metrics_calls.append(pred) or {"val_psnr": 1.0}
    )
    return s


def test_engine_is_built_from_the_arm(config: TrainingSettings) -> None:
    strategy = _strategy(config)
    engine = build_score_sde_engine(strategy)
    assert isinstance(engine, ScoreSDEEngine)
    assert (engine.sde.sigma_min, engine.sde.sigma_max) == (0.01, 50.0)
    assert engine.dc_method == "pc_sampler_consistency"
    assert (engine.acceleration, engine.center_fraction) == (4.0, 0.08)
    assert engine.num_timesteps == 1000
    # The validation step count comes from the shared reverse-steps resolver.
    assert engine.sampling_steps == strategy._resolve_validation_sampling_steps()


def test_no_sde_means_no_engine(config: TrainingSettings) -> None:
    s = _strategy(config)
    s.generator_model = ScoreBasedDiffusionGenerator(denoising_model=_Denoiser(), device="cpu")
    assert build_score_sde_engine(s) is None


def test_training_step_runs_vesde_score_matching(config: TrainingSettings) -> None:
    s = _strategy(config)
    s._score_sde = build_score_sde_engine(s)
    target = torch.rand(2, 1, 16, 16, 1)
    out = DiffusionTrainingStrategy._compute_losses_impl(s, target, target, 0)
    assert torch.isfinite(out["loss_total"]) and out["loss_total"].requires_grad
    t = s.generator_model.denoising_model.seen_t[-1]
    assert t.is_floating_point() and t.shape == (2,)
    assert bool(((t >= 0) & (t <= 999)).all())
    assert s.declared[-1]["score_sde"]["sde_type"] == "vesde"


def test_validation_runs_the_pc_sampler_with_data_consistency(
    config: TrainingSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _strategy(config)
    s._score_sde = build_score_sde_engine(s)
    s._score_sde.sampling_steps = 3
    calls = []
    real = PCSamplerConsistency._consistency
    monkeypatch.setattr(
        PCSamplerConsistency,
        "_consistency",
        lambda self, x, y, m: calls.append(m) or real(self, x, y, m),
    )
    x = torch.rand(2, 1, 16, 16)
    metrics = DiffusionTrainingStrategy.validation_step(s, x, x, None, batch_idx=0)
    assert metrics == {"val_psnr": 1.0}
    pred = s.metrics_calls[-1]
    assert pred.shape == x.shape and torch.isfinite(pred).all()
    # One corrector and one predictor DC per level on x, plus one on x_mean.
    assert len(calls) == 3 * 3
    mask = calls[0]
    assert 0.0 < float(mask.mean()) < 1.0


def test_prediction_hook_routes_to_the_engine(config: TrainingSettings) -> None:
    s = _strategy(config)
    s._score_sde = build_score_sde_engine(s)
    s._score_sde.sampling_steps = 2
    x = torch.rand(1, 1, 16, 16)
    pred = DiffusionTrainingStrategy._generate_validation_prediction(s, x, x, None, None, {})
    assert pred.shape == x.shape


def test_unregistered_dc_method_raises(config: TrainingSettings) -> None:
    dc = config.physics.data_consistency.model_copy(update={"method": "hard"})
    physics = config.physics.model_copy(update={"data_consistency": dc})
    s = _strategy(config.model_copy(update={"physics": physics}))
    with pytest.raises(ConfigurationError, match="not a registered sampler"):
        build_score_sde_engine(s)


def test_production_setup_hook_builds_the_engine_for_a_factory_built_generator(
    config: TrainingSettings,
) -> None:
    """ModelFactory delivers the SDE knobs; the strategy's own setup hook builds the engine."""
    kw = config.model.model_kwargs
    gen = ModelFactory().create_generator(
        "score_based_diffusion",
        denoising_model={
            "type": "score_sde",
            "base_features": 8,
            "in_channels": 1,
            "out_channels": 1,
        },
        sde_type=kw["sde_type"],
        sigma_min=kw["sigma_min"],
        sigma_max=kw["sigma_max"],
        device="cpu",
    )
    assert (gen.sde_type, gen.sigma_min, gen.sigma_max) == ("vesde", 0.01, 50.0)
    strategy = object.__new__(DiffusionTrainingStrategy)
    strategy.config = config
    strategy.env = types.SimpleNamespace(
        generator=gen, config=config, model_type="score_based_diffusion"
    )
    strategy.logging_service = MagicMock()
    strategy._diffusion_initialized = True
    strategy.num_timesteps = config.training.diffusion.timesteps
    strategy._setup_strategy_specific_components()
    assert isinstance(strategy._score_sde, ScoreSDEEngine)
    target = torch.rand(2, 1, 8, 8)
    strategy.env.losses = {"mse": nn.MSELoss()}
    strategy._declare_model_input = lambda tensors, **k: None
    out = strategy._compute_losses_impl(target, target, 0)
    assert torch.isfinite(out["loss_total"]) and out["loss_total"].requires_grad


def test_a_subclass_that_overrides_a_hooked_step_is_refused(config: TrainingSettings) -> None:
    class _Override(_Strategy):
        def _compute_losses_impl(self, *a, **k):  # an EDM-style inline objective
            return {}

    s = _strategy(config)
    s.__class__ = _Override
    with pytest.raises(ConfigurationError, match="overrides"):
        build_score_sde_engine(s)


@pytest.mark.parametrize(
    ("pe_axis", "direction", "axis"), [(-2, None, -2), (-1, None, -1), (-2, "readout", -1)]
)
def test_measurement_lines_follow_the_declared_axis(
    config: TrainingSettings, pe_axis: int, direction: str | None, axis: int
) -> None:
    from spectramr.infrastructure.training.strategies.score_sde_config import line_axis

    cfg = types.SimpleNamespace(
        data=types.SimpleNamespace(acquisition=types.SimpleNamespace(phase_encode_axis=pe_axis)),
        undersampling=types.SimpleNamespace(mask_direction=direction),
    )
    assert line_axis(cfg) == axis
    engine = build_score_sde_engine(_strategy(config))
    engine.line_axis = axis
    mask = engine._mask(16, 16, torch.device("cpu"))[0, 0]
    lines = mask if axis == -2 else mask.T
    assert bool((lines == lines[:, :1]).all())  # each sampled line is fully acquired
