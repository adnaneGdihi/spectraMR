"""Unit tests for the Poisson-resetting reverse SDE sampler (LM-06)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.config.schemas.training.levy_denoising import LevyDenoisingTrainingConfigSchema
from spectramr.infrastructure.training.strategies.levy_denoising_strategy import (
    SAMPLERS,
    LevyDenoisingStrategy,
)
from spectramr.infrastructure.training.strategies.resetting_sde_sampler import (
    noise_ladder,
    reset_index,
    resetting_sde_sampler,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

_RESET = {"sampler": "resetting_sde", "reset_sigma": 0.5, "max_resets": 2}


class _Counting(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 1, 1)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.conv(x)


def _strategy(**levy: object) -> LevyDenoisingStrategy:
    block = {"alpha": 2.0, "sampling_steps": 6, **levy}
    config = SimpleNamespace(
        training=TrainingStrategyConfigSchema.model_validate({"levy_denoising": block}),
        model=SimpleNamespace(in_channels=3, out_channels=1),
        run=SimpleNamespace(seed=11),
    )
    strat = object.__new__(LevyDenoisingStrategy)
    strat.config = config
    strat.env = SimpleNamespace(generator=_Counting())
    strat.logging_service = None
    strat._verify_strategy_config = lambda expected_modes: None  # type: ignore[method-assign]
    strat._apply_input_conditioning = lambda image, ctx: image  # type: ignore[method-assign]
    strat._setup_strategy_specific_components()
    return strat


def _cond() -> torch.Tensor:
    return torch.rand(3, 1, 16, 16, generator=torch.Generator().manual_seed(5))


def _sample(strat: LevyDenoisingStrategy) -> torch.Tensor:
    with torch.no_grad():
        return resetting_sde_sampler(strat, _cond(), torch.Generator().manual_seed(9))


def _plain_em(strat: LevyDenoisingStrategy) -> torch.Tensor:
    """Reference Euler-Maruyama reverse VE SDE, written independently of the sampler."""
    cond, gen = _cond(), torch.Generator().manual_seed(9)
    sig = noise_ladder(strat.levy_cfg.sigma_max, strat.levy_cfg.sigma_min, 6)
    x = sig[0] * strat.draw_noise((3, 1, 16, 16), cond.device, gen)
    with torch.no_grad():
        for i, s in enumerate(sig):
            x0 = strat.denoise(cond, x, torch.full((3,), s))
            if i == len(sig) - 1:
                return x0
            var = s**2 - sig[i + 1] ** 2
            z = strat.draw_noise((3, 1, 16, 16), cond.device, gen)
            x = x + (var / s**2) * (x0 - x) + var**0.5 * z
    raise AssertionError("unreachable")


class TestSchema:
    def test_resetting_needs_the_gaussian_score(self) -> None:
        with pytest.raises(ValidationError, match=r"alpha=2\.0"):
            LevyDenoisingTrainingConfigSchema(alpha=1.7, sampler="resetting_sde")

    def test_reset_knobs_on_dlim_are_unread(self) -> None:
        with pytest.raises(ValidationError, match="read only by"):
            LevyDenoisingTrainingConfigSchema(alpha=2.0, reset_rate=1.0)

    @pytest.mark.parametrize("sigma", [None, 0.005, 5.0])
    def test_positive_rate_needs_an_in_range_reset_sigma(self, sigma: float | None) -> None:
        with pytest.raises(ValidationError, match="reset_sigma"):
            LevyDenoisingTrainingConfigSchema(
                alpha=2.0, sampler="resetting_sde", reset_rate=1.0, reset_sigma=sigma, max_resets=1
            )

    def test_reset_knobs_at_rate_zero_are_unread(self) -> None:
        with pytest.raises(ValidationError, match="unread at reset_rate=0"):
            LevyDenoisingTrainingConfigSchema(alpha=2.0, **_RESET)

    def test_positive_rate_without_resets_is_inert(self) -> None:
        with pytest.raises(ValidationError, match="never resets"):
            LevyDenoisingTrainingConfigSchema(
                alpha=2.0, sampler="resetting_sde", reset_rate=1.0, reset_sigma=0.5
            )

    def test_mode_is_routed_to_the_denoiser(self) -> None:
        path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["resetting_diffusion"]
        assert path.endswith("LevyDenoisingStrategy")
        assert SAMPLERS["resetting_sde"] is resetting_sde_sampler


class TestSampler:
    def test_rate_zero_is_the_plain_sde_draw_for_draw(self) -> None:
        strat = _strategy(sampler="resetting_sde")
        reference = _plain_em(_strategy(sampler="resetting_sde"))
        # Same draws; only float32-tensor vs Python-float rounding separates the two.
        torch.testing.assert_close(_sample(strat), reference, rtol=1e-5, atol=1e-6)
        assert strat.env.generator.calls == 6

    def test_positive_rate_resets_and_changes_the_sample(self) -> None:
        plain = _sample(_strategy(sampler="resetting_sde"))
        strat = _strategy(reset_rate=50.0, **_RESET)
        out = _sample(strat)
        jr = reset_index(torch.tensor(noise_ladder(2.0, 0.01, 6)), 0.5)
        assert 6 < strat.env.generator.calls <= 6 + 2 * (6 - 1 - jr)
        assert not torch.allclose(out, plain)
        assert torch.isfinite(out).all()

    def test_reset_index_is_nearest_in_log_space(self) -> None:
        assert reset_index(torch.tensor([2.0, 0.2, 0.02]), 0.5) == 1

    def test_training_seam_does_not_depend_on_the_sampler(self) -> None:
        cond, target = _cond(), torch.rand(3, 1, 16, 16)
        a, _ = _strategy(sampler="resetting_sde")._generate_predictions(cond, {}, {"hr": target})
        b, _ = _strategy(reset_rate=2.0, **_RESET)._generate_predictions(cond, {}, {"hr": target})
        assert torch.equal(a, b)
