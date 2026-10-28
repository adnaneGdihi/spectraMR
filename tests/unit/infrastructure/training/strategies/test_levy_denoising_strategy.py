"""Unit tests for LevyDenoisingStrategy (LM-09).

The base ``__init__`` stands up AMP, the loss computer and DI services, so the
strategy is built with ``object.__new__`` and its own setup is run against a
real ``training`` schema. The training seam (``_generate_predictions``) and the
validation sampler are exercised on 16x16 tensors.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
from spectramr.config.schemas.training.levy_denoising import LevyDenoisingTrainingConfigSchema
from spectramr.infrastructure.training.strategies.levy_denoising_strategy import (
    SAMPLERS,
    LevyDenoisingStrategy,
    noise_ladder,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory


class _Spy(nn.Module):
    """1x1 conv denoiser that records the tensor it was handed."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x.detach().clone())
        return self.conv(x)


def _strategy(alpha: float = 1.7, in_channels: int = 3, **levy) -> LevyDenoisingStrategy:
    training = TrainingStrategyConfigSchema.model_validate(
        {"levy_denoising": {"alpha": alpha, "sampling_steps": 4, **levy}}
    )
    config = SimpleNamespace(
        training=training,
        model=SimpleNamespace(in_channels=in_channels, out_channels=1),
        run=SimpleNamespace(seed=7),
    )
    strat = object.__new__(LevyDenoisingStrategy)
    strat.config = config
    strat.env = SimpleNamespace(generator=_Spy(in_channels))
    strat.logging_service = None
    strat._verify_strategy_config = lambda expected_modes: None  # type: ignore[method-assign]
    strat._apply_input_conditioning = lambda image, ctx: image  # type: ignore[method-assign]
    strat._setup_strategy_specific_components()
    return strat


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    return torch.rand(2, 1, 16, 16, generator=g), torch.rand(2, 1, 16, 16, generator=g)


class TestSchema:
    def test_alpha_is_required_and_bounded(self) -> None:
        with pytest.raises(ValidationError):
            LevyDenoisingTrainingConfigSchema()
        with pytest.raises(ValidationError):
            LevyDenoisingTrainingConfigSchema(alpha=2.5)

    def test_noise_range_must_be_ordered(self) -> None:
        with pytest.raises(ValidationError, match="sigma_min"):
            LevyDenoisingTrainingConfigSchema(alpha=1.7, sigma_min=1.0, sigma_max=0.5)

    def test_unknown_sampler_and_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LevyDenoisingTrainingConfigSchema(alpha=1.7, sampler="ddpm")
        with pytest.raises(ValidationError):
            LevyDenoisingTrainingConfigSchema(alpha=1.7, typo=1)

    def test_routed_by_the_factory(self) -> None:
        path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS["levy_diffusion"]
        assert path.endswith("levy_denoising_strategy.LevyDenoisingStrategy")


class TestSetup:
    def test_missing_block_raises(self) -> None:
        strat = object.__new__(LevyDenoisingStrategy)
        strat.config = SimpleNamespace(training=TrainingStrategyConfigSchema.model_validate({}))
        strat._verify_strategy_config = lambda expected_modes: None  # type: ignore[method-assign]
        with pytest.raises(ValueError, match=r"training\.levy_denoising"):
            strat._setup_strategy_specific_components()

    def test_wrong_in_channels_raises(self) -> None:
        strat = _strategy(in_channels=2)
        cond, target = _pair()
        with pytest.raises(ValueError, match="in_channels"):
            strat._generate_predictions(cond, {}, {"hr": target})


class TestTrainingSeam:
    def test_network_sees_condition_noisy_target_and_level(self) -> None:
        strat = _strategy()
        cond, target = _pair()
        pred, inter = strat._generate_predictions(cond, {}, {"hr": target})
        seen = strat.env.generator.seen[-1]
        assert inter is None and pred.shape == target.shape and pred.requires_grad
        assert torch.equal(seen[:, :1], cond)
        assert not torch.allclose(seen[:, 1:2], target)
        level = seen[:, 2]
        assert torch.allclose(level, level[:, :1, :1].expand_as(level))

    def test_alpha_changes_the_training_loss(self) -> None:
        cond, target = _pair()
        losses = {}
        for alpha in (1.2, 2.0):
            strat = _strategy(alpha=alpha)
            pred, _ = strat._generate_predictions(cond, {}, {"hr": target})
            losses[alpha] = (pred - target).square().mean().item()
        assert losses[1.2] != pytest.approx(losses[2.0], rel=1e-3)

    def test_noise_clip_bounds_the_draw(self) -> None:
        strat = _strategy(alpha=1.1, noise_clip=3.0)
        eps = strat.draw_noise((4, 1, 32, 32), torch.device("cpu"), strat._levy_rng)
        assert eps.abs().max().item() <= 3.0


class TestValidationSampler:
    def _validate(self, strat: LevyDenoisingStrategy, cond: torch.Tensor) -> torch.Tensor:
        with (
            patch.object(
                LevyDenoisingStrategy,
                "_prepare_generator_inputs",
                lambda self, ctx, x, **k: (x, {}),
            ),
            torch.no_grad(),
        ):
            return strat._validation_forward(cond, {})

    def test_runs_the_declared_number_of_steps_and_is_reproducible(self) -> None:
        strat = _strategy()
        cond, _ = _pair()
        first = self._validate(strat, cond)
        assert len(strat.env.generator.seen) == 4
        assert torch.equal(first, self._validate(strat, cond))

    def test_output_depends_on_condition_and_alpha(self) -> None:
        cond, other = _pair()
        heavy, gauss = _strategy(alpha=1.5), _strategy(alpha=2.0)
        out = self._validate(heavy, cond)
        assert not torch.allclose(out, self._validate(heavy, other))
        assert not torch.allclose(out, self._validate(gauss, cond))

    def test_dlim_step_keeps_the_noise_direction(self) -> None:
        strat = _strategy()
        cond, _ = _pair()
        fixed = torch.full((2, 1, 16, 16), 0.25)
        strat.denoise = lambda c, x, s: fixed  # type: ignore[method-assign]
        out = SAMPLERS["dlim"](strat, cond, torch.Generator().manual_seed(0))
        assert torch.equal(out, fixed)

    def test_noise_ladder_is_geometric(self) -> None:
        ladder = noise_ladder(2.0, 0.02, 3)
        assert ladder == pytest.approx([2.0, 0.2, 0.02])
        assert noise_ladder(2.0, 0.02, 1) == [2.0]
