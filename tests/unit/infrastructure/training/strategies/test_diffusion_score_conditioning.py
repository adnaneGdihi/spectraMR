"""A score-based epsilon arm with condition_on_input samples conditioned on the ULF input (#2379).

Inverts the issue's repro on the validation sampler: under one RNG state, two different
inputs must give two different samples. CPU, a one-conv denoiser, no model built.
"""

from __future__ import annotations

import functools
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.schedulers.diffusion_scheduler import DiffusionScheduler
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from spectramr.models.generators.score_based_diffusion_generator import (
    ScoreBasedDiffusionGenerator,
)

pytestmark = pytest.mark.unit

T = 8


class _Eps(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.c = nn.Conv2d(in_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.c(x)


def _sample(conditioned: bool, cond: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(1)
    gen = ScoreBasedDiffusionGenerator(
        denoising_model=_Eps(2 if conditioned else 1), timesteps=T, device="cpu"
    ).eval()
    host = MagicMock()
    host.num_timesteps = T
    host.scheduler = DiffusionScheduler(num_timesteps=T, beta_schedule="linear")
    host._resolve_validation_sampling_steps = lambda: None
    host._conditions_on_input = lambda: conditioned
    host._maybe_condition_on_input = functools.partial(
        DiffusionTrainingStrategy._maybe_condition_on_input, host
    )

    def forward_chunked(x_in, t_vec, _is_latent, _kw):
        return gen(x_in, timesteps=t_vec)

    torch.manual_seed(0)
    with torch.no_grad():
        return DiffusionTrainingStrategy._sample_pixel_epsilon(
            host, cond, {}, cond.shape, forward_chunked
        )


def test_conditioned_sample_depends_on_the_input() -> None:
    y1 = torch.rand(2, 1, 8, 8)
    y2 = 5 * torch.rand(2, 1, 8, 8)
    assert not torch.equal(_sample(True, y1), _sample(True, y2))


def test_unconditional_control_ignores_the_input() -> None:
    y1 = torch.rand(2, 1, 8, 8)
    y2 = 5 * torch.rand(2, 1, 8, 8)
    assert torch.equal(_sample(False, y1), _sample(False, y2))
