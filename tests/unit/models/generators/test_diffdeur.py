"""DiffDeuR's score-distillation step draws t from its prior's horizon (#2442).

It drew ``randint(0, 1000)`` and weighted by ``1 - t / 1000`` whatever horizon
the prior was trained on. The linear form of that weight is not the prior's own
ᾱ either; #2442 keeps that half open.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from spectramr.models.generators.diffdeur import DiffDeuR


class _PriorSpy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.steps: list[torch.Tensor] = []

    def forward(self, x, t):
        self.steps.append(t)
        return torch.zeros_like(x)


def test_score_distillation_draws_t_from_the_priors_horizon():
    """PLANTED: 30 draws from [0, 1000) all landing below 29 has probability ~1e-46."""
    torch.manual_seed(0)
    model = DiffDeuR(hidden_dim=16, num_timesteps=29)
    model.prior = _PriorSpy()
    model.enhance(torch.rand(1, 1, 8, 8), num_steps=30)
    steps = torch.cat(model.prior.steps)
    assert steps.numel() == 30
    assert int(steps.min()) >= 0 and int(steps.max()) < 29


def test_the_horizon_defaults_to_the_shared_constant():
    from spectramr.models.constants import DEFAULT_DIFFUSION_TIMESTEPS

    assert DiffDeuR(hidden_dim=16).num_timesteps == DEFAULT_DIFFUSION_TIMESTEPS
