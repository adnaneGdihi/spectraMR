"""VESDE.sigma is the one owner of the noise level that marginal_prob and sde read."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.score_sde.sde import VESDE

pytestmark = pytest.mark.unit


def test_sigma_spans_the_declared_range_geometrically() -> None:
    sde = VESDE(sigma_min=0.01, sigma_max=50.0, N=10)
    t = torch.tensor([0.0, 0.5, 1.0])
    sig = sde.sigma(t)
    assert torch.allclose(sig, torch.tensor([0.01, (0.01 * 50.0) ** 0.5, 50.0]), rtol=1e-5)


def test_marginal_std_and_diffusion_follow_sigma() -> None:
    sde = VESDE(sigma_min=0.01, sigma_max=50.0, N=10)
    x = torch.zeros(2, 1, 4, 4)
    t = torch.tensor([0.25, 0.75])
    mean, std = sde.marginal_prob(x, t)
    assert torch.equal(mean, x)
    assert torch.allclose(std[:, 0, 0, 0], sde.sigma(t))
    _, g = sde.sde(x, t)
    assert torch.allclose(g[:, 0, 0, 0] / sde.sigma(t), g[0, 0, 0, 0] / sde.sigma(t[:1]))
