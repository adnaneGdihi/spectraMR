"""ScoreBasedDiffusionGenerator.forward: the strategy's x_t and t are used as given (#2379)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.models.generators.score_based_diffusion_generator import (
    ScoreBasedDiffusionGenerator,
)

pytestmark = pytest.mark.unit


class _Eps(nn.Module):
    """Records the timestep it receives; output depends on x only through a conv."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.c = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.last_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.last_t = t
        return self.c(x)


def _gen(in_channels: int = 1) -> ScoreBasedDiffusionGenerator:
    return ScoreBasedDiffusionGenerator(
        denoising_model=_Eps(in_channels), timesteps=8, device="cpu"
    ).eval()


def test_given_timesteps_input_is_not_renoised() -> None:
    gen = _gen()
    x_t = torch.rand(2, 1, 16, 16)
    t = torch.tensor([3, 5])
    with torch.no_grad():
        out = gen(x_t, timesteps=t)
        expected = gen.denoising_model.c(x_t)
    assert torch.equal(out, expected)
    assert torch.equal(gen.denoising_model.last_t, t)


def test_given_timesteps_consumes_no_rng() -> None:
    gen = _gen()
    x_t = torch.rand(1, 1, 8, 8)
    torch.manual_seed(0)
    before = torch.get_rng_state()
    with torch.no_grad():
        gen(x_t, timesteps=torch.tensor([2]))
    assert torch.equal(before, torch.get_rng_state())


def test_condition_channel_reaches_denoiser() -> None:
    gen = _gen(in_channels=2)
    x_t = torch.rand(1, 1, 8, 8)
    t = torch.tensor([4])
    with torch.no_grad():
        o1 = gen(torch.cat([x_t, torch.zeros_like(x_t)], dim=1), timesteps=t)
        o2 = gen(torch.cat([x_t, torch.ones_like(x_t)], dim=1), timesteps=t)
    assert not torch.equal(o1, o2)


def test_without_timesteps_keeps_self_noising() -> None:
    gen = _gen()
    x0 = torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        torch.manual_seed(0)
        a = gen(x0)
        torch.manual_seed(1)
        b = gen(x0)
    assert not torch.equal(a, b)


def test_strategy_dispatch_passes_timesteps() -> None:
    # DiffusionTrainingStrategy._forward_through_model passes t only to a forward that names it.
    from spectramr.infrastructure.training.strategies.mixins.utils import _callable_accepts_kwarg

    assert _callable_accepts_kwarg(_gen().forward, "timesteps")


def _vesde_gen() -> ScoreBasedDiffusionGenerator:
    return ScoreBasedDiffusionGenerator(
        denoising_model=_Eps(),
        timesteps=8,
        device="cpu",
        sde_type="vesde",
        sigma_min=0.01,
        sigma_max=50.0,
    ).eval()


def test_vesde_knobs_are_held() -> None:
    gen = _vesde_gen()
    assert (gen.sde_type, gen.sigma_min, gen.sigma_max) == ("vesde", 0.01, 50.0)
    assert _gen().sde_type is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sde_type": "vpsde", "sigma_min": 0.01, "sigma_max": 50.0},
        {"sde_type": "vesde"},
        {"sde_type": "vesde", "sigma_min": 5.0, "sigma_max": 1.0},
        {"sigma_min": 0.01, "sigma_max": 50.0},
    ],
)
def test_unknown_or_incomplete_sde_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ScoreBasedDiffusionGenerator(denoising_model=_Eps(), timesteps=8, device="cpu", **kwargs)


def test_vesde_refuses_the_discrete_path() -> None:
    gen = _vesde_gen()
    x = torch.rand(1, 1, 8, 8)
    with pytest.raises(RuntimeError, match="sde_type='vesde'"):
        gen(x)
    with pytest.raises(RuntimeError):
        gen.generate(x)
    with torch.no_grad():
        # A time index in [0, N) is the same conditioning domain the engine feeds (t * (N - 1)).
        assert gen(x, timesteps=torch.tensor([499.5])).shape == x.shape
        assert gen(x, timesteps=torch.tensor([3])).shape == x.shape
