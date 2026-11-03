"""pc_sampler_consistency: registered, integrates the VE-SDE, and re-imposes the measurement."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.diffusion.samplers import get_sampler, list_available
from spectramr.models.diffusion.samplers.pc_sampler_consistency import PCSamplerConsistency
from spectramr.models.diffusion.score_sde.sde import VESDE

pytestmark = pytest.mark.unit


def _point_mass_score(x0: torch.Tensor, sde: VESDE, floor: float = 1e-2):
    """Exact score of N(x0, (sigma(t)^2 + floor^2) I): the data distribution is a point mass."""

    def score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        var = sde.sigma(t).view(-1, 1, 1, 1) ** 2 + floor**2
        return (x0 - x) / var

    return score


def test_registered_under_the_dc_method_name() -> None:
    assert "pc_sampler_consistency" in list_available()


def test_unconditional_chain_recovers_the_point_mass() -> None:
    torch.manual_seed(0)
    sde = VESDE(sigma_min=0.01, sigma_max=5.0, N=200)
    x0 = torch.rand(1, 1, 8, 8)
    sampler = get_sampler(
        "pc_sampler_consistency", score_fn=_point_mass_score(x0, sde), sde=sde, n_steps=200
    )
    zero_mask = torch.zeros(1, 1, 8, 8)
    out = sampler.sample(torch.zeros(1, 1, 8, 8, dtype=torch.complex64), zero_mask)
    assert out.shape == x0.shape and not torch.is_complex(out)
    assert torch.allclose(out, x0, atol=0.05)


def test_full_mask_returns_the_measured_image() -> None:
    sde = VESDE(sigma_min=0.01, sigma_max=5.0, N=10)
    x_true = torch.rand(2, 1, 8, 8)
    wrong_prior = _point_mass_score(torch.zeros_like(x_true), sde)
    sampler = PCSamplerConsistency(wrong_prior, sde, n_steps=10)
    mask = torch.ones(1, 1, 8, 8)
    out = sampler.sample(fft2c(x_true) * mask, mask)
    assert torch.allclose(out, x_true, atol=1e-5)


def test_consistency_pins_the_measured_bins_on_a_conjugate_symmetric_mask() -> None:
    """On a support closed under k -> -k the real projection is lossless, so y is held exactly."""
    sde = VESDE(sigma_min=0.01, sigma_max=5.0, N=4)
    sampler = PCSamplerConsistency(lambda x, t: torch.zeros_like(x), sde, n_steps=4)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 3:6, :] = 1.0  # rows 3, 4, 5 around the centre row 4
    y = fft2c(torch.rand(1, 1, 8, 8)) * mask
    x = torch.randn(1, 1, 8, 8)
    out = sampler._consistency(x, y, mask)
    assert not torch.is_complex(out)
    assert torch.allclose(fft2c(out) * mask, y, atol=1e-5)
    assert torch.allclose(fft2c(out) * (1 - mask), fft2c(x) * (1 - mask), atol=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [({"n_steps": 0}, ValueError), ({"snr": 0.0}, ValueError), ({"dc_weight": 0.0}, ValueError)],
)
def test_invalid_knobs_raise(kwargs: dict, error: type) -> None:
    sde = VESDE(sigma_min=0.01, sigma_max=5.0, N=4)
    base = {"score_fn": lambda x, t: x, "sde": sde, "n_steps": 4}
    with pytest.raises(error):
        PCSamplerConsistency(**{**base, **kwargs})


def test_non_sde_and_missing_measurement_raise() -> None:
    with pytest.raises(TypeError):
        PCSamplerConsistency(lambda x, t: x, object(), n_steps=2)
    sampler = PCSamplerConsistency(lambda x, t: x, VESDE(0.01, 5.0, N=2), n_steps=2)
    with pytest.raises(ValueError):
        sampler.sample(None, torch.ones(1, 1, 4, 4))
    with pytest.raises(TypeError):
        sampler.sample(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
