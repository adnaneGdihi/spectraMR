"""Annealed Langevin + hard data consistency: projection, ladder, oracle convergence."""

from __future__ import annotations

import itertools

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.diffusion.samplers import SamplerRegistry
from spectramr.models.diffusion.samplers.annealed_langevin_dc import (
    AnnealedLangevinDCSampler,
    complex_to_interleaved,
    interleaved_to_complex,
    project_measured,
)


def _sampler(**over) -> AnnealedLangevinDCSampler:
    kw = {
        "sigma_min": 1e-3,
        "sigma_max": 1.0,
        "num_levels": 6,
        "steps_per_level": 4,
        "step_size": 0.3,
        "seed": 0,
    }
    kw.update(over)
    return AnnealedLangevinDCSampler(**kw)


def test_interleave_round_trip_matches_the_project_layout() -> None:
    x = torch.randn(2, 4, 8, 8)
    z = interleaved_to_complex(x)
    assert torch.equal(z.real, x[:, 0::2]) and torch.equal(z.imag, x[:, 1::2])
    assert torch.equal(complex_to_interleaved(z), x)


def test_projection_full_mask_returns_measurement_and_empty_mask_is_identity() -> None:
    x = torch.randn(1, 2, 8, 8)
    y = fft2c(torch.randn(1, 1, 8, 8, dtype=torch.complex64))
    full = project_measured(x, y, torch.ones(1, 1, 8, 8))
    assert torch.allclose(interleaved_to_complex(full), ifft2c(y), atol=1e-5)
    assert torch.allclose(project_measured(x, y, torch.zeros(1, 1, 8, 8)), x, atol=1e-5)


def test_ladder_is_geometric_and_descending() -> None:
    s = _sampler().sigmas()
    assert s[0] == pytest.approx(1.0) and s[-1] == pytest.approx(1e-3)
    ratios = [b / a for a, b in itertools.pairwise(s)]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)


def test_registry_resolves_the_sampler_by_name_and_alias() -> None:
    kw = {
        "sigma_min": 0.01,
        "sigma_max": 1.0,
        "num_levels": 2,
        "steps_per_level": 1,
        "step_size": 0.1,
        "seed": 1,
    }
    assert isinstance(SamplerRegistry.get("annealed_langevin_dc", **kw), AnnealedLangevinDCSampler)
    assert isinstance(SamplerRegistry.get("annealed_langevin", **kw), AnnealedLangevinDCSampler)


@pytest.mark.parametrize("bad", [{"step_size": 2.0}, {"num_levels": 1}, {"sigma_min": 2.0}])
def test_invalid_settings_raise(bad) -> None:
    with pytest.raises(ValueError):
        _sampler(**bad)


def test_oracle_score_recovers_the_truth_and_keeps_measured_lines() -> None:
    torch.manual_seed(0)
    truth = torch.randn(2, 2, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., ::3, :] = 1.0
    y = fft2c(interleaved_to_complex(truth)) * mask
    calls = []

    def oracle(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return -(x - truth) / sigma.view(-1, 1, 1, 1) ** 2

    sampler = _sampler()
    zf = complex_to_interleaved(ifft2c(y))
    out = sampler.sample(oracle, zf, y, mask)
    assert len(calls) == sampler.num_score_evaluations
    assert (out - truth).pow(2).mean() < 0.1 * (zf - truth).pow(2).mean()
    k_out = fft2c(interleaved_to_complex(out))
    assert torch.allclose(k_out * mask, y, atol=1e-4)
    # Private generator: the same call gives the same reconstruction.
    assert torch.equal(sampler.sample(oracle, zf, y, mask), out)
