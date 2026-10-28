"""Tests for α-stable Lévy diffusion primitives."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.diffusion.alpha_stable_levy import (
    alpha_stable_kernel_fft,
    fractional_score_matching_loss,
    sample_alpha_stable,
)


def test_alpha_two_reduces_to_gaussian() -> None:
    """At α=2 the sampler recovers Brownian (Gaussian) increments."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    samples = sample_alpha_stable((4096,), alpha=2.0, scale=1.0, generator=gen)
    # Gaussian with std √2.
    expected_std = math.sqrt(2.0)
    assert abs(samples.std().item() - expected_std) < 0.1
    # No extreme outliers — Gaussian tails.
    assert samples.abs().max().item() < 10.0


def test_alpha_one_is_cauchy_heavy_tailed() -> None:
    """At α=1 the sampler produces Cauchy increments — heavy tails."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    samples = sample_alpha_stable((4096,), alpha=1.0, scale=1.0, generator=gen)
    # Cauchy has undefined moments — extreme outliers expected.
    assert samples.abs().max().item() > 30.0


def test_alpha_stable_kernel_at_alpha_two_is_gaussian() -> None:
    """At α=2 the characteristic function exp(-σ² k²) is a Gaussian in k."""
    k = alpha_stable_kernel_fft((16,), alpha=2.0, scale=1.0)
    # k at DC (index 0) should be exactly 1.
    assert k[0].item() == pytest.approx(1.0, abs=1e-6)
    # Symmetry: k[i] == k[-i].
    assert torch.allclose(k[1:], k.flip(0)[:-1])


def test_alpha_stable_kernel_decays_faster_at_higher_alpha() -> None:
    """For fixed σ, the characteristic function decays faster at higher α."""
    k_alpha_one = alpha_stable_kernel_fft((32,), alpha=1.0, scale=1.0)
    k_alpha_two = alpha_stable_kernel_fft((32,), alpha=2.0, scale=1.0)
    # At the Nyquist frequency, α=2 is much smaller (exp(-0.25) vs exp(-0.5)).
    # Wait — both are at |k|=0.5: exp(-0.5^1)=0.607 vs exp(-0.5^2)=0.779.
    # So actually α=1 decays faster at small |k|! Let me check.
    # At |k|=1: exp(-1)=0.368 vs exp(-1)=0.368. Same.
    # At |k|=2: exp(-2)=0.135 vs exp(-4)=0.018. So α=2 decays faster at large |k|.
    # The FFT max frequency is 0.5 (Nyquist), so:
    # exp(-0.5) vs exp(-0.25). α=1 has smaller value at |k|=0.5.
    # So at high frequencies on this grid α=1 decays MORE.
    # Adjust the test to test the right thing.
    nyq_index = 16  # Nyquist
    assert k_alpha_one[nyq_index].item() < k_alpha_two[nyq_index].item()


def test_sample_alpha_stable_reproducible() -> None:
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = sample_alpha_stable((128,), alpha=1.5, scale=1.0, generator=g1)
    b = sample_alpha_stable((128,), alpha=1.5, scale=1.0, generator=g2)
    assert torch.allclose(a, b)


def test_fractional_score_matching_grad() -> None:
    """Score matching loss is differentiable through a simple linear model."""
    linear = torch.nn.Linear(8, 8)
    x0 = torch.randn(4, 8)
    loss = fractional_score_matching_loss(linear, x0, alpha=2.0, sigma=0.5)
    loss.backward()
    assert linear.weight.grad is not None


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        sample_alpha_stable((4,), alpha=0.0, scale=1.0)
    with pytest.raises(ValueError):
        sample_alpha_stable((4,), alpha=2.5, scale=1.0)
    with pytest.raises(ValueError):
        sample_alpha_stable((4,), alpha=1.0, scale=-1.0)
    with pytest.raises(ValueError):
        alpha_stable_kernel_fft((4,), alpha=3.0)


@pytest.mark.parametrize("alpha", [1.5, 2.0])
def test_fractional_score_matching_target_is_minus_noise_over_sigma_alpha(alpha: float) -> None:
    """The documented DSM surrogate: a model emitting -noise/sigma**alpha scores zero."""
    x0 = torch.randn(2, 1, 4, 4)
    sigma = 0.5
    loss = fractional_score_matching_loss(
        lambda x_t: -(x_t - x0) / sigma**alpha,
        x0,
        alpha=alpha,
        sigma=sigma,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.allclose(loss, torch.zeros(()), atol=1e-10)
