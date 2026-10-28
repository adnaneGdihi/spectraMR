"""Tests for spectramr.data.m4raw_noise_units (exp 32 series-gain gate)."""

from __future__ import annotations

import torch

from spectramr.data.m4raw_noise_units import noise_power, noise_unit_rss


def _scan(sigma: float, n_avg: int = 1, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    s, c, h, w = 10, 4, 32, 32
    signal = torch.zeros(s, c, h, w, dtype=torch.complex64)
    signal[..., 12:20, 12:20] = 5.0
    noise = (
        sum(
            torch.complex(
                torch.randn(s, c, h, w, generator=g), torch.randn(s, c, h, w, generator=g)
            )
            for _ in range(n_avg)
        )
        * (sigma / (2**0.5))
        / n_avg
    )
    return signal + noise


def test_noise_power_recovers_the_injected_variance() -> None:
    assert abs(noise_power(_scan(0.5)) / 0.25 - 1) < 0.15


def test_noise_units_are_invariant_to_averaging() -> None:
    """Noise units are SNR units: an averaged series is brighter in them, as it must be."""
    single, averaged = _scan(0.5, 1), _scan(0.5, 4, seed=1)
    a = noise_unit_rss(single[5], noise_power(single))
    b = noise_unit_rss(averaged[5], noise_power(averaged))
    assert b[12:20, 12:20].mean() > 1.5 * a[12:20, 12:20].mean()


def test_phase_ramp_preserves_noise_power() -> None:
    """A registered series is a phase-ramped copy; its noise estimate must not move."""
    k = _scan(0.5)
    ramp = torch.exp(1j * 2 * torch.pi * 3.3 * torch.arange(32) / 32).to(k.dtype)
    assert abs(noise_power(k * ramp) / noise_power(k) - 1) < 0.02


def test_registered_copy_keeps_its_noise_power_given_the_source_support() -> None:
    """Registration's FFT round trip un-zeroes skipped lines; the source support restores f."""
    from spectramr.data.contrast_registration import RigidFit, apply_fit
    from spectramr.data.datasets.m4raw_dataset import _pe_support

    k = _scan(0.5)
    k[..., 24:] = 0  # 25 % of phase-encode columns never acquired
    registered, _ = apply_fit(
        k, RigidFit(shift_px=(0.0, 0.0, 0.0), slice_offset=0, rotation_deg=0.0)
    )
    support = _pe_support(k[:8])
    ratio = noise_power(registered, support=support) / noise_power(k)
    assert abs(ratio - 1) < 0.05
