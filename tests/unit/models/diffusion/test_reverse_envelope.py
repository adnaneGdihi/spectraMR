"""Band-energy envelope and the divisor that enforces it (``reverse_envelope``)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.diffusion.kspace_process import _radial_band_partition, paired_magnitude
from spectramr.models.diffusion.reverse_envelope import (
    normalize_band_energy,
    observed_band_envelope,
    per_coefficient_support,
)

pytestmark = pytest.mark.unit

B, C, H, W = 1, 2, 32, 32
NB = 4


def _bands():
    return _radial_band_partition(H, W, NB, 1e-8, torch.device("cpu"))


def _band_rms(x: torch.Tensor, support: torch.Tensor, b: int) -> float:
    mag = paired_magnitude(x)[0, 0]
    sel = (_bands() == b) & (support[0, 0] > 0)
    return float(mag[sel].pow(2).mean().sqrt())


def test_per_coefficient_support_pairs_down_real_channels() -> None:
    x = torch.zeros(B, 4, H, W)
    per_channel = torch.zeros(B, 4, H, W)
    per_channel[:, 0::2, 3] = 1.0
    per_channel[:, 1::2, 3] = 1.0
    assert per_coefficient_support(per_channel, x).shape == (B, 2, H, W)
    broadcast = torch.ones(B, 1, H, W)
    assert per_coefficient_support(broadcast, x) is broadcast


def test_populated_bands_carry_the_measured_rms() -> None:
    torch.manual_seed(0)
    meas = torch.randn(B, C, H, W)
    obs = torch.ones(B, 1, H, W)
    env = observed_band_envelope(meas, obs, log_scaled=False, num_bands=NB)
    for b in range(NB):
        assert env[0, b] == pytest.approx(_band_rms(meas, obs, b), rel=1e-5)


def test_empty_bands_inherit_the_nearest_populated_band() -> None:
    """Outer empty bands take the inner neighbour; leading empty bands the outer one."""
    torch.manual_seed(0)
    band = _bands()
    meas = torch.randn(B, C, H, W)
    obs = ((band == 1) | (band == 2)).float()[None, None]
    env = observed_band_envelope(meas, obs, log_scaled=False, num_bands=NB)
    assert env[0, 0] == pytest.approx(float(env[0, 1]))
    assert env[0, 3] == pytest.approx(float(env[0, 2]))
    assert bool((env > 0).all())


def test_an_estimate_within_the_ratio_is_untouched() -> None:
    torch.manual_seed(0)
    meas = torch.randn(B, C, H, W)
    obs = torch.ones(B, 1, H, W)
    env = observed_band_envelope(meas, obs, log_scaled=False, num_bands=NB)
    out, divisor = normalize_band_energy(meas * 1.2, env, obs, ratio=1.3, log_scaled=False)
    assert torch.equal(divisor, torch.ones_like(divisor))
    assert torch.allclose(out, meas * 1.2)


@pytest.mark.parametrize("log_scaled", [False, True])
def test_an_overshooting_band_is_brought_to_the_ratio(log_scaled) -> None:
    torch.manual_seed(0)
    meas = torch.rand(B, C, H, W) + 0.1
    obs = torch.ones(B, 1, H, W)
    env = observed_band_envelope(meas, obs, log_scaled=log_scaled, num_bands=NB)
    support = torch.zeros(B, 1, H, W)
    support[..., 8:, :] = 1.0
    x0 = meas * 5.0
    out, divisor = normalize_band_energy(x0, env, support, ratio=1.3, log_scaled=log_scaled)
    assert bool((divisor > 1.0).any())
    phys = torch.expm1 if log_scaled else (lambda m: m)
    for b in range(NB):
        sel = (_bands() == b) & (support[0, 0] > 0)
        if not bool(sel.any()):
            continue
        rms = float(phys(paired_magnitude(out)[0, 0][sel]).pow(2).mean().sqrt())
        assert rms <= 1.3 * float(env[0, b]) * (1 + 1e-4)
    off = support[0, 0] == 0
    assert torch.equal(out[..., off], x0[..., off]), "coefficients off the support must not move"


def test_the_divisor_preserves_phase() -> None:
    torch.manual_seed(0)
    meas = torch.randn(B, C, H, W)
    obs = torch.ones(B, 1, H, W)
    env = observed_band_envelope(meas, obs, log_scaled=False, num_bands=NB)
    x0 = torch.randn(B, C, H, W) * 10.0
    out, _ = normalize_band_energy(x0, env, obs, ratio=1.0, log_scaled=False)
    before = torch.atan2(x0[:, 1], x0[:, 0])
    after = torch.atan2(out[:, 1], out[:, 0])
    assert torch.allclose(before, after, atol=1e-5)


def test_one_divisor_per_band_keeps_relative_magnitudes() -> None:
    torch.manual_seed(0)
    meas = torch.randn(B, C, H, W)
    obs = torch.ones(B, 1, H, W)
    env = observed_band_envelope(meas, obs, log_scaled=False, num_bands=NB)
    x0 = torch.randn(B, C, H, W) * 10.0
    out, _ = normalize_band_energy(x0, env, obs, ratio=1.0, log_scaled=False)
    ratio = paired_magnitude(out)[0, 0] / paired_magnitude(x0)[0, 0]
    band = _bands()
    for b in range(NB):
        r = ratio[band == b]
        assert torch.allclose(r, r[0].expand_as(r), rtol=1e-4)
