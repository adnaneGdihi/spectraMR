"""through_plane_pve: slice-profile averaging across slices (multislice), a kz window (3-D)."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc

from .invariants import GEOMETRIES, assert_axis_invariants

NAME = "through_plane_pve"
M = AcquisitionMode
MS = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0))
VOL = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0))


def _stack(d=21, bright=10, h=4, w=4):
    x = torch.zeros(1, 1, d, h, w, dtype=torch.complex64)
    x[:, :, bright] = 1.0 + 2.0j
    return x


@pytest.mark.parametrize("geom", ["ms_interleaved", "ms_sequential", "volume", "volume_centric"])
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


MS_THICK = AcquisitionGeometry(
    mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_thickness_mm=4.0
)


def _taps(thickness, sigma, half):
    """Tap j: the rect(thickness) * Gaussian(sigma) profile averaged over slice extent j (pitch 1),
    by a direct 2-D Riemann sum; normalised."""
    n = 400
    v = (torch.arange(n, dtype=torch.float64) + 0.5) / n * thickness - thickness / 2
    w = []
    for j in range(-half, half + 1):
        z = j + (torch.arange(n, dtype=torch.float64) + 0.5) / n - 0.5
        g = torch.exp(-((z[:, None] - v[None, :]) ** 2) / (2 * sigma**2))
        w.append(float(g.mean()))
    w = torch.tensor(w, dtype=torch.float64)
    return w / w.sum()


@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_a_bright_slice_spreads_with_the_profile_weights(theta):
    # 4 mm slices on a 5 mm pitch: thk = 0.8 pitch, sigma = theta * thk / 2.
    thk = 0.8
    out = apply_degradation(NAME, _stack(), theta, geometry=MS_THICK)
    w = _taps(thk, theta * thk / 2, 3)
    got = out[0, 0, 7:14, 0, 0]
    torch.testing.assert_close(got, w.to(torch.float32) * (1.0 + 2.0j), rtol=0, atol=2e-4)
    assert float(out.abs().sum()) == pytest.approx(float(_stack().abs().sum()), rel=1e-5)


def test_the_spread_starts_continuously_from_zero():
    for geom in (MS, MS_THICK):
        full = (apply_degradation(NAME, _stack(), 1.0, geometry=geom) - _stack()).abs().sum()
        tiny = (apply_degradation(NAME, _stack(), 1e-4, geometry=geom) - _stack()).abs().sum()
        assert 0 <= float(tiny) < 1e-3 * float(full)


def test_the_kz_window_is_the_declared_hamming_weighting():
    d, theta = 12, 0.7
    x = torch.randn(1, 1, d, 5, 5, dtype=torch.complex64)
    out = apply_degradation(NAME, x, theta, geometry=VOL)
    f = (torch.arange(d) - d // 2) / d
    w = 1.0 - theta * (1.0 - (0.54 + 0.46 * torch.cos(2 * math.pi * f)))
    expected = ifftnc(fftnc(x, dims=(-3,)) * w.view(1, 1, d, 1, 1), dims=(-3,))
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)
    k = fftnc(out, dims=(-3,))
    ratio = (k / fftnc(x, dims=(-3,))).real
    assert float(ratio[0, 0, d // 2].mean()) == pytest.approx(1.0, abs=1e-5)  # DC is untouched
    assert float(ratio[0, 0, 0].mean()) == pytest.approx(1 - theta * 0.92, abs=1e-5)  # edge: 0.08


def test_3d_partitions_are_mixed_only_along_kz():
    x = torch.randn(1, 1, 8, 6, 6, dtype=torch.complex64)
    out = apply_degradation(NAME, x, 1.0, geometry=VOL)
    row = apply_degradation(NAME, x[:, :, :, :1, :1], 1.0, geometry=VOL)
    torch.testing.assert_close(out[:, :, :, :1, :1], row, rtol=0, atol=1e-6)


def test_slice_2d_names_the_proxy():
    with pytest.raises(ValueError, match="slice_pve"):
        apply_degradation(NAME, torch.ones(1, 1, 8, 8), 0.5)
