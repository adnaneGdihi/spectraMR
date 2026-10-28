"""Native multislice and 3-D noise operators: invariants and one closed-form check each."""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.volume.noise import apodization_window
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc

M = AcquisitionMode
GEOM = {
    M.MULTISLICE_2D: AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None),
    M.VOLUME_3D: AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None),
}
DIMS = {M.MULTISLICE_2D: (-2, -1), M.VOLUME_3D: (-3, -2, -1)}
BOTH_MODES = ("complex_gaussian", "rician", "resolution_snr", "emi")
CASES = [(n, m) for n in BOTH_MODES for m in (M.MULTISLICE_2D, M.VOLUME_3D)] + [
    ("spike", M.VOLUME_3D)
]


def _x(shape=(1, 2, 4, 16, 16), seed=3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _apply(name, x, theta, mode, seed=1, **kw):
    return apply_degradation(name, x, theta, seed=seed, geometry=GEOM[mode], **kw)


def _residual(name, x, y) -> float:
    if name == "rician":  # a magnitude operator: compare magnitudes
        x, y = x.abs(), y.abs()
    return float((y - x).abs().pow(2).mean().sqrt())


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_declared_native(name, mode):
    assert DEGRADATION_REGISTRY[name].realisation[mode] is Realisation.NATIVE


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_theta_zero_is_identity(name, mode):
    if name == "complex_gaussian":
        pytest.skip("declared 40 dB pedestal; checked by the SNR test")
    x = _x()
    assert torch.equal(_apply(name, x, 0.0, mode), x)


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_monotone_finite_shape_deterministic(name, mode):
    x = _x()
    outs = [_apply(name, x, t, mode) for t in (0.0, 0.3, 0.6, 1.0)]
    for y in outs:
        assert y.shape == x.shape
        assert torch.isfinite(torch.view_as_real(y) if y.is_complex() else y).all()
    res = [_residual(name, x, y) for y in outs]
    assert all(a < b for a, b in pairwise(res)), res
    assert torch.equal(_apply(name, x, 0.6, mode), outs[2])


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_batch_invariant(name, mode):
    x = _x()
    alone = _apply(name, x, 0.6, mode)
    pair = _apply(name, torch.cat([x, 10 * _x(seed=9)]), 0.6, mode)
    torch.testing.assert_close(pair[:1], alone, rtol=1e-5, atol=1e-5)


def _snr_db(signal_k, noisy_k) -> float:
    p_sig = signal_k.abs().pow(2).mean()
    p_noise = (noisy_k - signal_k).abs().pow(2).mean()
    return float(10 * torch.log10(p_sig / p_noise))


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
@pytest.mark.parametrize("theta", [0.0, 0.5, 1.0])
def test_complex_gaussian_realised_snr(mode, theta):
    x = _x((1, 2, 8, 32, 32))
    y = _apply("complex_gaussian", x, theta, mode)
    declared = 40.0 + (2.0 - 40.0) * theta
    realised = _snr_db(fftnc(x, DIMS[mode]), fftnc(y, DIMS[mode]))
    assert abs(realised - declared) < 0.1, (realised, declared)


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_rician_background_mean(mode):
    x = torch.zeros(1, 1, 8, 32, 32)
    x[..., :, :, :16] = 1.0  # p99 = 1, right half is zero signal
    y = _apply("rician", x, 1.0, mode)
    sigma = 0.3
    bg = float(y[..., 16:].mean())
    assert abs(bg - sigma * math.sqrt(math.pi / 2)) < 0.05 * sigma * math.sqrt(math.pi / 2)


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_apodization_window_closed_form(mode):
    shape = (1, 2, 6, 12, 10)
    fwhm = 2.5
    w = apodization_window(shape, DIMS[mode], fwhm, torch.device("cpu"))
    s2 = (fwhm / 2.35482) ** 2
    full = w.expand(shape)[0, 0]
    centre = [s // 2 for s in shape[2:]]
    for axis, n in enumerate(shape[2:]):
        dim = axis - 3
        f = centred_freqs(n)
        idx = [slice(c, c + 1) for c in centre]
        idx[axis] = slice(None)
        line = full[tuple(idx)].reshape(-1)  # through the DC sample along one axis
        expected = torch.exp(-2 * math.pi**2 * s2 * f**2)
        if dim in DIMS[mode]:
            torch.testing.assert_close(line, expected, rtol=1e-6, atol=1e-7)
        else:
            torch.testing.assert_close(line, torch.ones(n), rtol=0, atol=0)


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_emi_rms_ratio_and_columns(mode):
    x = _x((1, 2, 4, 16, 32))
    theta = 0.7
    k_in = fftnc(x, DIMS[mode])
    k_out = fftnc(_apply("emi", x, theta, mode), DIMS[mode])
    field = k_out - k_in
    ratio = float(field.abs().pow(2).mean().sqrt() / k_in.abs().pow(2).mean().sqrt())
    assert abs(ratio - 0.3 * theta) < 0.02 * 0.3 * theta
    # Every tone lands in the same readout columns in every slice / kz plane.
    img = ifftnc(field, (-1,)).abs().pow(2).mean(dim=(0, 1, 3))  # (D, W)
    top = img.argmax(dim=-1)
    assert torch.equal(top, top[:1].expand_as(top))


def test_spike_volume_via_registry_hits_3d_kspace():
    x = _x((1, 2, 8, 16, 16))
    y = _apply("spike", x, 1.0, M.VOLUME_3D)
    hit = (fftnc(y - x, (-3, -2, -1)).abs() > 1e-3 * x.abs().max())[:, 0]
    assert 0.02 < float(hit.float().mean()) < 0.09


def test_spike_multislice_draws_independently_per_slice():
    p = 0.05
    one = _x((1, 2, 1, 64, 64), seed=5)
    x = one.expand(1, 2, 4, 64, 64).contiguous()
    y = _apply("spike", x, 1.0, M.MULTISLICE_2D, seed=2)
    diff = fftnc(y - x, (-2, -1))
    hit = diff.abs()[0, 0] > 1e-3 * fftnc(x, (-2, -1)).abs().pow(2).mean().sqrt()  # (D, H, W)
    for d in range(1, 4):
        assert not torch.equal(hit[d], hit[0])
    frac = hit.float().mean(dim=(-2, -1))
    assert ((frac - p).abs() < 0.2 * p).all(), frac


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_resolution_snr_noise_referenced_to_clean_rms(mode, theta):
    x = _x((1, 2, 8, 32, 32))
    k_clean = fftnc(x, DIMS[mode])
    window = apodization_window(tuple(k_clean.shape), DIMS[mode], theta * 2.5, x.device)
    k_out = fftnc(_apply("resolution_snr", x, theta, mode), DIMS[mode])
    sigma = float((k_out - k_clean * window).abs().pow(2).mean().div(2).sqrt())
    rms_clean = float(k_clean.abs().pow(2).mean().sqrt())
    expected = theta * rms_clean / 10 ** (20 / 20) / math.sqrt(2)
    assert abs(sigma - expected) < 0.02 * expected, (sigma, expected)
