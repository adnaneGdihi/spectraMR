"""Closed-form physics of correlated coil noise, EMI bursts and the anti-alias roll-off."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.receive_common import (
    ENCODED_DIMS,
    acquisition_ranks,
)
from spectramr.infrastructure.physics.degradations.catalogue.receive_noise import (
    SYNTHETIC_COIL_RHO,
    butterworth_response,
    coil_noise_factor,
)
from spectramr.infrastructure.physics.degradations.geometry import LineOrder, SliceOrder
from spectramr.infrastructure.physics.fft_ops import fftnc
from spectramr.infrastructure.physics.scanner_presets import M4RAW_03T

M = AcquisitionMode
GEOMS = {
    "slice": None,
    "ms_interleaved": AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0)),
    "ms_sequential": AcquisitionGeometry(
        mode=M.MULTISLICE_2D, voxel_mm=(5.0, 1.0, 1.0), slice_order=SliceOrder.SEQUENTIAL
    ),
    "vol_linear": AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0)),
    "vol_centric": AcquisitionGeometry(
        mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0), line_order=LineOrder.CENTRIC
    ),
}


def _shape(geom, c=3, d=6, h=16, w=16):
    return (1, c, h, w) if geom is None else (1, c, d, h, w)


def _dims(geom):
    return ENCODED_DIMS[M.SLICE_2D if geom is None else geom.mode]


def _empirical_cov(noise: torch.Tensor) -> torch.Tensor:
    flat = noise.movedim(1, 0).reshape(noise.shape[1], -1).to(torch.complex128)
    return flat @ flat.mH / flat.shape[1]


@pytest.mark.parametrize("geom", GEOMS.values(), ids=GEOMS.keys())
def test_noise_covariance_equals_sigma(geom) -> None:
    # Constant unit image: whole-subject k-space RMS is 1, so the noise RMS is theta * 0.3.
    a = torch.randn(3, 3, dtype=torch.complex128, generator=torch.Generator().manual_seed(4))
    cov = a @ a.mH + torch.eye(3, dtype=torch.complex128)
    x = torch.ones((4, *_shape(geom, d=8, h=64, w=64)[1:]), dtype=torch.complex64)
    out = apply_degradation(
        "correlated_coil_noise", x, 1.0, seed=2, geometry=geom, coil_covariance=cov
    )
    expected = 0.3**2 * cov / cov.diagonal().real.mean()
    got = _empirical_cov(out - x)
    assert float((got - expected).abs().max()) < 0.04 * float(expected.abs().max())
    assert float(got[0, 1].imag.abs()) > 0.01  # the coupling is complex, not decorative


def test_default_covariance_is_the_declared_synthetic_one() -> None:
    assert SYNTHETIC_COIL_RHO == 0.3
    n = 4
    x = torch.ones(4, n, 8, 64, 64, dtype=torch.complex64)
    geom = GEOMS["vol_linear"]
    out = apply_degradation("correlated_coil_noise", x, 1.0, seed=1, geometry=geom)
    idx = torch.arange(n, dtype=torch.float64)
    expected = 0.3**2 * 0.3 ** (idx[:, None] - idx[None, :]).abs()
    got = _empirical_cov(out - x)
    assert float((got - expected).abs().max()) < 0.04 * float(expected.abs().max())


def test_profile_covariance_is_used_and_equals_the_explicit_kwarg() -> None:
    cov = ((1.0, 0.6, 0.0), (0.6, 2.0, 0.1), (0.0, 0.1, 1.5))
    profile = replace(M4RAW_03T, n_coils=3, coil_covariance=cov)
    x = torch.ones(2, 3, 64, 64, dtype=torch.complex64)
    via_profile = apply_degradation("correlated_coil_noise", x, 0.8, seed=5, profile=profile)
    via_kwarg = apply_degradation(
        "correlated_coil_noise", x, 0.8, seed=5, coil_covariance=torch.tensor(cov)
    )
    torch.testing.assert_close(via_profile, via_kwarg, rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="set by scanner profile"):
        apply_degradation(
            "correlated_coil_noise", x, 0.8, profile=profile, coil_covariance=torch.tensor(cov)
        )


@pytest.mark.parametrize(
    ("cov", "match"),
    [
        (torch.eye(2), "3 x 3"),
        (torch.tensor([[1.0, 0.5, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]), "Hermitian"),
        (torch.tensor([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "positive definite"),
    ],
)
def test_bad_covariance_raises(cov, match) -> None:
    with pytest.raises(ValueError, match=match):
        coil_noise_factor(cov, 3)


def test_noise_factor_reproduces_the_normalised_covariance() -> None:
    cov = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    chol = coil_noise_factor(cov, 2)
    torch.testing.assert_close(chol @ chol.mH, (cov / 3.0).to(torch.complex128))


def _line_changes(geom, x, out):
    dims = _dims(geom)
    diff = (fftnc(out, dims) - fftnc(x, dims)).abs()
    return diff.amax(dim=(1, -1))[0]  # [D, H]


@pytest.mark.parametrize("geom", GEOMS.values(), ids=GEOMS.keys())
def test_emi_burst_leaves_lines_outside_the_block_unchanged(geom) -> None:
    g = torch.Generator().manual_seed(8)
    shape = _shape(geom, c=2, d=6, h=24, w=32)
    x = torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))
    theta = 0.6
    out = apply_degradation("emi_burst", x, theta, seed=4, geometry=geom)
    x5, out5 = (t.unsqueeze(2) if geom is None else t for t in (x, out))
    changed = _line_changes(geom, x5, out5) > 1e-4
    d, h = changed.shape
    n_burst = math.ceil(theta * 0.2 * d * h)
    assert int(changed.sum()) == n_burst
    # Contiguous in ACQUISITION order, whatever the slice or line order.
    mode_geom = geom or AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None)
    rank, _ = acquisition_ranks(mode_geom, d, h, torch.device("cpu"))
    hit = rank[changed]
    assert int(hit.max() - hit.min()) + 1 == n_burst


def test_emi_burst_noise_is_white_across_kx_with_the_declared_rms() -> None:
    x = torch.ones(1, 2, 8, 64, 64, dtype=torch.complex64)  # subject RMS of k-space = 1
    geom = GEOMS["vol_linear"]
    out = apply_degradation("emi_burst", x, 1.0, seed=3, geometry=geom)
    noise = fftnc(out, (-3, -2, -1)) - fftnc(x, (-3, -2, -1))
    lines = noise.abs().amax(dim=(1, -1))[0] > 1e-4
    burst = noise[:, :, lines, :]  # [1, C, n_lines, W]
    assert float(burst.abs().pow(2).mean().sqrt()) == pytest.approx(0.5, rel=0.06)
    spectrum = burst.abs().pow(2).mean(dim=(0, 1, 2))  # power per kx
    assert float(spectrum.std() / spectrum.mean()) < 0.5  # no kx structure beyond sampling noise


def test_emi_burst_run_is_nested_in_theta() -> None:
    x = torch.randn(1, 2, 6, 24, 16, dtype=torch.complex64)
    geom = GEOMS["vol_linear"]
    runs = []
    for theta in (0.3, 0.6, 1.0):
        out = apply_degradation("emi_burst", x, theta, seed=9, geometry=geom)
        runs.append(_line_changes(geom, x, out) > 1e-4)
    assert bool((runs[0] <= runs[1]).all()) and bool((runs[1] <= runs[2]).all())


def test_butterworth_matches_the_closed_form() -> None:
    nu = torch.linspace(-1, 1, 41)
    got = butterworth_response(nu, 0.7, order=4)
    ref = 1.0 / np.sqrt(1.0 + (np.abs(nu.numpy()) / 0.7) ** 8)
    np.testing.assert_allclose(got.numpy(), ref, rtol=1e-6)
    assert float(butterworth_response(torch.tensor(0.7), 0.7)) == pytest.approx(2**-0.5)


@pytest.mark.parametrize("geom", GEOMS.values(), ids=GEOMS.keys())
@pytest.mark.parametrize("theta", [0.4, 0.8, 1.0])
def test_antialias_gain_along_the_readout_is_the_butterworth_response(geom, theta) -> None:
    w = 20
    g = torch.Generator().manual_seed(2)
    shape = _shape(geom, c=2, d=3, h=8, w=w)
    x = torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))
    out = apply_degradation("antialias_rolloff", x, theta, geometry=geom)
    ratio = out / x
    cutoff = 0.6 / theta
    nu = np.abs(2.0 * (np.arange(w) - w // 2) / w)
    expected = 1.0 / np.sqrt(1.0 + (nu / cutoff) ** 8)
    got = ratio.real.reshape(-1, w)
    np.testing.assert_allclose(got.numpy(), np.broadcast_to(expected, got.shape), atol=2e-6)
    assert float(ratio.imag.abs().max()) < 2e-6  # magnitude response only: no phase change


def test_antialias_cutoff_at_full_severity_is_minus_3_db_at_0p6_nyquist() -> None:
    w = 20  # column w = 16 sits at nu = 2 * 6 / 20 = 0.6
    x = torch.ones(1, 1, 4, w, dtype=torch.complex64)
    out = apply_degradation("antialias_rolloff", x, 1.0)
    assert float(out[0, 0, 0, 16].real) == pytest.approx(2**-0.5, rel=1e-5)
    assert float(out[0, 0, 0, 10].real) == pytest.approx(1.0, rel=1e-6)


def test_antialias_filter_colours_the_noise_added_before_it() -> None:
    w, sigma, theta = 40, 0.5, 1.0
    x = torch.ones(4, 2, 8, 64, w, dtype=torch.complex64)  # subject RMS of k-space = 1
    geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None)
    out = apply_degradation("antialias_rolloff", x, theta, seed=1, geometry=geom, noise_sigma=sigma)
    gain = np.asarray(
        1.0 / np.sqrt(1.0 + (np.abs(2.0 * (np.arange(w) - w // 2) / w) / (0.6 / theta)) ** 8),
        dtype=np.float32,
    )
    noise = out - x * torch.from_numpy(gain)
    var = noise.abs().pow(2).movedim(-1, 0).reshape(w, -1).mean(dim=1).numpy()
    keep = gain > 0.3
    np.testing.assert_allclose(var[keep], sigma**2 * gain[keep] ** 2, rtol=0.1)


def test_antialias_noise_sigma_adds_unfiltered_noise_at_theta_zero() -> None:
    x = torch.ones(2, 2, 4, 32, 32, dtype=torch.complex64)
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None)
    assert torch.equal(apply_degradation("antialias_rolloff", x, 0.0, geometry=geom), x)
    out = apply_degradation("antialias_rolloff", x, 0.0, seed=2, geometry=geom, noise_sigma=0.3)
    assert float((out - x).abs().pow(2).mean().sqrt()) == pytest.approx(0.3, rel=0.05)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("correlated_coil_noise", {"sigma_max": 0.0}),
        ("emi_burst", {"sigma_max": -1.0}),
        ("emi_burst", {"frac_max": 1.5}),
        ("antialias_rolloff", {"noise_sigma": -0.1}),
        ("antialias_rolloff", {"noise_sigma": float("nan")}),
    ],
)
def test_bad_knobs_raise_even_at_theta_zero(name, kwargs) -> None:
    x = torch.randn(1, 3, 8, 8, dtype=torch.complex64)
    with pytest.raises(ValueError):
        apply_degradation(name, x, 0.0, **kwargs)


def test_antialias_response_tends_continuously_to_the_identity() -> None:
    g = torch.Generator().manual_seed(5)
    x = torch.complex(
        torch.randn(1, 1, 64, 64, generator=g), torch.randn(1, 1, 64, 64, generator=g)
    )

    def effect(theta: float) -> float:
        return float((apply_degradation("antialias_rolloff", x, theta) - x).abs().pow(2).sum())

    assert effect(1e-4) < 1e-3 * effect(1.0)
    curve = [effect(t) for t in (0.25, 0.5, 0.75, 1.0)]
    assert all(a < b for a, b in itertools.pairwise(curve))
