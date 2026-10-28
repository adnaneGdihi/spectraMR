"""cs_recon_artifacts: a fixed-iteration ISTA with an orthonormal Haar prior."""

from __future__ import annotations

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
from spectramr.infrastructure.physics.degradations.catalogue.recon_cs import haar_transform, ista
from spectramr.infrastructure.physics.fft_ops import fftnc
from spectramr.infrastructure.physics.wavelet_ops import OrthogonalWavelet2D

from .invariants import GEOMETRIES, assert_axis_invariants, phantom

M = AcquisitionMode
NAME = "cs_recon_artifacts"
VOLUME = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0))


def _cplx(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g))


def test_modes_and_gate():
    spec = DEGRADATION_REGISTRY[NAME]
    assert spec.realisation == {
        M.SLICE_2D: Realisation.NATIVE,
        M.MULTISLICE_2D: Realisation.PER_SLICE_EXACT,
        M.VOLUME_3D: Realisation.NATIVE,
    }
    assert {str(g) for g in spec.active_when} == {"accelerated"}


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


@pytest.mark.parametrize(
    ("dims", "shape"), [((-2, -1), (1, 2, 1, 16, 8)), ((-3, -2, -1), (1, 1, 8, 16, 8))]
)
def test_haar_is_orthonormal_and_invertible(dims, shape):
    x = _cplx(*shape)
    c = haar_transform(x, dims, 2)
    assert float(c.abs().pow(2).sum()) == pytest.approx(float(x.abs().pow(2).sum()), rel=1e-5)
    torch.testing.assert_close(haar_transform(c, dims, 2, inverse=True), x, rtol=0, atol=1e-5)


def test_2d_haar_is_the_existing_wavelet_operator():
    x = _cplx(1, 1, 1, 16, 16)
    ours = haar_transform(x, (-2, -1), 2)
    ref = OrthogonalWavelet2D("haar", 2).forward(x)
    torch.testing.assert_close(ours, ref, rtol=0, atol=1e-5)


def test_one_ista_step_shrinks_a_detail_coefficient_by_tau_and_leaves_the_coarse_band():
    """4x4 image, one Haar level: a constant c plus a +-a pattern on the 2x2 block at the origin
    ([[a, a], [-a, -a]]) has the single detail coefficient 2a there (LL = 2c per block). Soft-thresholding
    by tau < 2a leaves amplitude a - tau / 2; the coarse band (the constant) is not thresholded."""
    c, a, tau = 3.0, 1.0, 0.5
    x = torch.full((1, 1, 1, 4, 4), c, dtype=torch.complex64)
    x[..., 0, :2], x[..., 1, :2] = c + a, c - a
    dims = (-2, -1)
    out = ista(fftnc(x, dims), torch.ones(1, 1, 1, 4, 1, dtype=torch.bool), tau, 1, dims, 1)
    want = torch.full_like(x, c)
    want[..., 0, :2], want[..., 1, :2] = c + a - tau / 2, c - a + tau / 2
    torch.testing.assert_close(out, want, rtol=0, atol=1e-5)


def test_full_sampling_and_no_regularisation_is_the_identity():
    x = _cplx(1, 2, 1, 16, 16)
    dims = (-2, -1)
    out = ista(fftnc(x, dims), torch.ones(1, 1, 1, 16, 1, dtype=torch.bool), 0.0, 20, dims, 2)
    torch.testing.assert_close(out, x, rtol=0, atol=1e-5)


def test_data_consistency_keeps_the_acquired_lines_and_leaves_the_rest_empty_without_a_prior():
    x = _cplx(1, 1, 1, 32, 16)
    out = apply_degradation(NAME, x[:, :, 0], 1.0, seed=3, lambda_max=0.0, r_max=4.0)
    k_in, k_out = fftnc(x[:, :, 0], (-2, -1)), fftnc(out, (-2, -1))
    kept = k_out.abs().sum((0, 1, 3)) > 1e-3 * k_out.abs().sum((0, 1, 3)).max()
    assert int(kept.sum()) == 8  # round(32 / 4) lines, the ACS included
    torch.testing.assert_close(k_out[..., kept, :], k_in[..., kept, :], rtol=0, atol=1e-4)


def test_volume_mask_is_over_kz_ky():
    x = _cplx(1, 1, 8, 16, 8)
    out = apply_degradation(NAME, x, 1.0, seed=3, geometry=VOLUME, lambda_max=0.0, r_max=4.0)
    line = fftnc(out, (-3, -2, -1)).abs().sum((0, 1, 4))  # [kz, ky]
    k = line > 1e-3 * line.max()
    assert int(k.sum()) == 8 * 16 // 4


def test_regularisation_removes_wavelet_detail():
    x = phantom(None)
    dims = (-2, -1)

    def detail_l1(lam):
        out = apply_degradation(NAME, x, 1.0, seed=1, lambda_max=lam, r_max=1.0)
        c = haar_transform(out.unsqueeze(2), dims, 2).abs()
        c[..., :8, :8] = 0.0
        return float(c.sum())

    assert detail_l1(0.05) < 0.8 * detail_l1(0.0)


def test_theta_maps_to_r_and_lambda_as_declared():
    params = {p.name: p for p in DEGRADATION_REGISTRY[NAME].severity.params}
    assert (params["acceleration"].at_theta_min, params["acceleration"].at_theta_max) == (1.0, 6.0)


def test_odd_matrix_is_padded_and_cropped():
    x = _cplx(1, 2, 33, 31)
    out = apply_degradation(NAME, x, 0.7, seed=2)
    assert out.shape == x.shape and torch.isfinite(torch.view_as_real(out)).all()


def test_iterations_are_a_kwarg_and_more_of_them_change_the_result():
    x = phantom(None)
    a = apply_degradation(NAME, x, 1.0, seed=1, iterations=2)
    b = apply_degradation(NAME, x, 1.0, seed=1, iterations=20)
    assert not torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"iterations": 2.5},
        {"r_max": 0.5},
        {"lambda_max": -1.0},
        {"center_fraction": 0.0},
        {"lambda_ramp": 0.0},
        {"lambda_ramp": -0.05},
    ],
)
def test_bad_knobs_raise(kwargs):
    with pytest.raises(ValueError, match="cs_recon_artifacts"):
        apply_degradation(NAME, phantom(None), 0.5, **kwargs)


def test_lambda_is_fixed_above_a_short_ramp_and_continuous_at_zero():
    """lambda = lambda_max min(1, theta / lambda_ramp): at R = 1 (r_max = 1) theta above the ramp is
    identical, and below it the output moves continuously towards the input."""
    x = phantom(None)
    kw = {"r_max": 1.0, "seed": 1}
    a = apply_degradation(NAME, x, 0.05, **kw)
    torch.testing.assert_close(apply_degradation(NAME, x, 0.5, **kw), a, rtol=0, atol=1e-6)
    d = [float((apply_degradation(NAME, x, t, **kw) - x).norm()) for t in (1e-4, 0.01, 0.025, 0.05)]
    assert d[0] < 0.01 * d[-1] and all(p < q for p, q in pairwise(d))


def _square(n):
    x = torch.zeros(1, 1, n, n, dtype=torch.complex64)
    x[..., n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 1.0 + 0.3j
    return x


def _disc(n):
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
    disc = ((xx - 0.1) ** 2 + (1.3 * yy) ** 2 < 0.55).float()
    tex = (0.6 + 0.4 * torch.cos(6 * xx) * torch.sin(5 * yy)).abs()
    return (disc * tex).expand(1, 1, n, n).to(torch.complex64)


@pytest.mark.parametrize(
    ("name", "x", "grid"),
    [
        ("square", _square(128), (0.0, 0.25, 0.5, 0.75, 1.0)),
        ("disc", _disc(48), tuple(0.125 * k for k in range(1, 9))),
    ],
)
@pytest.mark.parametrize("seed", range(8))
def test_severity_is_monotone_in_theta_for_every_seed(name, x, grid, seed):
    """Nested masks plus a lambda that does not change with theta: only R grows (checked at 8 seeds x 2 phantoms)."""
    c = [float((apply_degradation(NAME, x, t, seed=seed).abs() - x.abs()).norm()) for t in grid]
    assert all(b > a for a, b in pairwise(c)), (name, seed, c)
