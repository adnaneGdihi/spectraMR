"""recon_apodization and zero_fill_interp: exact k-space multipliers and truncations."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.recon_kspace import (
    WINDOWS,
    apodization_window,
    truncation_mask,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc

from .invariants import GEOMETRIES, assert_axis_invariants

M = AcquisitionMode
VOLUME = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(5.0, 1.0, 1.0))


def _cplx(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=g), torch.randn(*shape, generator=g))


@pytest.mark.parametrize("name", ["recon_apodization", "zero_fill_interp"])
def test_modes(name):
    assert DEGRADATION_REGISTRY[name].realisation == {
        M.SLICE_2D: Realisation.NATIVE,
        M.MULTISLICE_2D: Realisation.PER_SLICE_EXACT,
        M.VOLUME_3D: Realisation.NATIVE,
    }


@pytest.mark.parametrize("name", ["recon_apodization", "zero_fill_interp"])
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(name, geom):
    assert_axis_invariants(name, GEOMETRIES[geom])


def test_windows_have_their_published_values():
    rho = torch.tensor([0.0, 0.5, 1.0])
    torch.testing.assert_close(
        WINDOWS["hamming"](rho), torch.tensor([1.0, 0.54, 0.08]), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        WINDOWS["hann"](rho), torch.tensor([1.0, 0.5, 0.0]), rtol=0, atol=1e-6
    )
    fermi = WINDOWS["fermi"](torch.tensor([0.0, 0.75, 1.0]))
    assert fermi[0] > 0.999 and float(fermi[1]) == pytest.approx(0.5) and fermi[2] < 0.1


@pytest.mark.parametrize("window", sorted(WINDOWS))
@pytest.mark.parametrize("theta", [0.3, 1.0])
def test_kspace_ratio_is_exactly_the_mixed_window_2d(window, theta):
    x = _cplx(1, 2, 16, 24)
    out = apply_degradation("recon_apodization", x, theta, window=window)
    w = apodization_window(tuple(x.shape), (-2, -1), window, x.device, torch.float32)
    ratio = fftnc(out, (-2, -1)) / fftnc(x, (-2, -1))
    torch.testing.assert_close(
        ratio.real, (1 - theta * (1 - w)).expand_as(ratio.real), rtol=0, atol=1e-4
    )
    assert float(ratio.imag.abs().max()) < 1e-4


@pytest.mark.parametrize("window", sorted(WINDOWS))
def test_3d_window_is_the_product_of_three_axis_windows(window):
    shape = (1, 1, 6, 16, 24)
    out = apply_degradation("recon_apodization", _cplx(*shape), 1.0, geometry=VOLUME, window=window)
    w = apodization_window(shape, (-3, -2, -1), window, torch.device("cpu"), torch.float32)
    axes = [
        WINDOWS[window](2 * centred_freqs(n).abs()).view([-1 if i == j else 1 for j in range(5)])
        for i, n in zip((2, 3, 4), shape[2:], strict=True)
    ]
    torch.testing.assert_close(w, axes[0] * axes[1] * axes[2], rtol=0, atol=1e-6)
    assert out.shape == shape


def test_apodization_is_a_low_pass():
    x = _cplx(1, 1, 32, 32)
    k_in = fftnc(x, (-2, -1)).abs()
    k_out = fftnc(apply_degradation("recon_apodization", x, 1.0), (-2, -1)).abs()
    assert float(k_out[..., 0, :].sum()) < 0.2 * float(k_in[..., 0, :].sum())  # the highest ky row
    torch.testing.assert_close(
        k_out[..., 16, 16], k_in[..., 16, 16], rtol=0, atol=1e-4
    )  # DC untouched


def test_unknown_window_raises_in_every_mode():
    x = _cplx(1, 1, 8, 8)
    with pytest.raises(ValueError, match="unknown window"):
        apply_degradation("recon_apodization", x, 0.5, window="kaiser")
    with pytest.raises(ValueError, match="unknown window"):
        apply_degradation(
            "recon_apodization", x.unsqueeze(2), 0.5, geometry=VOLUME, window="kaiser"
        )


def test_default_apodization_is_not_collinear_with_gibbs_or_resolution_snr():
    from tests.unit.infrastructure.physics.test_digital_twin_extensions import _brain

    x = _brain()

    def resid(name):
        r = (apply_degradation(name, x, 1.0, seed=0).abs() - x.abs()).flatten()
        return r / r.norm()

    for new in ("recon_apodization", "zero_fill_interp"):
        for old in ("gibbs", "resolution_snr"):
            assert abs(float(resid(new) @ resid(old))) < 0.9, (new, old)


@pytest.mark.parametrize("n", [16, 15])
def test_truncation_mask_keeps_the_central_samples_about_dc(n):
    keep = torch.arange(n)[truncation_mask((n,), (0,), 0.5, torch.device("cpu"))]
    assert len(keep) == max(1, round(n * 0.5))
    assert (n // 2) in keep.tolist()
    assert truncation_mask((n,), (0,), 1.0, torch.device("cpu")).all()


@pytest.mark.parametrize(
    ("shape", "geometry", "dims"),
    [((1, 2, 20, 17), None, (-2, -1)), ((1, 1, 6, 20, 17), VOLUME, (-3, -2, -1))],
)
def test_outside_the_kept_window_kspace_is_exactly_zero_and_inside_unchanged(shape, geometry, dims):
    x, theta = _cplx(*shape), 0.6
    out = apply_degradation("zero_fill_interp", x, theta, geometry=geometry)
    keep = truncation_mask(shape, dims, 1.0 - 0.5 * theta, x.device)
    k_in, k_out = fftnc(x, dims), fftnc(out, dims)
    assert float(k_out[~keep.expand_as(k_out)].abs().max()) < 1e-5
    torch.testing.assert_close(
        k_out[keep.expand_as(k_out)], k_in[keep.expand_as(k_out)], rtol=0, atol=1e-5
    )
    assert 0 < int(keep.sum()) < keep.numel()


def test_zero_filling_a_delta_gives_the_dirichlet_sinc_kernel():
    """|out(t)| = |sin(pi m t / n) / (n sin(pi t / n))| for m central samples of n (2-D separable)."""
    n, m = 32, 16
    x = torch.zeros(1, 1, n, n, dtype=torch.complex64)
    x[..., n // 2, n // 2] = 1.0
    out = apply_degradation("zero_fill_interp", x, 1.0, max_fraction=0.5)
    t = torch.arange(n, dtype=torch.float32) - n // 2
    d = torch.where(
        t == 0,
        torch.tensor(m / n),
        (torch.sin(math.pi * m * t / n) / (n * torch.sin(math.pi * t / n))).abs(),
    )
    torch.testing.assert_close(
        out.abs()[0, 0], d.abs()[:, None] * d.abs()[None, :], rtol=0, atol=2e-5
    )
    assert (
        float(d[1]) > 0 and float((out.real < -1e-3).float().sum()) > 0
    )  # truncation ringing: negative lobes


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_zero_fill_max_fraction_is_validated(bad):
    with pytest.raises(ValueError, match="max_fraction"):
        apply_degradation("zero_fill_interp", _cplx(1, 1, 8, 8), 0.5, max_fraction=bad)
