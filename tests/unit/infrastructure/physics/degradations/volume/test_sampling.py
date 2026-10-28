"""Native 3-D sampling operators: invariants and the closed-form sampling checks."""

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
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc

M = AcquisitionMode
VOL = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None)
MS = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None)
NAMES = ("cartesian_undersamp", "vd_cartesian", "vd_undersamp", "partial_fourier")
DIMS = (-3, -2, -1)


def _x(shape=(1, 2, 8, 32, 16), seed=3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _apply(name, x, theta, seed=1, geometry=VOL, **kw):
    return apply_degradation(name, x, theta, seed=seed, geometry=geometry, **kw)


def _mask(name, theta, shape=(1, 2, 8, 32, 16), seed=1, **kw) -> torch.Tensor:
    """The realised 3-D sampling mask: k-space of a constant-spectrum input after the operator."""
    ones_k = torch.ones(shape, dtype=torch.complex64)
    y = _apply(name, ifftnc(ones_k, DIMS), theta, seed=seed, **kw)
    k = fftnc(y, DIMS)
    assert torch.allclose(k.abs(), (k.abs() > 0.5).float(), atol=1e-4)  # a binary mask
    return k.abs() > 0.5


@pytest.mark.parametrize("name", NAMES)
def test_declared_native(name):
    assert DEGRADATION_REGISTRY[name].realisation[M.VOLUME_3D] is Realisation.NATIVE


@pytest.mark.parametrize("name", NAMES)
def test_theta_zero_is_identity(name):
    x = _x()
    assert torch.equal(_apply(name, x, 0.0), x)


@pytest.mark.parametrize("name", NAMES)
def test_monotone_finite_shape_deterministic(name):
    x = _x()
    outs = [_apply(name, x, t) for t in (0.0, 0.3, 0.6, 1.0)]
    for y in outs:
        assert y.shape == x.shape
        assert torch.isfinite(torch.view_as_real(y)).all()
    res = [float((y - x).abs().pow(2).mean().sqrt()) for y in outs]
    assert all(a < b for a, b in pairwise(res)), res
    assert torch.equal(_apply(name, x, 0.6), outs[2])


@pytest.mark.parametrize("name", NAMES)
def test_batch_invariant(name):
    x = _x()
    alone = _apply(name, x, 0.6)
    pair = _apply(name, torch.cat([x, 10 * _x(seed=9)]), 0.6)
    torch.testing.assert_close(pair[:1], alone, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_real_input_handled_as_complex(name):
    x = _x().real
    y = _apply(name, x, 0.6)
    assert y.is_complex() and y.shape == x.shape


@pytest.mark.parametrize("name", NAMES)
def test_mask_shared_by_coils(name):
    m = _mask(name, 0.6)
    assert torch.equal(m[:, 0], m[:, 1])


@pytest.mark.parametrize("name", ("cartesian_undersamp", "vd_cartesian"))
@pytest.mark.parametrize("theta", [0.3, 0.6, 1.0])
def test_cartesian_realised_r_and_acs(name, theta):
    m = _mask(name, theta)[0, 0]
    n_kz, n_ky, _ = m.shape
    # Whole readouts: every kx of an acquired (kz, ky) line is kept.
    lines = m[..., 0]
    assert torch.equal(m, lines[..., None].expand_as(m))
    R = 1.0 + 7.0 * theta
    acs_kz, acs_ky = max(1, int(n_kz * 0.08)), max(1, int(n_ky * 0.08))
    assert int(lines.sum()) == max(acs_kz * acs_ky, round(n_kz * n_ky / R))
    z0, y0 = n_kz // 2 - acs_kz // 2, n_ky // 2 - acs_ky // 2
    assert lines[z0 : z0 + acs_kz, y0 : y0 + acs_ky].all()


def test_cartesian_acs_scales_with_both_encoded_axes():
    m = _mask("cartesian_undersamp", 1.0, shape=(1, 1, 32, 64, 8))[0, 0, ..., 0]
    acs_kz, acs_ky = int(32 * 0.08), int(64 * 0.08)  # 2 x 5
    assert m[16 - 1 : 16 + 1, 32 - 2 : 32 + 3].all()
    assert int(m.sum()) == round(32 * 64 / 8) and acs_kz * acs_ky == 10


def test_vd_cartesian_favours_the_centre():
    m = torch.stack([_mask("vd_cartesian", 1.0, seed=s)[0, 0, ..., 0] for s in range(8)])
    inner = m[:, 2:6, 8:24].float().mean()
    outer = torch.cat([m[:, :, :8], m[:, :, 24:]], dim=2).float().mean()
    assert inner > 1.5 * outer


@pytest.mark.parametrize("theta", [0.3, 1.0])
def test_vd_undersamp_realised_r_and_dc_block(theta):
    shape = (1, 1, 8, 32, 32)
    masks = [_mask("vd_undersamp", theta, shape=shape, seed=s)[0, 0] for s in range(6)]
    R = 1.0 + 7.0 * theta
    realised = sum(m.numel() for m in masks) / sum(int(m.sum()) for m in masks)
    assert abs(realised - R) / R < 0.03, (realised, R)
    for m in masks:
        assert m[4 - 2 : 4 + 3, 16 - 2 : 16 + 3, 16 - 2 : 16 + 3].all()


def test_vd_undersamp_is_a_point_mask_not_lines():
    m = _mask("vd_undersamp", 1.0)[0, 0]
    partial = m.any(-1) & ~m.all(-1)  # (kz, ky) lines with some but not all kx kept
    assert partial.any()


@pytest.mark.parametrize(
    ("axis", "kept"),
    [
        ("ky", lambda n_kz, n_ky, n_kx, f: (n_kz, round(n_ky * f), n_kx)),
        ("kz", lambda n_kz, n_ky, n_kx, f: (round(n_kz * f), n_ky, n_kx)),
        ("ky+kz", lambda n_kz, n_ky, n_kx, f: (round(n_kz * f), round(n_ky * f), n_kx)),
        ("kx", lambda n_kz, n_ky, n_kx, f: (n_kz, n_ky, n_kx - round(n_kx * (1 - f)))),
    ],
)
@pytest.mark.parametrize("theta", [0.5, 1.0])
def test_partial_fourier_kept_fraction_per_axis(axis, kept, theta):
    shape = (1, 2, 8, 32, 16)
    m = _mask("partial_fourier", theta, shape=shape, axis=axis)[0, 0]
    frac = 1.0 - 0.375 * theta
    kz, ky, kx = kept(*shape[2:], frac)
    expected = torch.zeros(shape[2:], dtype=torch.bool)
    expected[:kz, :ky, :kx] = True  # the low-index side is acquired
    assert torch.equal(m, expected)


@pytest.mark.parametrize("axis", ["kz", "ky+kz"])
def test_partial_fourier_volume_axes_rejected_in_multislice(axis):
    x = _x((1, 1, 4, 16, 16))
    with pytest.raises(ValueError, match="'ky' or 'kx'"):
        _apply("partial_fourier", x, 0.5, geometry=MS, axis=axis)


def test_partial_fourier_unknown_axis_names_the_valid_set():
    with pytest.raises(ValueError, match="ky\\+kz"):
        _apply("partial_fourier", _x(), 0.5, axis="kq")


@pytest.mark.parametrize(
    ("name", "kw"),
    [
        ("cartesian_undersamp", {"center_fraction": 0.0}),
        ("vd_cartesian", {"center_fraction": 1.5}),
        ("vd_undersamp", {"R_max": 0.5}),
        ("partial_fourier", {"min_fraction": 0.4}),
    ],
)
def test_bad_knobs_raise(name, kw):
    with pytest.raises(ValueError):
        _apply(name, _x(), 0.5, **kw)


@pytest.mark.parametrize("name", NAMES)
def test_rejects_non_volume_input(name):
    with pytest.raises(ValueError, match="D, H, W"):
        DEGRADATION_REGISTRY[name].implementations[M.VOLUME_3D](_x()[:, :, 0], 0.5, 1, VOL)


@pytest.mark.parametrize("name", ("cartesian_undersamp", "vd_cartesian"))
def test_acs_block_is_center_fraction_of_each_encoded_axis(name):
    # ACS = (int(32 * 0.5), int(64 * 0.5)) = 16 x 32 = 512 lines > 2048 / 8, so it is all acquired.
    m = _mask(name, 1.0, shape=(1, 1, 32, 64, 4), center_fraction=0.5)[0, 0, ..., 0]
    expected = torch.zeros(32, 64, dtype=torch.bool)
    expected[8:24, 16:48] = True
    assert torch.equal(m, expected)
