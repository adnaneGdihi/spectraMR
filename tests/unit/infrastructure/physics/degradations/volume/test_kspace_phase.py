"""Native 3-D gibbs, eddy and iq_ghost: invariants and one closed-form check each."""

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
from spectramr.infrastructure.physics.degradations.volume import kspace_phase as kp
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc

M = AcquisitionMode.VOLUME_3D
GEOM = AcquisitionGeometry(mode=M, voxel_mm=None)
NAMES = ("gibbs", "eddy", "iq_ghost")


def _x(shape=(1, 2, 6, 16, 16), seed=3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _apply(name, x, theta, seed=1, **kw):
    return apply_degradation(name, x, theta, seed=seed, geometry=GEOM, **kw)


@pytest.mark.parametrize("name", NAMES)
def test_declared_native(name):
    assert DEGRADATION_REGISTRY[name].realisation[M] is Realisation.NATIVE


@pytest.mark.parametrize("name", NAMES)
def test_theta_zero_is_identity(name):
    x = _x()
    assert torch.equal(_apply(name, x, 0.0), x)


@pytest.mark.parametrize("name", ("gibbs", "iq_ghost"))
def test_monotone_finite_shape_deterministic(name):
    x = _x()
    outs = [_apply(name, x, t) for t in (0.0, 0.3, 0.6, 1.0)]
    for y in outs:
        assert y.shape == x.shape
        assert torch.isfinite(torch.view_as_real(y)).all()
    res = [float((y - x).abs().pow(2).mean().sqrt()) for y in outs]
    assert all(a < b for a, b in pairwise(res)), res
    assert torch.equal(_apply(name, x, 0.6), outs[2])


def test_eddy_finite_deterministic_and_coefficients_grow_with_theta():
    x = _x()
    y = _apply("eddy", x, 0.6)
    assert y.shape == x.shape
    assert torch.isfinite(torch.view_as_real(y)).all()
    assert torch.equal(_apply("eddy", x, 0.6), y)
    mags = [
        float(sum(c.abs().sum() for c in kp._eddy_coefficients(1, t * 5.0, x.device, 1)))
        for t in (0.3, 0.6, 1.0)
    ]
    assert all(a < b for a, b in pairwise(mags)), mags


@pytest.mark.parametrize("name", NAMES)
def test_batch_invariant(name):
    x = _x()
    alone = _apply(name, x, 0.6)
    pair = _apply(name, torch.cat([x, 10 * _x(seed=9)]), 0.6)
    torch.testing.assert_close(pair[:1], alone, rtol=1e-5, atol=1e-5)


def test_gibbs_keeps_exactly_the_ellipsoid():
    x = _x((1, 1, 8, 12, 16))
    y = _apply("gibbs", x, 0.5)
    kept = 1.0 - 0.5 * 0.915
    kz, ky, kx = (
        (2 * centred_freqs(n, x.device)).view(*[-1 if i == a else 1 for i in range(3)])
        for a, n in enumerate(x.shape[-3:])
    )
    inside = ((kz**2 + ky**2 + kx**2).sqrt() / 3**0.5) <= kept
    k_in, k_out = fftnc(x, dims=(-3, -2, -1)), fftnc(y, dims=(-3, -2, -1))
    torch.testing.assert_close(k_out, k_in * inside, rtol=1e-5, atol=1e-5)
    assert 0 < int(inside.sum()) < inside.numel()


def test_gibbs_corner_is_unit_radius():
    x = _x((1, 1, 4, 4, 4))
    assert torch.equal(_apply("gibbs", x, 0.0, truncation_fraction=1.0), x)
    y = kp.gibbs_volume(x, 0.5, 0, GEOM, truncation_fraction=0.999)
    assert not torch.equal(y, x)


def _patch_coeffs(monkeypatch, **vals):
    def fake(batch, strength, device, seed):
        z = torch.zeros(batch, device=device)
        return tuple(z + vals.get(k, 0.0) for k in ("a_y", "b_y", "a_z", "b_z"))

    monkeypatch.setattr(kp, "_eddy_coefficients", fake)


@pytest.mark.parametrize(("coef", "axis"), [("a_z", -3), ("a_y", -2)])
def test_eddy_linear_term_is_an_exact_circular_shift(monkeypatch, coef, axis):
    _patch_coeffs(monkeypatch, **{coef: 2.0})
    x = _x((1, 2, 8, 16, 16))
    y = _apply("eddy", x, 0.5)
    torch.testing.assert_close(y, torch.roll(x, shifts=-2, dims=axis), rtol=1e-4, atol=1e-4)


def test_eddy_leaves_readout_untouched_in_magnitude_per_line():
    x = _x((1, 1, 8, 16, 16))
    y = _apply("eddy", x, 1.0)
    kx, ky = fftnc(x, dims=(-3, -2, -1)), fftnc(y, dims=(-3, -2, -1))
    torch.testing.assert_close(ky.abs(), kx.abs(), rtol=1e-4, atol=1e-4)
    ratio = ky / kx
    torch.testing.assert_close(ratio, ratio[..., :1].expand_as(ratio), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(5, 6, 7), (4, 5, 6), (6, 6, 6)])
def test_mirror_is_an_involution(shape):
    x = _x((1, 1, *shape))
    assert torch.equal(kp.conjugate_mirror(kp.conjugate_mirror(x)), x)


def test_impulse_ghosts_through_the_centre():
    d, h, w = 6, 8, 9
    x = torch.zeros(1, 1, d, h, w, dtype=torch.complex64)
    a, b, c = 1, 2, -3
    x[0, 0, d // 2 + a, h // 2 + b, w // 2 + c] = 1.0
    y = _apply("iq_ghost", x, 1.0)
    ghost = y[0, 0, d // 2 - a, h // 2 - b, w // 2 - c]
    assert 0.3 < float(ghost.abs()) < 0.4
    assert float(y[0, 0, d // 2, h // 2, w // 2].abs()) > 0.15


def test_iq_spike_is_per_sample():
    x = _x()
    alone = _apply("iq_ghost", x, 0.6)
    pair = _apply("iq_ghost", torch.cat([x, 10 * _x(seed=9)]), 0.6)
    c = tuple(s // 2 for s in x.shape[-3:])
    torch.testing.assert_close(pair[:1][(..., *c)], alone[(..., *c)], rtol=1e-5, atol=1e-5)
