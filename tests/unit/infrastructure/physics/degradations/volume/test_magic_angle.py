"""Native multislice and 3-D magic-angle T2 modulation."""

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
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import mm_grid

M = AcquisitionMode
MODES = (M.MULTISLICE_2D, M.VOLUME_3D)


def _geom(mode, voxel_mm=(2.0, 1.0, 1.0)) -> AcquisitionGeometry:
    return AcquisitionGeometry(mode=mode, voxel_mm=voxel_mm)


def _ball(shape=(1, 2, 8, 24, 24), seed=3, background=0.01) -> torch.Tensor:
    # A bright ellipsoid on a background below the 5% support threshold, random phase.
    g = torch.Generator().manual_seed(seed)
    z, y, x = mm_grid(tuple(shape[-3:]), None)
    inside = (z**2 + y**2 + x**2) < 0.5
    mag = torch.where(inside, 1.0, background).expand(shape)
    phase = 2.0 * torch.pi * torch.rand(shape, generator=g)
    return torch.polar(mag * (1.0 + torch.rand(shape, generator=g)), phase)


def _apply(x, theta, mode, seed=1, geometry=None, **kw):
    return apply_degradation(
        "magic_angle", x, theta, seed=seed, geometry=geometry or _geom(mode), **kw
    )


@pytest.mark.parametrize("mode", MODES)
def test_declared_native(mode):
    assert DEGRADATION_REGISTRY["magic_angle"].realisation[mode] is Realisation.NATIVE


@pytest.mark.parametrize("mode", MODES)
def test_theta_zero_is_identity(mode):
    x = _ball()
    assert torch.equal(_apply(x, 0.0, mode), x)


@pytest.mark.parametrize("mode", MODES)
def test_monotone_finite_shape_deterministic(mode):
    x = _ball()
    outs = [_apply(x, t, mode) for t in (0.0, 0.3, 0.6, 1.0)]
    for y in outs:
        assert y.shape == x.shape
        assert torch.isfinite(torch.view_as_real(y)).all()
    res = [float((y - x).abs().pow(2).mean().sqrt()) for y in outs]
    assert all(a < b for a, b in pairwise(res)), res
    assert torch.equal(_apply(x, 0.6, mode), outs[2])


@pytest.mark.parametrize("mode", MODES)
def test_batch_invariant(mode):
    x = _ball()
    alone = _apply(x, 0.6, mode)
    pair = _apply(torch.cat([x, 10 * _ball(seed=9)]), 0.6, mode)
    torch.testing.assert_close(pair[:1], alone, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", MODES)
def test_magnitude_only(mode):
    x = _ball()
    y = _apply(x, 1.0, mode)
    assert float((torch.angle(y * x.conj())).abs().max()) < 1e-6


@pytest.mark.parametrize("mode", MODES)
def test_boosts_only_inside_the_3d_support(mode):
    x = _ball()
    y = _apply(x, 1.0, mode)
    support = x.abs().amax(1, keepdim=True) > 0.05 * x.abs().amax()
    ratio = y.abs() / x.abs()
    assert torch.equal(y[~support.expand_as(x)], x[~support.expand_as(x)])
    assert float(ratio.min()) >= 1.0 - 1e-6
    boosted = (ratio > 1.0 + 1e-4) & support
    assert float(boosted.float().sum()) > 0
    # Patches are 3-D: the boost reaches more than one slice.
    assert int(boosted.any(-1).any(-1).sum()) > 1


@pytest.mark.parametrize("mode", MODES)
def test_peak_boost_matches_the_declared_t2_range(mode):
    # At the magic angle T2_eff = t2_magic: boost = exp(-TE/30) / exp(-TE/10) at theta = 1.
    x = _ball()
    ratio = (_apply(x, 1.0, mode).abs() / x.abs()).max()
    bound = torch.exp(torch.tensor(-20.0 / 30.0)) / torch.exp(torch.tensor(-20.0 / 10.0))
    assert float(ratio) <= float(bound) + 1e-4
    assert float(ratio) > 1.0 + 0.5 * (float(bound) - 1.0)


@pytest.mark.parametrize("mode", MODES)
def test_non_msk_tissue_is_a_no_op(mode):
    x = _ball()
    assert torch.equal(_apply(x, 1.0, mode, tissue_class="brain"), x)


@pytest.mark.parametrize("mode", MODES)
def test_foreground_mask_restricts_the_patches(mode):
    x = _ball()
    mask = torch.zeros(1, 1, *x.shape[-3:])
    mask[..., :4, :, :] = 1.0
    y = _apply(x, 1.0, mode, foreground_mask=mask)
    assert torch.equal(y[..., 4:, :, :], x[..., 4:, :, :])
    assert not torch.equal(y[..., :4, :, :], x[..., :4, :, :])


def test_multislice_and_volume_share_one_realisation():
    x = _ball()
    torch.testing.assert_close(
        _apply(x, 0.7, M.MULTISLICE_2D), _apply(x, 0.7, M.VOLUME_3D), rtol=0, atol=0
    )
