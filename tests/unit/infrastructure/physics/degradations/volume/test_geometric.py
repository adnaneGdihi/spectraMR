"""Native multislice and 3-D geometric operators: geometric_warp, gradient_nonlinearity."""

from __future__ import annotations

from itertools import pairwise, product

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
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.degradations.volume.geometric import (
    gnl_displacement_px,
    warp_displacement_mm,
)

M = AcquisitionMode
MODES = (M.MULTISLICE_2D, M.VOLUME_3D)
NAMES = ("geometric_warp", "gradient_nonlinearity")
CASES = [(n, m) for n in NAMES for m in MODES]
VOXEL = (2.0, 1.0, 1.0)


def _geom(mode, voxel_mm=VOXEL) -> AcquisitionGeometry:
    return AcquisitionGeometry(mode=mode, voxel_mm=voxel_mm)


def _phantom(shape=(1, 2, 6, 24, 24), seed=3) -> torch.Tensor:
    # Smooth complex phantom: a warp's residual on it grows with the displacement.
    g = torch.Generator().manual_seed(seed)
    z, y, x = mm_grid(tuple(shape[-3:]), None)
    amp = 1.0 + torch.rand((*shape[:2], 1, 1, 1), generator=g)
    mag = torch.exp(-(x**2 + y**2 + z**2) / 0.3) + 0.5 * torch.cos(2.0 * x + y)
    return torch.polar(amp * mag.abs(), (1.5 * x - y + z).expand(shape))


def _apply(name, x, theta, mode, seed=1, geometry=None, **kw):
    return apply_degradation(name, x, theta, seed=seed, geometry=geometry or _geom(mode), **kw)


def _rms(a, b) -> float:
    return float((a - b).abs().pow(2).mean().sqrt())


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_declared_native(name, mode):
    assert DEGRADATION_REGISTRY[name].realisation[mode] is Realisation.NATIVE


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_theta_zero_is_identity(name, mode):
    x = _phantom()
    assert torch.equal(_apply(name, x, 0.0, mode), x)


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_monotone_finite_shape_deterministic(name, mode):
    x = _phantom()
    outs = [_apply(name, x, t, mode) for t in (0.0, 0.3, 0.6, 1.0)]
    for y in outs:
        assert y.shape == x.shape
        assert torch.isfinite(torch.view_as_real(y)).all()
    res = [_rms(x, y) for y in outs]
    assert all(a < b for a, b in pairwise(res)), res
    assert torch.equal(_apply(name, x, 0.6, mode), outs[2])


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_batch_invariant(name, mode):
    x = _phantom()
    alone = _apply(name, x, 0.6, mode)
    pair = _apply(name, torch.cat([x, 10 * _phantom(seed=9)]), 0.6, mode)
    torch.testing.assert_close(pair[:1], alone, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("name", "mode"), CASES)
def test_runs_without_voxel_size_and_on_real_input(name, mode):
    x = _phantom().abs()
    y = _apply(name, x, 0.6, mode, geometry=_geom(mode, voxel_mm=None))
    assert y.shape == x.shape and y.is_complex() and torch.isfinite(torch.view_as_real(y)).all()


@pytest.mark.parametrize("name", NAMES)
def test_multislice_and_volume_share_one_field(name):
    x = _phantom()
    ms = _apply(name, x, 0.7, M.MULTISLICE_2D)
    torch.testing.assert_close(ms, _apply(name, x, 0.7, M.VOLUME_3D), rtol=0, atol=0)


def _ramps(shape, voxel_mm) -> torch.Tensor:
    # Channels (z, y, x) in mm: a trilinear pull-back warp moves each ramp by exactly d_mm.
    z, y, x = mm_grid(shape, voxel_mm)
    return torch.stack([z, y, x]).unsqueeze(0).to(torch.complex64)


@pytest.mark.parametrize("voxel_mm", [(5.0, 1.0, 1.0), (1.0, 1.0, 1.0)])
def test_warp_peak_displacement_in_mm_is_the_declared_one(voxel_mm):
    shape, theta = (8, 32, 32), 0.5
    d_mm = warp_displacement_mm(1, shape, voxel_mm, theta, 0, torch.device("cpu"))
    declared = theta * 16.0 * voxel_mm[1]
    assert float(d_mm.pow(2).sum(1).sqrt().amax()) == pytest.approx(declared, rel=1e-5)

    # Measured through the operator: ramp phantoms record d_mm wherever r + d stays inside.
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=voxel_mm)
    ramps = _ramps(shape, voxel_mm)
    measured = (
        apply_degradation("geometric_warp", ramps, theta, seed=0, geometry=geom) - ramps
    ).real
    d_px = d_mm / torch.tensor(voxel_mm).view(1, 3, 1, 1, 1)
    idx = torch.stack(torch.meshgrid(*[torch.arange(n) for n in shape], indexing="ij"))
    lands = idx + d_px[0]
    inside = ((lands >= 0) & (lands <= torch.tensor(shape).view(3, 1, 1, 1) - 1)).all(0)
    torch.testing.assert_close(measured[0][:, inside], d_mm[0][:, inside], rtol=0, atol=1e-3)
    # Border padding can only shorten a displacement that would leave the volume.
    peak = measured[0].pow(2).sum(0).sqrt().amax()
    assert float(peak) <= declared * (1 + 1e-4)
    assert float(peak) > 0.8 * declared


def test_warp_thick_slices_move_by_mm_over_slice_thickness():
    shape, theta, voxel_mm = (8, 32, 32), 1.0, (5.0, 1.0, 1.0)
    geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=voxel_mm)
    d_mm = warp_displacement_mm(1, shape, voxel_mm, theta, 2, torch.device("cpu"))
    # The through-plane displacement is d_z / 5 slices, never 16 slices.
    z_idx = torch.arange(shape[0], dtype=torch.float32).view(1, 1, -1, 1, 1).expand(1, 1, *shape)
    out = apply_degradation(
        "geometric_warp", z_idx.to(torch.complex64), theta, seed=2, geometry=geom
    )
    moved = (out.real - z_idx)[0, 0]
    dz_slices = d_mm[0, 0] / 5.0
    inside = ((z_idx[0, 0] + dz_slices) >= 0) & ((z_idx[0, 0] + dz_slices) <= shape[0] - 1)
    torch.testing.assert_close(moved[inside], dz_slices[inside], rtol=0, atol=1e-4)
    assert float(dz_slices.abs().max()) <= 16.0 / 5.0 + 1e-4


def test_warp_without_voxel_size_peaks_at_sixteen_voxels():
    d_mm = warp_displacement_mm(2, (6, 24, 24), None, 1.0, 5, torch.device("cpu"))
    peak = d_mm.pow(2).sum(1).sqrt().amax(dim=(1, 2, 3))
    torch.testing.assert_close(peak, torch.full((2,), 16.0), rtol=1e-5, atol=0)


def _normalised(d_px, shape):
    return d_px * 2.0 / torch.tensor(shape, dtype=d_px.dtype).view(1, 3, 1, 1, 1)


def test_gnl_has_no_order_zero_or_one_component():
    shape = (7, 12, 12)
    d = _normalised(gnl_displacement_px(2, shape, 1.0, 4, torch.device("cpu")), shape)
    z, y, x = (c.reshape(-1).double() for c in mm_grid(shape, None))
    exps = [e for e in product(range(4), repeat=3) if sum(e) <= 3]
    basis = torch.stack([z**a * y**b * x**c for a, b, c in exps], 1)
    coef = torch.linalg.lstsq(basis, d.reshape(6, -1).double().T).solution  # [K, 6]
    low = torch.tensor([sum(e) <= 1 for e in exps])
    assert float(coef[low].abs().max()) < 1e-6 * float(coef[~low].abs().max())


def test_gnl_fov_fraction_is_grid_size_independent():
    # Same seed, nested grids: index 2k of the fine grid is index k of the coarse one.
    coarse, fine = (4, 16, 16), (8, 32, 32)
    dc = _normalised(gnl_displacement_px(1, coarse, 1.0, 3, torch.device("cpu")), coarse)
    df = _normalised(gnl_displacement_px(1, fine, 1.0, 3, torch.device("cpu")), fine)
    torch.testing.assert_close(df[..., ::2, ::2, ::2], dc, rtol=1e-5, atol=1e-7)


def test_gnl_multislice_curves_the_slice():
    shape = (6, 24, 24)
    d = gnl_displacement_px(1, shape, 1.0, 1, torch.device("cpu"))
    assert float(d[:, 0].abs().max()) > 0.1  # through-plane dz: the potato-chip slice
    z_idx = torch.arange(shape[0], dtype=torch.float32).view(1, 1, -1, 1, 1).expand(1, 1, *shape)
    geom = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None)
    out = apply_degradation(
        "gradient_nonlinearity", z_idx.to(torch.complex64), 1.0, seed=1, geometry=geom
    )
    torch.testing.assert_close(out, warp_volume(z_idx.to(torch.complex64), d))
    assert float((out.real - z_idx).abs().max()) > 0.1


def test_gnl_central_plane_matches_the_2d_coefficient_scale():
    # At z = 0 only the in-plane monomials survive: the 2-D order-2..3 family at the same scale.
    shape = (9, 32, 32)
    d = _normalised(gnl_displacement_px(64, shape, 1.0, 0, torch.device("cpu")), shape)
    centre = d[:, 1:, shape[0] // 2]  # (dy, dx) on the z = 0 plane
    _, y, x = (c[shape[0] // 2] for c in mm_grid(shape, None))
    exps = [(p, q) for p in range(4) for q in range(4) if 2 <= p + q <= 3]
    basis = torch.stack([y.reshape(-1) ** p * x.reshape(-1) ** q for p, q in exps], 1)
    coef = torch.linalg.lstsq(basis, centre.reshape(-1, basis.shape[0]).T).solution
    assert float(coef.std()) == pytest.approx(7e-5 * 256, rel=0.15)


def test_gnl_thick_slices_move_by_mm_over_slice_thickness():
    # Slice i at 5 mm sits where slice 5i at 1 mm does (same half-extent): same field in mm,
    # so dz in voxels is 1/5 and the in-plane voxel displacement is unchanged.
    cpu = torch.device("cpu")
    thick = gnl_displacement_px(1, (8, 64, 64), 1.0, 2, cpu, voxel_mm=(5.0, 1.0, 1.0))
    thin = gnl_displacement_px(1, (40, 64, 64), 1.0, 2, cpu, voxel_mm=(1.0, 1.0, 1.0))[:, :, ::5]
    torch.testing.assert_close(thick[:, 0], thin[:, 0] / 5.0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(thick[:, 1:], thin[:, 1:], rtol=1e-5, atol=1e-6)


def test_gnl_thin_slab_samples_the_field_of_its_position():
    # A 4 mm slab sees the cube's field at its own z, never z-terms scaled up to its extent.
    cpu, vox = torch.device("cpu"), (1.0, 1.0, 1.0)
    slab = gnl_displacement_px(1, (4, 64, 64), 1.0, 6, cpu, voxel_mm=vox)
    cube = gnl_displacement_px(1, (64, 64, 64), 1.0, 6, cpu, voxel_mm=vox)
    torch.testing.assert_close(slab, cube[:, :, 30:34], rtol=1e-5, atol=1e-5)
    assert float(slab[:, 0].abs().max()) <= float(cube[:, 0].abs().max())


def test_gnl_volume_rejects_the_2d_severity_override():
    with pytest.raises(ValueError, match="2-D only"):
        _apply("gradient_nonlinearity", _phantom(), 0.5, M.VOLUME_3D, severity=1e-5)
