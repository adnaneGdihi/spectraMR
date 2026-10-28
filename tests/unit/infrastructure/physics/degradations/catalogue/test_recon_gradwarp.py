"""gradwarp_residual: the net displacement left by a (1 + eps)-miscorrected gradient nonlinearity."""

from __future__ import annotations

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
from spectramr.infrastructure.physics.degradations.catalogue.recon_gradwarp import (
    residual_displacement_px,
)
from spectramr.infrastructure.physics.degradations.volume.geometric import gnl_displacement_px

from .invariants import GEOMETRIES, VOXEL, assert_axis_invariants, phantom

M = AcquisitionMode
NAME = "gradwarp_residual"
CPU = torch.device("cpu")


@pytest.mark.parametrize("mode", list(M))
def test_every_mode_is_native(mode):
    assert DEGRADATION_REGISTRY[NAME].realisation[mode] is Realisation.NATIVE


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


@pytest.mark.parametrize("voxel", [VOXEL, (2.0, 1.0, 0.5), None])
def test_peak_residual_displacement_is_eps_times_the_gradient_nonlinearity_peak_in_mm(voxel):
    shape, eps = (6, 32, 32), 0.15
    d_res = residual_displacement_px(1, shape, eps, 4, CPU, voxel)
    d_c = gnl_displacement_px(1, shape, 1.0, 4, CPU, voxel_mm=voxel)
    scale = torch.tensor(voxel or (1.0, 1.0, 1.0)).view(1, 3, 1, 1, 1)
    peak = lambda d: float((d * scale).pow(2).sum(1).sqrt().amax())  # noqa: E731 -- mm
    assert peak(d_res) == pytest.approx(eps * peak(d_c), rel=1e-5)
    assert peak(d_c) > 0 and torch.allclose(d_res, -eps * d_c)


def test_the_residual_is_the_same_field_as_gradient_nonlinearity_at_the_same_seed():
    """Small displacements of a smooth blob: gradient_nonlinearity warps by +theta d_c, the residual by
    -eps d_c (eps = 0.5 theta'), so the two deviations from the input are anti-parallel and equal in norm
    for theta' = 0.1 at theta = 0.05."""
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=(1.0, 1.0, 1.0))
    z, y, x = mm_grid((16, 48, 48), None)
    blob = torch.exp(-((x - 0.1) ** 2 + (1.2 * y) ** 2 + z**2) / 0.3).expand(1, 1, 16, 48, 48)
    blob = blob.to(torch.complex64)
    gnl = apply_degradation("gradient_nonlinearity", blob, 0.05, seed=2, geometry=geom) - blob
    res = apply_degradation(NAME, blob, 0.1, seed=2, geometry=geom) - blob
    cos = torch.vdot(gnl.flatten(), res.flatten()).real / (gnl.norm() * res.norm())
    assert float(cos) < -0.9
    assert float(res.norm() / gnl.norm()) == pytest.approx(1.0, rel=0.05)


def test_zero_epsilon_is_the_identity_of_the_net_displacement_form():
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL)
    x = phantom(geom)
    out = apply_degradation(NAME, x, 1.0, seed=2, geometry=geom, eps_max=0.0)
    torch.testing.assert_close(out, x, rtol=0, atol=1e-6)


def test_the_output_is_one_resampling_by_the_net_displacement():
    geom = AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL)
    x = phantom(geom)
    d = residual_displacement_px(1, (6, 32, 32), 0.5, 2, CPU, VOXEL)
    torch.testing.assert_close(
        apply_degradation(NAME, x, 1.0, seed=2, geometry=geom), warp_volume(x, d)
    )


def test_the_residual_is_first_order_in_epsilon():
    x = phantom(None)
    a, b = (float((apply_degradation(NAME, x, t, seed=1) - x).norm()) for t in (0.1, 0.2))
    assert b / a == pytest.approx(2.0, rel=0.1)


def test_a_slice_reads_its_in_plane_voxel_anisotropy_from_the_geometry():
    x = phantom(None)
    wide = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=(1.0, 1.0, 0.5))
    iso = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=(1.0, 1.0, 1.0))
    a = apply_degradation(NAME, x, 1.0, seed=1, geometry=wide)
    b = apply_degradation(NAME, x, 1.0, seed=1, geometry=iso)
    assert not torch.allclose(a, b, atol=1e-4)
    d_wide = residual_displacement_px(1, (1, 32, 32), 0.2, 1, CPU, wide.voxel_mm)
    d_iso = residual_displacement_px(1, (1, 32, 32), 0.2, 1, CPU, iso.voxel_mm)
    assert not torch.allclose(d_wide, d_iso)


def test_eps_max_is_validated():
    with pytest.raises(ValueError, match="eps_max"):
        apply_degradation(NAME, phantom(None), 0.5, eps_max=-1.0)


def test_displacement_in_mm_from_linear_ramps_on_anisotropic_voxels():
    """A ramp image x = index along one axis is displaced by exactly d (trilinear is exact on a ramp), so
    out - x is the displacement in voxels read from the OUTPUT of each axis. In mm (voxel (5, 1, 1)) the
    residual field must equal -eps times the gradient_nonlinearity field, component by component."""
    shape, eps, geom = (6, 32, 32), 0.3, AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=VOXEL)

    def field_mm(name, theta):
        comps, valid = [], torch.ones(shape, dtype=torch.bool)
        for ax in range(3):
            idx = torch.arange(shape[ax], dtype=torch.float32).view(
                [-1 if i == ax else 1 for i in range(3)]
            )
            out = apply_degradation(
                name,
                idx.expand(shape)[None, None].to(torch.complex64),
                theta,
                seed=4,
                geometry=geom,
            ).real[0, 0]
            valid &= (out > 0.01) & (out < shape[ax] - 1.01)
            comps.append((out - idx) * VOXEL[ax])
        return torch.stack(comps), valid

    res, v1 = field_mm(NAME, eps / 0.5)
    gnl, v2 = field_mm("gradient_nonlinearity", 1.0)
    ok = v1 & v2
    assert int(ok.sum()) > 0.5 * ok.numel()
    assert float(gnl[:, ok].abs().max()) > 0.5  # a real displacement, in mm
    torch.testing.assert_close(res[:, ok], -eps * gnl[:, ok], rtol=0, atol=2e-3)
