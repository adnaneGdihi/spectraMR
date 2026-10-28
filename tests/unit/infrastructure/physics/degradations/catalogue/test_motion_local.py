"""local_intermittent_motion: a compact regional warp present during one contiguous block of lines."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionMode,
    Realisation,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.degradations.catalogue.motion_local import (
    motion_block,
    regional_displacement_mm,
)
from spectramr.infrastructure.physics.degradations.catalogue.motion_timing import (
    slice_geometry,
)
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import ENCODING_DIMS
from spectramr.infrastructure.physics.fft_ops import fftnc

from .invariants import GEOMETRIES, assert_axis_invariants, phantom, rms

M = AcquisitionMode
NAME = "local_intermittent_motion"


def _apply(x, theta, geometry=None, seed=1, **kw):
    return apply_degradation(NAME, x, theta, seed=seed, geometry=geometry, **kw)


def _as5(x, geometry):
    return (x.unsqueeze(2), slice_geometry()) if geometry is None else (x, geometry)


@pytest.mark.parametrize("mode", list(M))
def test_every_mode_is_native(mode):
    spec = DEGRADATION_REGISTRY[NAME]
    assert spec.realisation[mode] is Realisation.NATIVE
    assert spec.requires == frozenset() and spec.condition is None


def test_declared_severity():
    params = {p.name: p for p in DEGRADATION_REGISTRY[NAME].severity.params}
    disp, frac = params["peak_regional_displacement"], params["motion_block_fraction"]
    assert (disp.units, disp.at_theta_min, disp.at_theta_max) == ("mm", 0.0, 10.0)
    assert (frac.units, frac.at_theta_min, frac.at_theta_max) == ("fraction", 0.0, 0.3)


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_invariants(geom):
    assert_axis_invariants(NAME, GEOMETRIES[geom])


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_outside_the_region_a_full_block_leaves_the_image_unchanged(geom):
    geometry = GEOMETRIES[geom]
    x = phantom(geometry)
    out = _apply(x, 1.0, geometry, max_block_fraction=1.0)
    x5, g5 = _as5(x, geometry)
    disp = regional_displacement_mm(
        1, tuple(x5.shape[-3:]), 1, planar=g5.mode is M.SLICE_2D, peak_mm=10.0
    )
    outside = (disp.abs().sum(1, keepdim=True) == 0).expand_as(x5)
    out5 = out.unsqueeze(2) if geometry is None else out
    assert 0.0 < float((~outside).float().mean()) < 0.5  # the region is local but not empty
    torch.testing.assert_close(out5[outside], x5[outside], rtol=0, atol=1e-5)
    assert rms(out5[~outside], x5[~outside]) > 1e-3


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_a_vanishing_block_returns_the_input(geom):
    geometry = GEOMETRIES[geom]
    x = phantom(geometry)
    assert torch.equal(_apply(x, 1.0, geometry, max_block_fraction=1e-6), x)
    small = rms(_apply(x, 1.0, geometry, max_block_fraction=0.05), x)
    large = rms(_apply(x, 1.0, geometry, max_block_fraction=0.5), x)
    assert small < large


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_the_block_is_the_declared_contiguous_fraction_of_acquired_lines(geom):
    x5, g5 = _as5(phantom(GEOMETRIES[geom]), GEOMETRIES[geom])
    d, h = x5.shape[-3], x5.shape[-2]
    block = motion_block(g5, d, h, 0.3, seed=1, tr_s=0.5)
    assert int(block.sum()) == round(0.3 * d * h)
    picked = shot_index(line_times(g5, d, h, tr_s=0.5), d * h)[block].sort().values
    assert torch.equal(picked, torch.arange(int(picked[0]), int(picked[0]) + picked.numel()))


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_only_the_block_lines_of_slices_the_region_reaches_move(geom):
    geometry = GEOMETRIES[geom]
    x = phantom(geometry)
    out = _apply(x, 1.0, geometry)
    x5, g5 = _as5(x, geometry)
    out5 = out.unsqueeze(2) if geometry is None else out
    dims = ENCODING_DIMS[g5.mode]
    moved = ((fftnc(out5, dims=dims) - fftnc(x5, dims=dims)).abs() > 1e-5).any(-1).any(1)[0]
    block = motion_block(g5, x5.shape[-3], x5.shape[-2], 0.3, seed=1, tr_s=0.5)
    if g5.mode is M.VOLUME_3D:  # one 3-D k-space: every block line sees the regional change
        assert torch.equal(moved, block)
        return
    disp = regional_displacement_mm(
        1, tuple(x5.shape[-3:]), 1, planar=g5.mode is M.SLICE_2D, peak_mm=10.0
    )
    reached = (disp.abs().sum((1, 3, 4)) > 0)[0]  # slices the region intersects
    assert reached.any()
    assert torch.equal(moved, block & reached[:, None])


@pytest.mark.parametrize("planar", [True, False])
def test_regional_field_peaks_at_the_declared_displacement(planar):
    shape = (1, 64, 64) if planar else (16, 32, 32)
    disp = regional_displacement_mm(2, shape, 4, planar=planar, peak_mm=10.0)
    assert disp.shape == (2, 3, *shape)
    peak = disp.norm(dim=1).flatten(1).max(1).values
    # The region is centred on a voxel, so the declared peak is realised exactly.
    torch.testing.assert_close(peak, torch.full((2,), 10.0), rtol=0, atol=1e-5)
    if planar:
        assert bool((disp[:, 0] == 0).all())


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"peak_mm": -1.0}, "peak_mm"),
        ({"max_block_fraction": 0.0}, "max_block_fraction"),
        ({"max_block_fraction": 1.5}, "max_block_fraction"),
        ({"tr_s": 0.0}, "tr_s"),
    ],
)
def test_bad_knobs_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        _apply(phantom(None), 0.5, **kw)


def test_a_profile_supplies_the_voxel_size_and_its_tr_is_not_read():
    from dataclasses import replace

    from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T

    x = phantom(None)
    by_profile = apply_degradation(NAME, x, 0.6, seed=1, profile=replace(FASTMRI_3T, tr_ms=7.0))
    torch.testing.assert_close(by_profile, _apply(x, 0.6, FASTMRI_3T.geometry), rtol=0, atol=0)
    # 0.5 mm pixels double the displacement in px, so the voxel size was read.
    assert rms(by_profile, x) != rms(_apply(x, 0.6), x)


def test_block_and_region_draw_from_distinct_streams():
    from spectramr.infrastructure.physics.degradations import derive_axis_seed

    assert derive_axis_seed(1, "block") != derive_axis_seed(1, "region")
    assert derive_axis_seed(1, "block") != derive_axis_seed(2, "block")
