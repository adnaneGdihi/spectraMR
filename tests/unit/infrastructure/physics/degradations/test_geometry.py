"""Acquisition geometry: validation, provenance, and the rank x mode resolution table."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations.geometry import (
    KZ_DIM,
    PE_DIM,
    RO_DIM,
    SLICE_DIM,
    AcquisitionGeometry,
    AcquisitionMode,
    LineOrder,
    SliceOrder,
    resolve_geometry,
)

M = AcquisitionMode


def test_axis_roles_fixed_by_layout() -> None:
    assert (SLICE_DIM, KZ_DIM, PE_DIM, RO_DIM) == (-3, -3, -2, -1)


def test_defaults_and_to_meta() -> None:
    g = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(3.0, 1.0, 1.0), slice_thickness_mm=3.0)
    meta = g.to_meta()
    assert meta == {
        "mode": "multislice_2d",
        "voxel_mm": (3.0, 1.0, 1.0),
        "slice_thickness_mm": 3.0,
        "slice_gap_mm": 0.0,
        "slice_order": "interleaved",
        "line_order": "linear",
        "b0_axis": "slice",
    }
    assert all(type(v) is not SliceOrder and type(v) is not LineOrder for v in meta.values())
    assert type(meta["mode"]) is str


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": M.VOLUME_3D, "voxel_mm": (1.0, 1.0)},
        {"mode": M.VOLUME_3D, "voxel_mm": (1.0, 0.0, 1.0)},
        {"mode": M.VOLUME_3D, "voxel_mm": (1.0, -1.0, 1.0)},
        {"mode": M.SLICE_2D, "voxel_mm": None, "slice_thickness_mm": 3.0},
        {"mode": M.VOLUME_3D, "voxel_mm": None, "slice_thickness_mm": 3.0},
        {"mode": M.SLICE_2D, "voxel_mm": None, "slice_gap_mm": 0.5},
        {"mode": M.VOLUME_3D, "voxel_mm": None, "slice_gap_mm": 0.5},
        {"mode": M.MULTISLICE_2D, "voxel_mm": None, "slice_gap_mm": -0.5},
        {"mode": M.MULTISLICE_2D, "voxel_mm": None, "slice_thickness_mm": 0.0},
        {"mode": M.MULTISLICE_2D, "voxel_mm": None, "slice_thickness_mm": -2.0},
        {"mode": M.MULTISLICE_2D, "voxel_mm": None, "b0_axis": "kz"},
        {"mode": "multislice_2d", "voxel_mm": None},
    ],
)
def test_invalid_geometry_raises_at_construction(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AcquisitionGeometry(**kwargs)


def test_valid_multislice_with_gap() -> None:
    g = AcquisitionGeometry(
        mode=M.MULTISLICE_2D,
        voxel_mm=(4.0, 0.9, 0.9),
        slice_thickness_mm=3.0,
        slice_gap_mm=1.0,
        slice_order=SliceOrder.SEQUENTIAL,
        line_order=LineOrder.CENTRIC,
        b0_axis="pe",
    )
    assert g.slice_gap_mm == 1.0


def test_slice_spacing_below_thickness_plus_gap_raises() -> None:
    with pytest.raises(ValueError, match=r"voxel_mm\[0\].*3\.0.*\(slice, pe, ro\)"):
        AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(1.0, 1.0, 3.0), slice_thickness_mm=3.0)
    with pytest.raises(ValueError, match="slice_gap_mm"):
        AcquisitionGeometry(
            mode=M.MULTISLICE_2D, voxel_mm=(3.5, 1.0, 1.0), slice_thickness_mm=3.0, slice_gap_mm=1.0
        )


def test_slice_spacing_equal_to_thickness_passes() -> None:
    g = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=(3.0, 1.0, 1.0), slice_thickness_mm=3.0)
    assert g.voxel_mm == (3.0, 1.0, 1.0)


def test_4d_none_resolves_slice_2d() -> None:
    g = resolve_geometry(torch.zeros(1, 1, 4, 4), None)
    assert g.mode is M.SLICE_2D and g.voxel_mm is None


def test_4d_slice_2d_passes_through() -> None:
    g = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=(1.0, 1.0, 1.0))
    assert resolve_geometry(torch.zeros(1, 1, 4, 4), g) is g


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_4d_volume_mode_raises(mode: AcquisitionMode) -> None:
    with pytest.raises(ValueError, match=rf"4-D.*{mode.value}"):
        resolve_geometry(torch.zeros(1, 1, 4, 4), AcquisitionGeometry(mode=mode, voxel_mm=None))


def test_5d_none_raises() -> None:
    with pytest.raises(ValueError, match="5-D input needs an explicit AcquisitionMode"):
        resolve_geometry(torch.zeros(1, 1, 2, 4, 4), None)


def test_5d_slice_2d_raises() -> None:
    g = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None)
    with pytest.raises(ValueError, match=r"5-D.*slice_2d"):
        resolve_geometry(torch.zeros(1, 1, 2, 4, 4), g)


@pytest.mark.parametrize("mode", [M.MULTISLICE_2D, M.VOLUME_3D])
def test_5d_volume_mode_passes_through(mode: AcquisitionMode) -> None:
    g = AcquisitionGeometry(mode=mode, voxel_mm=None)
    assert resolve_geometry(torch.zeros(1, 1, 2, 4, 4), g) is g


@pytest.mark.parametrize("shape", [(4, 4), (1, 4, 4), (1, 1, 1, 2, 4, 4)])
@pytest.mark.parametrize("mode", [None, M.SLICE_2D, M.MULTISLICE_2D, M.VOLUME_3D])
def test_other_ranks_raise(shape: tuple[int, ...], mode: AcquisitionMode | None) -> None:
    g = None if mode is None else AcquisitionGeometry(mode=mode, voxel_mm=None)
    with pytest.raises(ValueError, match=rf"{len(shape)}-D"):
        resolve_geometry(torch.zeros(shape), g)
