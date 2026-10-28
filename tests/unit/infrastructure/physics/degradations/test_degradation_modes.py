"""apply_degradation dispatch: every registry name in every acquisition mode."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    MissingInput,
    Requirement,
    SliceOrder,
    UnsupportedAcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.registry import check_requirements
from tests.utils.degradation_inputs import apply_degradation

M = AcquisitionMode
MOTION_TRIO = ("translation_drift", "sudden_motion", "random_motion")
GEOMETRIES = {
    M.SLICE_2D: None,
    M.MULTISLICE_2D: AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None),
    M.VOLUME_3D: AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None),
}


# Concomitant phase needs physical distances in volume modes (Ruling 26).
NEEDS_VOXEL_MM = {"concomitant_phase"}


def _geometry(name: str, mode: AcquisitionMode) -> AcquisitionGeometry | None:
    geom = GEOMETRIES[mode]
    if geom is not None and name in NEEDS_VOXEL_MM:
        return replace(geom, voxel_mm=(1.0, 1.0, 1.0))
    return geom


def _input(mode: AcquisitionMode, channels: int = 1) -> torch.Tensor:
    shape = (1, channels, 16, 16) if mode is M.SLICE_2D else (1, channels, 4, 16, 16)
    g = torch.Generator().manual_seed(3)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


@pytest.mark.parametrize("mode", list(M))
@pytest.mark.parametrize("name", list(DEGRADATION_REGISTRY))
def test_every_name_in_every_mode(name: str, mode: AcquisitionMode) -> None:
    spec = DEGRADATION_REGISTRY[name]
    x = _input(mode, 2 if Requirement.MULTICOIL in spec.requires else 1)
    if mode in spec.supported_modes:
        y = apply_degradation(name, x, 0.6, seed=1, geometry=_geometry(name, mode))
        assert y.shape == x.shape
        assert torch.isfinite(torch.view_as_real(y) if y.is_complex() else y).all()
        return
    with pytest.raises(UnsupportedAcquisitionMode) as err:
        apply_degradation(name, x, 0.6, seed=1, geometry=GEOMETRIES[mode])
    for supported in spec.supported_modes:
        assert supported.value in str(err.value)


@pytest.mark.parametrize("name", MOTION_TRIO)
def test_motion_trio_multislice_serves_both_slice_orders(name: str) -> None:
    # Task 14: a sequential stack is timed natively, so it differs from the interleaved one.
    x = _input(M.MULTISLICE_2D)
    inter = replace(GEOMETRIES[M.MULTISLICE_2D], slice_order=SliceOrder.INTERLEAVED)
    seq = replace(inter, slice_order=SliceOrder.SEQUENTIAL)
    a = apply_degradation(name, x, 0.6, seed=1, geometry=inter)
    b = apply_degradation(name, x, 0.6, seed=1, geometry=seq)
    assert a.shape == b.shape == x.shape
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("name", MOTION_TRIO)
def test_motion_trio_interleaved_shares_one_displacement(name: str) -> None:
    # Every slice of an interleaved stack sees the same d(ky), so equal slices stay equal.
    one = _input(M.SLICE_2D).unsqueeze(2)
    y = apply_degradation(
        name, one.expand(1, 1, 4, 16, 16), 0.6, seed=1, geometry=GEOMETRIES[M.MULTISLICE_2D]
    )
    for d in range(1, 4):
        assert torch.equal(y[:, :, d], y[:, :, 0])


def test_slice_pve_points_to_through_plane_pve() -> None:
    assert "through_plane_pve" in DEGRADATION_REGISTRY["slice_pve"].description


def test_volume_without_geometry_raises() -> None:
    with pytest.raises(ValueError, match="explicit AcquisitionMode"):
        apply_degradation("gibbs", _input(M.MULTISLICE_2D), 0.5)


def test_slice_with_volume_geometry_raises() -> None:
    with pytest.raises(ValueError, match="cannot carry acquisition mode"):
        apply_degradation("gibbs", _input(M.SLICE_2D), 0.5, geometry=GEOMETRIES[M.VOLUME_3D])


def test_unknown_name_lists_valid_names() -> None:
    with pytest.raises(KeyError, match="Valid:"):
        apply_degradation("nope", _input(M.SLICE_2D), 0.5)


def _requiring(*reqs: Requirement):
    return replace(DEGRADATION_REGISTRY["gibbs"], requires=frozenset(reqs))


@pytest.mark.parametrize(
    ("req", "kwarg"),
    [
        (Requirement.QMAPS, "qmaps"),
        (Requirement.FAT_MAP, "fat_map"),
        (Requirement.VESSEL_MAP, "vessel_map"),
        (Requirement.TRAJECTORY_RADIAL, "trajectory"),
    ],
)
def test_requirement_kwargs_checked(req: Requirement, kwarg: str) -> None:
    x = _input(M.SLICE_2D)
    with pytest.raises(MissingInput, match=req.value):
        check_requirements(_requiring(req), x, {})
    check_requirements(_requiring(req), x, {kwarg: object()})


def test_multicoil_requirement_reads_channels() -> None:
    with pytest.raises(MissingInput, match="multicoil"):
        check_requirements(_requiring(Requirement.MULTICOIL), _input(M.SLICE_2D), {})
    check_requirements(_requiring(Requirement.MULTICOIL), torch.zeros(1, 2, 4, 4), {})


def test_epi_requirement_is_unmet_without_a_profile() -> None:
    with pytest.raises(MissingInput, match="sequence_epi"):
        check_requirements(_requiring(Requirement.SEQUENCE_EPI), _input(M.SLICE_2D), {})
