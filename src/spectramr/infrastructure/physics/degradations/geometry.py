"""How a degradation's input was acquired, and which acquisition a tensor's rank admits.

Axis roles are fixed by the canonical layout rather than declared per call: 2-D
``(B, C, H, W)`` and volumes ``(B, C, D, H, W)``, where dim -3 is the slice axis
(multislice) or kz (3-D), dim -2 the phase-encode axis and dim -1 the readout.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum, StrEnum
from typing import Any, Literal

import torch

__all__ = [
    "KZ_DIM",
    "PE_DIM",
    "RO_DIM",
    "SLICE_DIM",
    "AcquisitionGeometry",
    "AcquisitionMode",
    "LineOrder",
    "SliceOrder",
    "UNIT_VOXEL_MM",
    "resolve_geometry",
    "voxel_mm_or_unit",
]

SLICE_DIM = -3
KZ_DIM = -3
PE_DIM = -2
RO_DIM = -1

_B0_AXES = frozenset({"slice", "pe", "ro"})

#: Without a declared voxel size an mm-declared operator takes 1 mm isotropic voxels (px).
UNIT_VOXEL_MM: tuple[float, float, float] = (1.0, 1.0, 1.0)


def voxel_mm_or_unit(voxel_mm: tuple[float, float, float] | None) -> tuple[float, float, float]:
    """``voxel_mm``, or :data:`UNIT_VOXEL_MM` when the geometry leaves it unknown."""
    return UNIT_VOXEL_MM if voxel_mm is None else voxel_mm


class AcquisitionMode(StrEnum):
    """The encoding an input tensor came from."""

    SLICE_2D = "slice_2d"
    MULTISLICE_2D = "multislice_2d"
    VOLUME_3D = "volume_3d"


class SliceOrder(StrEnum):
    """Temporal order in which a multislice stack's slices are excited.

    ``SEQUENTIAL``: each slice is fully acquired before the next (separate excitation blocks
    in time). ``INTERLEAVED``: every slice is excited within each TR, so PE line ``ky`` of
    every slice shares one TR.
    """

    SEQUENTIAL = "sequential"
    INTERLEAVED = "interleaved"


class LineOrder(StrEnum):
    """Temporal order in which k-space lines (or kz/ky partitions) are acquired."""

    LINEAR = "linear"
    CENTRIC = "centric"
    KZ_INNER = "kz_inner"
    KY_INNER = "ky_inner"


@dataclass(frozen=True, slots=True)
class AcquisitionGeometry:
    """Acquisition mode plus the physical facts a geometry-aware degradation needs.

    ``voxel_mm`` is ``(slice/kz, pe, ro)`` in mm, following the dim order; ``None``
    means the voxel size is unknown and only pixel-unit degradations may run.
    """

    mode: AcquisitionMode
    voxel_mm: tuple[float, float, float] | None
    slice_thickness_mm: float | None = None
    slice_gap_mm: float = 0.0
    slice_order: SliceOrder = SliceOrder.INTERLEAVED
    line_order: LineOrder = LineOrder.LINEAR
    b0_axis: Literal["slice", "pe", "ro"] = "slice"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` on any field the declared mode cannot carry."""
        _require_member("mode", self.mode, AcquisitionMode)
        _require_member("slice_order", self.slice_order, SliceOrder)
        _require_member("line_order", self.line_order, LineOrder)
        if self.b0_axis not in _B0_AXES:
            raise ValueError(f"b0_axis must be one of {sorted(_B0_AXES)}, got {self.b0_axis!r}")
        if self.voxel_mm is not None:
            v = self.voxel_mm
            if not (
                isinstance(v, tuple)
                and len(v) == 3
                and all(isinstance(s, int | float) and not isinstance(s, bool) for s in v)
                and all(s > 0 for s in v)
            ):
                raise ValueError(f"voxel_mm must be a 3-tuple of positive floats, got {v!r}")
        if self.slice_gap_mm < 0:
            raise ValueError(f"slice_gap_mm must be >= 0, got {self.slice_gap_mm}")
        if self.slice_thickness_mm is not None and self.slice_thickness_mm <= 0:
            raise ValueError(f"slice_thickness_mm must be > 0, got {self.slice_thickness_mm}")
        if self.mode is not AcquisitionMode.MULTISLICE_2D and (
            self.slice_thickness_mm is not None or self.slice_gap_mm != 0
        ):
            raise ValueError(
                f"slice_thickness_mm / slice_gap_mm describe a multislice stack; "
                f"they cannot be set in mode {self.mode.value}"
            )
        if (
            self.voxel_mm is not None
            and self.slice_thickness_mm is not None
            and self.voxel_mm[0] < self.slice_thickness_mm + self.slice_gap_mm
        ):
            raise ValueError(
                f"slice spacing voxel_mm[0]={self.voxel_mm[0]} is smaller than "
                f"slice_thickness_mm={self.slice_thickness_mm} + slice_gap_mm={self.slice_gap_mm}; "
                "voxel_mm is ordered (slice, pe, ro)"
            )

    def to_meta(self) -> dict[str, Any]:
        """Every field as a plain value, enums as their strings, for provenance."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self)}


def _require_member(name: str, value: object, enum: type[Enum]) -> None:
    if not isinstance(value, enum):
        raise ValueError(f"{name} must be a {enum.__name__}, got {value!r}")


def _plain(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def resolve_geometry(x: torch.Tensor, geometry: AcquisitionGeometry | None) -> AcquisitionGeometry:
    """Check ``geometry`` against the rank of ``x``; a 4-D input defaults to SLICE_2D."""
    rank = x.dim()
    mode = None if geometry is None else geometry.mode
    if rank == 4:
        if geometry is None:
            return AcquisitionGeometry(mode=AcquisitionMode.SLICE_2D, voxel_mm=None)
        if mode is AcquisitionMode.SLICE_2D:
            return geometry
        raise ValueError(f"4-D input (B, C, H, W) cannot carry acquisition mode {mode}")
    if rank == 5:
        if geometry is None:
            raise ValueError("5-D input needs an explicit AcquisitionMode")
        if mode is AcquisitionMode.SLICE_2D:
            raise ValueError(
                f"5-D input (B, C, D, H, W) cannot carry acquisition mode {mode}; "
                "declare multislice_2d or volume_3d"
            )
        return geometry
    raise ValueError(
        f"{rank}-D input is not a degradation layout (expected 4-D or 5-D) "
        f"for acquisition mode {mode}"
    )
