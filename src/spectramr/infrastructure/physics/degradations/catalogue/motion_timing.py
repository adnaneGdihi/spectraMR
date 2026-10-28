"""What the patient-motion catalogue axes share: one 5-D path for every mode, and the axis names.

A 2-D slice runs as a one-slice ``SLICE_2D`` volume, so every mode times its lines with
``acquisition_timing.line_times`` and assembles k-space with ``time_states``: lines are ``ky``
(2-D), ``(slice, ky)`` (multislice, each slice its own 2-D k-space) or ``(kz, ky)`` (3-D).
Anatomical directions cannot be read off a tensor, so callers name them by encoding axis.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import MOTION_TR_S

__all__ = [
    "AXIS_INDEX",
    "MOTION_TR_S",
    "check_input",
    "encoded_axes",
    "run_as_volume",
    "slice_geometry",
]

_M = AcquisitionMode
#: Index of each named axis in ``(D, H, W)`` and in ``voxel_mm``.
AXIS_INDEX = {"slice": 0, "pe": 1, "ro": 2}
# Axes the encoding resolves in k-space; a multislice stack's slice axis is excited, not encoded.
_ENCODED = {
    _M.SLICE_2D: frozenset({"pe", "ro"}),
    _M.MULTISLICE_2D: frozenset({"pe", "ro"}),
    _M.VOLUME_3D: frozenset({"slice", "pe", "ro"}),
}


def encoded_axes(mode: AcquisitionMode) -> frozenset[str]:
    """The named axes that ``mode`` Fourier-encodes."""
    return _ENCODED[mode]


def slice_geometry(voxel_mm: tuple[float, float, float] | None = None) -> AcquisitionGeometry:
    """The geometry a 2-D kernel runs under: one slice of the given (or unknown) voxel size."""
    return AcquisitionGeometry(mode=_M.SLICE_2D, voxel_mm=voxel_mm)


def check_input(x: Tensor, geometry: AcquisitionGeometry, name: str) -> None:
    """Raise unless ``x`` is ``(B, C, D, H, W)``, with ``D == 1`` for a ``SLICE_2D`` geometry."""
    if x.dim() != 5:
        raise ValueError(f"{name} expects (B, C, D, H, W), got {tuple(x.shape)}")
    if geometry.mode is _M.SLICE_2D and x.shape[-3] != 1:
        raise ValueError(f"{name}: a slice_2d geometry holds one slice, got D={x.shape[-3]}")


def run_as_volume(
    impl: Callable[..., Tensor],
    x: Tensor,
    theta: float,
    seed: int,
    voxel_mm: tuple[float, float, float] | None = None,
    **kwargs,
) -> Tensor:
    """Run a 5-D implementation on a ``(B, C, H, W)`` slice as a one-slice ``SLICE_2D`` volume."""
    if x.dim() != 4:
        raise ValueError(f"the 2-D kernel expects (B, C, H, W), got {tuple(x.shape)}")
    return impl(x.unsqueeze(2), theta, seed, slice_geometry(voxel_mm), **kwargs).squeeze(2)
