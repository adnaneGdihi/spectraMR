"""Frame conventions the repeat-session registration and the warp artifact share.

SimpleITK reads a NIfTI header into ITK's LPS frame while nibabel and TorchIO read the same
header into RAS; the two place one voxel at coordinates that differ by ``diag(-1, -1, 1)``. The
hardware-warp contract (:mod:`spectramr.data.hardware_warp_artifacts`) is expressed in RAS, the
frame of the framework's data path, so everything the registration measures in LPS is mapped at
its boundary with the helpers here. ``LPS_TO_RAS`` is a proper rotation (a half turn about the
z axis), so rotation vectors map as ordinary vectors, ``omega_ras = P omega_lps``, and rotation
matrices by conjugation, ``M_ras = P M_lps P``.

The expansion centre of the harmonic basis is the field of view's centre, a magnet-fixed point on
a fixed-FOV scanner, and never a head-fixed one such as the mask centroid: the warp lives in the
magnet frame, and a centre that moved with the head would fold the pose into the field. A Swoop
export puts the NIfTI origin at the FOV corner, so the header origin is not a usable centre.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = [
    "LPS_TO_RAS",
    "fov_center_lps",
    "matrix_to_ras",
    "require_shared_frame",
    "vectors_to_ras",
]

LPS_TO_RAS: np.ndarray = np.diag([-1.0, -1.0, 1.0])


def fov_center_lps(image: Any) -> np.ndarray:
    """Physical centre (mm, LPS) of ``image``'s field of view, from its SimpleITK header."""
    size = np.asarray(image.GetSize(), dtype=np.float64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    return origin + direction @ (spacing * (size - 1.0) / 2.0)


def require_shared_frame(fixed: Any, moving: Any, *, tolerance_mm: float) -> np.ndarray:
    """The FOV centre (mm, LPS) two sessions share, or ``ValueError`` when they do not.

    The headers place both sessions in the magnet frame only if the field of view did not move
    between them. A centre that differs by more than ``tolerance_mm`` or direction cosines that
    differ mean the two images are not in one frame, and a pair like that cannot enter a pose set
    whose generators and residuals are all expressed about one point.
    """
    c_fixed, c_moving = fov_center_lps(fixed), fov_center_lps(moving)
    shift = float(np.linalg.norm(c_fixed - c_moving))
    same_axes = np.allclose(
        np.asarray(fixed.GetDirection()), np.asarray(moving.GetDirection()), atol=1e-6
    )
    if shift > tolerance_mm or not same_axes:
        raise ValueError(
            f"the two sessions do not share a frame: FOV centres {c_fixed.round(2).tolist()} and "
            f"{c_moving.round(2).tolist()} differ by {shift:.2f} mm (tolerance {tolerance_mm:g}) "
            f"or their direction cosines differ; the pose set cannot mix frames"
        )
    return (c_fixed + c_moving) / 2.0


def vectors_to_ras(vectors: torch.Tensor) -> torch.Tensor:
    """``[..., 3]`` LPS positions or displacements in RAS (a rotation, so both map alike)."""
    return vectors @ torch.as_tensor(LPS_TO_RAS, dtype=vectors.dtype).T


def matrix_to_ras(matrix: torch.Tensor) -> torch.Tensor:
    """A ``3x3`` linear map on LPS vectors as the same map on RAS vectors, ``P M P``."""
    p = torch.as_tensor(LPS_TO_RAS, dtype=matrix.dtype)
    return p @ matrix @ p
