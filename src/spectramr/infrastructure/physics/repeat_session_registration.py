"""Register two repeat sessions of one head and read off the commutator observation.

The hardware-warp identification (:mod:`.hardware_warp_calibration`) consumes, per session pair,
the rigid generator ``xi = (omega, t)`` of the head motion and the non-rigid residual
``phi - h`` of the registration between the two observed images, sampled on points inside the
brain. This module produces both from two same-contrast volumes of the same scanner with
SimpleITK: a multi-resolution rigid fit gives ``h``; symmetric-forces Demons on the rigidly
aligned pair gives a displacement ``d`` with ``moving(h(x + d(x))) ~ fixed(x)``, so the map the
theory calls ``phi`` is ``x -> h(x + d(x)) = h(x) + M d(x)`` and the residual is ``M d`` with
``M`` the rotation matrix of ``h``.

Frame. The observation is in RAS millimetres about the field of view's centre
(:mod:`.scanner_frame`): SimpleITK measures in LPS, and a Swoop export puts the NIfTI origin at
the FOV corner, so a ball about the header origin would model the warp where the head is not.
Both sessions must agree on the centre or the pair is refused. Intensities: the scanner
recalibrates gain per session and both the mean-squares metric and Demons assume matched
intensities, so ``moving`` is histogram-matched to ``fixed`` first; the Demons smoothing is given
in mm and converted per axis because ITK takes it in voxels. Sign: ``phi`` maps fixed-session to
moving-session positions, and the fit is invariant to swapping the sessions, so the calibrated
``v`` keeps the theory's observed-to-true convention (:mod:`spectramr.data.hardware_warp_artifacts`).

SimpleITK is optional here so that the rest of the physics package imports without it; every
entry point raises a named ``ImportError`` when it is missing (non-negotiable 18).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from spectramr.infrastructure.physics.scanner_frame import (
    matrix_to_ras,
    require_shared_frame,
    vectors_to_ras,
)

try:
    import SimpleITK as sitk  # noqa: N813

    HAVE_SITK = True
except ImportError:  # pragma: no cover - exercised by the missing-dependency test
    sitk = None
    HAVE_SITK = False

__all__ = [
    "HAVE_SITK",
    "PairRegistration",
    "demons_displacement",
    "foreground_mask",
    "generator_from_rigid",
    "match_intensities",
    "physical_points",
    "register_session_pair",
    "require_sitk",
    "rigid_transform",
    "sample_residual",
]

logger = logging.getLogger(__name__)


def require_sitk() -> None:
    """Raise a named error when SimpleITK is absent; the caller decides whether to degrade."""
    if not HAVE_SITK:
        raise ImportError(
            "SimpleITK is required for repeat-session registration; install spectramr[preprocess]"
        )


@dataclass(frozen=True)
class PairRegistration:
    """One session pair's observation in RAS mm about ``center``: generator and residual."""

    omega: torch.Tensor  # axis-angle (rad), fixed -> moving
    translation: torch.Tensor  # mm, about ``center``
    rigid_matrix: torch.Tensor  # 3x3 rotation of h
    points: torch.Tensor  # [n, 3] mm relative to ``center``
    residual: torch.Tensor  # [n, 3] mm, phi(x) - h(x) = M d(x)
    center: torch.Tensor  # [3] mm RAS, the FOV centre both sessions share
    n_foreground: int  # voxels inside the fixed image's foreground mask
    n_in_ball: int  # of those, inside the ball, before the cap


def _axis_angle(matrix: np.ndarray) -> np.ndarray:
    """Rotation vector of a proper rotation matrix (the matrix logarithm on SO(3))."""
    cos = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cos))
    axis = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    )
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.zeros(3)
    return axis / norm * angle


def match_intensities(moving: Any, fixed: Any):
    """``moving`` histogram-matched to ``fixed`` (256 levels, 16 match points, above the mean)."""
    require_sitk()
    return sitk.HistogramMatching(
        sitk.Cast(moving, sitk.sitkFloat32), sitk.Cast(fixed, sitk.sitkFloat32), 256, 16, True
    )


def rigid_transform(
    fixed: Any, moving: Any, *, iterations: int = 300, metric: str = "mean_squares"
):
    """Multi-resolution rigid (Euler 3-D) fit; returns the fixed -> moving physical transform.

    Mean squares is the default for the same-contrast, intensity-matched pairs this serves;
    ``metric="mattes"`` selects Mattes mutual information for a cross-contrast pair or when the
    matching cannot be trusted.
    """
    require_sitk()
    if metric not in {"mean_squares", "mattes"}:
        raise ValueError(f"metric must be 'mean_squares' or 'mattes', got {metric!r}")
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    transform = sitk.Euler3DTransform(initial)
    reg = sitk.ImageRegistrationMethod()
    if metric == "mattes":
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    else:
        reg.SetMetricAsMeanSquares()
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-5,
        numberOfIterations=iterations,
        gradientMagnitudeTolerance=1e-8,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(transform, inPlace=True)
    reg.Execute(fixed, moving)
    return transform


def generator_from_rigid(
    transform: Any, center: np.ndarray | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(omega, t, M)`` of ``x -> M (x - c) + c + tau`` rewritten about ``center`` (default origin).

    SimpleITK parameterizes a rigid transform by a rotation about a centre ``c`` plus a
    translation ``tau``; about the origin the same map reads ``x -> M x + (c - M c + tau)``, and
    in coordinates ``y = x - center`` its translation gains ``(M - I) center``.
    """
    matrix = np.asarray(transform.GetMatrix(), dtype=np.float64).reshape(3, 3)
    c = np.asarray(transform.GetCenter(), dtype=np.float64)
    tau = np.asarray(transform.GetTranslation(), dtype=np.float64)
    translation = c - matrix @ c + tau
    if center is not None:
        translation = translation + (matrix - np.eye(3)) @ np.asarray(center, dtype=np.float64)
    as_t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)  # noqa: E731
    return as_t(_axis_angle(matrix)), as_t(translation), as_t(matrix)


def demons_displacement(
    fixed: Any, moving: Any, transform: Any, *, iterations: int = 300, smoothing_sigma: float = 2.0
):
    """Symmetric-forces Demons on ``moving`` resampled through ``transform`` onto ``fixed``.

    Returns the displacement-field image ``d`` (mm, physical) with ``moving_r(x + d(x)) ~ fixed(x)``;
    ``smoothing_sigma`` is in mm. Symmetric forces make the estimate inverse-consistent to first
    order, which removes the interpolation bias correlated with the sub-voxel shift.
    """
    require_sitk()
    moving_r = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    demons = sitk.SymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(int(iterations))
    demons.SetStandardDeviations([float(smoothing_sigma) / s for s in fixed.GetSpacing()])
    demons.SetSmoothDisplacementField(True)
    return demons.Execute(sitk.Cast(fixed, sitk.sitkFloat32), moving_r)


def physical_points(image: Any) -> torch.Tensor:
    """Physical position (mm) of every voxel, ``[n, 3]`` in ``(i, j, k)`` = ``(x, y, z)`` index order."""
    size = np.asarray(image.GetSize(), dtype=np.int64)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in size], indexing="ij"), -1).reshape(-1, 3)
    points = origin + (idx * spacing) @ direction.T
    return torch.as_tensor(points, dtype=torch.float64)


def _array_xyz(image: Any) -> np.ndarray:
    """SimpleITK's ``[z, y, x(, c)]`` array in ``(x, y, z(, c))`` index order."""
    arr = sitk.GetArrayFromImage(image)
    return np.ascontiguousarray(
        arr.transpose(2, 1, 0, 3) if arr.ndim == 4 else arr.transpose(2, 1, 0)
    )


def foreground_mask(image: Any) -> np.ndarray:
    """Otsu foreground of ``image`` as a boolean ``(x, y, z)`` array."""
    require_sitk()
    mask = sitk.OtsuThreshold(sitk.Cast(image, sitk.sitkFloat32), 0, 1, 200)
    return _array_xyz(mask).astype(bool)


def sample_residual(
    displacement: Any,
    rigid_matrix: torch.Tensor,
    *,
    mask: np.ndarray,
    ball_radius_mm: float,
    center: np.ndarray | None = None,
    max_points: int = 20000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """``(points, residual, n_in_ball)`` on foreground voxels inside the ball about ``center``.

    Points are relative to ``center``, the residual is ``M d`` in mm, and at most ``max_points``
    voxels are kept, drawn without replacement with ``seed`` so a pair's weight is bounded.
    """
    require_sitk()
    points = physical_points(displacement)
    if center is not None:
        points = points - torch.as_tensor(np.asarray(center, dtype=np.float64))
    field = torch.as_tensor(_array_xyz(displacement).reshape(-1, 3), dtype=torch.float64)
    keep = torch.as_tensor(mask.reshape(-1), dtype=torch.bool) & (
        points.norm(dim=1) < ball_radius_mm
    )
    keep &= torch.isfinite(field).all(dim=1)
    index = torch.nonzero(keep).squeeze(1)
    n_in_ball = int(index.numel())
    if n_in_ball == 0:
        raise ValueError(
            "no foreground voxel lies inside the ball; check the mask, center and ball_radius_mm"
        )
    if n_in_ball > max_points:
        gen = torch.Generator().manual_seed(seed)
        index = index[torch.randperm(n_in_ball, generator=gen)[:max_points]].sort().values
    residual = field[index] @ rigid_matrix.T
    return points[index], residual, n_in_ball


def register_session_pair(
    fixed: Any,
    moving: Any,
    *,
    ball_radius_mm: float,
    rigid_iterations: int = 300,
    demons_iterations: int = 300,
    demons_sigma: float = 2.0,
    max_points: int = 20000,
    seed: int = 0,
    metric: str = "mean_squares",
    histogram_match: bool = True,
    frame_tolerance_mm: float = 1.0,
) -> PairRegistration:
    """The full observation of one pair in RAS about the shared FOV centre."""
    require_sitk()
    center_lps = require_shared_frame(fixed, moving, tolerance_mm=frame_tolerance_mm)
    moving_m = match_intensities(moving, fixed) if histogram_match else moving
    transform = rigid_transform(fixed, moving_m, iterations=rigid_iterations, metric=metric)
    omega, translation, matrix = generator_from_rigid(transform, center_lps)
    displacement = demons_displacement(
        fixed, moving_m, transform, iterations=demons_iterations, smoothing_sigma=demons_sigma
    )
    mask = foreground_mask(fixed)
    points, residual, n_in_ball = sample_residual(
        displacement,
        matrix,
        mask=mask,
        ball_radius_mm=ball_radius_mm,
        center=center_lps,
        max_points=max_points,
        seed=seed,
    )
    out = PairRegistration(
        omega=vectors_to_ras(omega),
        translation=vectors_to_ras(translation),
        rigid_matrix=matrix_to_ras(matrix),
        points=vectors_to_ras(points),
        residual=vectors_to_ras(residual),
        center=vectors_to_ras(torch.as_tensor(center_lps, dtype=torch.float64)),
        n_foreground=int(mask.sum()),
        n_in_ball=n_in_ball,
    )
    norms = residual.norm(dim=1)
    centre = [round(float(c), 1) for c in out.center]
    stats = (out.n_foreground, n_in_ball, points.shape[0], float(norms.mean()))
    quantiles = torch.quantile(norms, torch.tensor([0.5, 0.95], dtype=norms.dtype)).tolist()
    logger.info(
        "pair registered about RAS centre %s: |omega|=%.4f rad, |t|=%.2f mm; foreground %d, "
        "in ball %d, kept %d; residual mean/median/p95 = %.2f/%.2f/%.2f mm",
        *(centre, float(omega.norm()), float(translation.norm()), *stats, *quantiles),
    )
    return out
