"""Synthetic repeat sessions of one head through a planted hardware warp.

One owner for the volume-level fixture the Phase-3 tests share. A smooth analytic anatomy
``A(y0)`` (soft ellipsoid, Gaussian texture, a ventricle) is observed through the scanner map
``Psi: true -> observed``, ``x = y + v(y)`` with ``v`` a planted harmonic-gradient field, in a
head pose ``g_s``: the session image is ``I_s(x) = A(g_s^-1(Psi^-1(x)))``. The planted ``v`` is
the *true-to-observed* displacement, so the calibrated field, which the theory defines as
observed-to-true, must come out as its negative; the sign test in the registration suite relies
on that and the unwarp transform's convention test on the same fixture.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from spectramr.infrastructure.physics.hardware_warp_commutator import (
    HarmonicGradientBasis,
    ball_points,
)

__all__ = ["SyntheticSessions", "rotation_matrix"]


def rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula; the inverse of the module's ``_axis_angle``."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    k = rotvec / angle
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


class SyntheticSessions:
    """Head volumes at chosen poses through a planted warp, as SimpleITK images or arrays."""

    def __init__(
        self,
        *,
        n_voxels: int = 36,
        fov_mm: float = 192.0,
        ball_radius_mm: float = 100.0,
        max_degree: int = 3,
        min_degree: int = 2,
        warp_mm_at_70: float = 3.0,
        seed: int = 0,
    ) -> None:
        self.n = n_voxels
        self.spacing = fov_mm / n_voxels
        self.origin = -(n_voxels - 1) / 2 * self.spacing
        self.ball_radius_mm = ball_radius_mm
        idx = np.arange(n_voxels) * self.spacing + self.origin
        grid = np.stack(np.meshgrid(idx, idx, idx, indexing="ij"), -1).reshape(-1, 3)
        self.grid = grid
        self.canonical = HarmonicGradientBasis(
            ball_points(9, ball_radius_mm), max_degree, min_degree
        )
        coeffs = torch.zeros(self.canonical.n_basis, dtype=torch.float64)
        coeffs[self.canonical.block_index(2)] = torch.tensor(
            [1.2, -0.8, 0.6, 0.9, -0.5], dtype=torch.float64
        )
        probe = torch.tensor([[0.7 * ball_radius_mm, 0.0, 0.0]], dtype=torch.float64)
        coeffs *= warp_mm_at_70 / float(self.canonical.synthesize(coeffs, probe).norm())
        self.coeffs_true_to_observed = coeffs
        rng = np.random.default_rng(seed)
        self._blobs = rng.uniform(-45, 45, size=(12, 3))
        self._amps = rng.uniform(0.3, 1.0, 12)
        self._sigmas = rng.uniform(8, 16, 12)

    def v(self, points: np.ndarray) -> np.ndarray:
        return self.canonical.synthesize(
            self.coeffs_true_to_observed, torch.as_tensor(points, dtype=torch.float64)
        ).numpy()

    def psi(self, true: np.ndarray) -> np.ndarray:
        """true -> observed."""
        return true + self.v(true)

    def psi_inverse(self, observed: np.ndarray, iterations: int = 8) -> np.ndarray:
        """observed -> true by fixed-point iteration (the warp is a few mm, so it converges fast)."""
        true = np.array(observed, dtype=np.float64)
        for _ in range(iterations):
            true = observed - self.v(true)
        return true

    def anatomy(self, y0: np.ndarray) -> np.ndarray:
        q = (y0[:, 0] / 70) ** 2 + (y0[:, 1] / 85) ** 2 + (y0[:, 2] / 65) ** 2
        head = 1 / (1 + np.exp((q - 1) * 12))
        texture = 1 + sum(
            a * np.exp(-((y0 - b) ** 2).sum(1) / (2 * s * s))
            for a, b, s in zip(self._amps, self._blobs, self._sigmas, strict=True)
        )
        ventricle = 0.7 * np.exp(
            -(((y0[:, 0]) / 12) ** 2 + ((y0[:, 1] - 5) / 25) ** 2 + ((y0[:, 2]) / 10) ** 2) / 2
        )
        return head * texture * (1 - ventricle)

    def session_array(self, rotvec: Any, translation: Any) -> np.ndarray:
        """``I_s`` on the grid, ``(x, y, z)`` index order."""
        rot = rotation_matrix(np.asarray(rotvec, dtype=np.float64))
        true = self.psi_inverse(self.grid)
        head_frame = (true - np.asarray(translation, dtype=np.float64)) @ rot
        return self.anatomy(head_frame).reshape(self.n, self.n, self.n)

    def unwarped_array(self, rotvec: Any, translation: Any) -> np.ndarray:
        """What the session would look like with no hardware warp, ``A(g_s^-1(x))``."""
        rot = rotation_matrix(np.asarray(rotvec, dtype=np.float64))
        head_frame = (self.grid - np.asarray(translation, dtype=np.float64)) @ rot
        return self.anatomy(head_frame).reshape(self.n, self.n, self.n)

    def sitk_image(self, array_xyz: np.ndarray):
        import SimpleITK as sitk  # noqa: N813

        image = sitk.GetImageFromArray(
            np.ascontiguousarray(array_xyz.transpose(2, 1, 0)).astype(np.float32)
        )
        image.SetSpacing((self.spacing,) * 3)
        image.SetOrigin((self.origin,) * 3)
        return image

    def affine(self) -> np.ndarray:
        """Voxel -> world 4x4 for the array's ``(x, y, z)`` index order."""
        out = np.eye(4)
        out[:3, :3] *= self.spacing
        out[:3, 3] = self.origin
        return out

    def true_generator(self, pose_a: tuple, pose_b: tuple) -> tuple[np.ndarray, np.ndarray]:
        """``(omega, t)`` of ``h = g_b g_a^-1`` in the true frame."""
        ra, ta = rotation_matrix(pose_a[0]), np.asarray(pose_a[1], dtype=np.float64)
        rb, tb = rotation_matrix(pose_b[0]), np.asarray(pose_b[1], dtype=np.float64)
        rh = rb @ ra.T
        angle = float(np.arccos(np.clip((np.trace(rh) - 1) / 2, -1, 1)))
        axis = np.array([rh[2, 1] - rh[1, 2], rh[0, 2] - rh[2, 0], rh[1, 0] - rh[0, 1]])
        omega = np.zeros(3) if np.linalg.norm(axis) < 1e-12 else axis / np.linalg.norm(axis) * angle
        return omega, tb - rh @ ta
