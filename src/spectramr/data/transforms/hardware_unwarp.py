r"""Undo the scanner's hardware warp from a calibration artifact.

The artifact of :mod:`spectramr.data.hardware_warp_artifacts` names a displacement field
:math:`v` in the scanner frame through its coefficients on
``HarmonicGradientBasis(ball_points(basis_points_per_axis, ball_radius_mm), max_degree,
min_degree)``. By the artifact's contract :math:`v` is the *observed-to-true* displacement: the
voxel observed at :math:`x` holds tissue whose true position is :math:`x + v(x)`. The corrected
image is therefore indexed by true position, and its voxel at :math:`r` reads the observed image
at the :math:`x` solving :math:`x + v(x) = r` -- the exact inverse of the warp, found by six
fixed-point iterations :math:`x \leftarrow r - v(x)`, which converge geometrically at rate
:math:`\|\nabla v\|` (a few per cent for a warp of millimetres over a decimetre), leaving a
sub-micron residual. Reading at
:math:`r + v(r)` instead would apply the warp a second time; the sign is pinned by the
end-to-end test of the registration module and by the convention test beside this one.

Coordinate convention. TorchIO stores an image as ``(C, I, J, K)`` with a 4x4 RAS affine
``A`` mapping voxel indices to world millimetres, ``r = A[:3, :3] (i, j, k) + A[:3, 3]``; the
artifact's coefficients are expressed in that same RAS frame relative to ``center_mm`` (the
registration converts its LPS measurements at its boundary), so the only alignment applied is
the shift by ``center_mm`` before the basis is evaluated. The sampling offset
``x - r`` (mm) is what the produced ``hardware_warp_displacement`` image stores, so a consumer
can read what was applied; in voxels it is ``A[:3, :3]^{-1} (x - r)``, read through
``grid_sample`` with ``align_corners=True`` so that a zero field returns the input
voxel-for-voxel. With ``apply_jacobian`` the sample is multiplied by
:math:`1 / \det(I + \nabla v(x))`, the volume-element ratio of :math:`r \mapsto x(r)`, which
makes the pull-back mass-preserving for a density image.

The field is evaluated in float64 and in chunks of voxels, so the memory the polynomial
evaluation needs is bounded by ``chunk_size`` rather than by the volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchio as tio

from spectramr.data.hardware_warp_artifacts import load_artifact
from spectramr.data.transforms.registry import register_transform
from spectramr.infrastructure.physics.hardware_warp_commutator import (
    HarmonicGradientBasis,
    ball_points,
    evaluate_polynomial_hessians,
)

__all__ = ["HardwareUnwarp"]


@register_transform("hardware_unwarp", produces=("hardware_warp_displacement",))
class HardwareUnwarp(tio.Transform):
    """Resample ``image_key`` by the inverse of the calibrated hardware warp.

    Args:
        artifact_path: The calibration artifact. Read and validated at construction, so a wrong
            path fails when the pipeline is built rather than at the first batch.
        image_key: Subject key of the image to unwarp. Default ``"input"``.
        apply_jacobian: Multiply by :math:`\\det(I + \\nabla v)` (mass-preserving pull-back).
        displacement_key: Subject key under which the displacement (mm, 3 channels, the image's
            affine) is stored, so a consumer can read what was applied.
        chunk_size: Voxels per polynomial-evaluation chunk.
    """

    def __init__(
        self,
        artifact_path: str | Path,
        image_key: str = "input",
        apply_jacobian: bool = False,
        displacement_key: str = "hardware_warp_displacement",
        chunk_size: int = 65536,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        artifact = load_artifact(artifact_path)
        self.artifact_path = str(artifact_path)
        self.unit_id = artifact["unit_id"]
        self.software_version = artifact["software_version"]
        self.basis = HarmonicGradientBasis(
            ball_points(artifact["basis_points_per_axis"], artifact["ball_radius_mm"]),
            artifact["max_degree"],
            artifact["min_degree"],
        )
        self.coeffs = torch.tensor(artifact["coeffs"], dtype=torch.float64)
        self.center = torch.tensor(artifact["center_mm"], dtype=torch.float64)
        self.image_key = image_key
        self.apply_jacobian = bool(apply_jacobian)
        self.displacement_key = displacement_key
        self.chunk_size = int(chunk_size)
        self.args_names = ("artifact_path", "image_key", "apply_jacobian", "displacement_key")

    INVERSE_ITERATIONS = 6

    def observed_position(self, true: torch.Tensor) -> torch.Tensor:
        """The ``x`` with ``x + v(x) = true``, by fixed-point iteration from ``x = true``."""
        observed = true
        for _ in range(self.INVERSE_ITERATIONS):
            observed = true - self.basis.synthesize(self.coeffs, observed)
        return observed

    def displacement_mm(
        self, shape: tuple[int, int, int], affine: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sampling offset ``x - r`` (mm) at every voxel, ``[I, J, K, 3]``, and the Jacobian factor.

        ``r`` is the voxel's world position (true frame) relative to ``center_mm``, ``x`` the
        observed position it reads from; the factor is ``1 / det(I + grad v(x))`` when
        ``apply_jacobian`` is set.
        """
        n_i, n_j, n_k = shape
        n = n_i * n_j * n_k
        linear, offset = affine[:3, :3], affine[:3, 3]
        field = torch.empty(n, 3, dtype=torch.float64)
        det = torch.empty(n, dtype=torch.float64) if self.apply_jacobian else None
        for start in range(0, n, self.chunk_size):
            flat = torch.arange(start, min(start + self.chunk_size, n))
            idx = torch.stack([flat // (n_j * n_k), (flat // n_k) % n_j, flat % n_k], dim=-1).to(
                torch.float64
            )
            world = idx @ linear.T + offset - self.center
            observed = self.observed_position(world)
            field[start : start + flat.numel()] = observed - world
            if det is not None:
                det[start : start + flat.numel()] = 1.0 / self._jacobian_det(observed)
        return field.reshape(n_i, n_j, n_k, 3), None if det is None else det.reshape(shape)

    def _jacobian_det(self, points: torch.Tensor) -> torch.Tensor:
        grad = torch.zeros(points.shape[0], 3, 3, dtype=torch.float64)
        for degree, (exps, poly) in self.basis.blocks.items():
            hess = evaluate_polynomial_hessians(poly, exps, points)
            grad = grad + torch.einsum(
                "b,bnij->nij", self.coeffs[self.basis.block_index(degree)], hess
            )
        return torch.linalg.det(torch.eye(3, dtype=torch.float64) + grad)

    @staticmethod
    def _pull_back(
        data: torch.Tensor, offset_vox: torch.Tensor, det: torch.Tensor | None
    ) -> torch.Tensor:
        """Sample ``data (C, I, J, K)`` at each voxel plus ``offset_vox [I, J, K, 3]``."""
        is_complex = torch.is_complex(data)
        real = torch.view_as_real(data).permute(0, 4, 1, 2, 3).flatten(0, 1) if is_complex else data
        n_i, n_j, n_k = real.shape[1:]
        idx = torch.stack(
            torch.meshgrid(torch.arange(n_i), torch.arange(n_j), torch.arange(n_k), indexing="ij"),
            dim=-1,
        ).to(torch.float64)
        target = idx + offset_vox
        extent = torch.tensor([n_i - 1, n_j - 1, n_k - 1], dtype=torch.float64)
        normalized = 2.0 * target / extent - 1.0
        # grid_sample reads its last axis as (x, y, z) against input dims (W, H, D) = (K, J, I).
        grid = normalized.flip(-1).unsqueeze(0).to(real.dtype)
        out = F.grid_sample(
            real.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0]
        if det is not None:
            out = out * det.to(out.dtype).unsqueeze(0)
        if is_complex:
            out = torch.view_as_complex(
                out.reshape(data.shape[0], 2, n_i, n_j, n_k).permute(0, 2, 3, 4, 1).contiguous()
            )
        return out

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.image_key not in subject:
            raise KeyError(
                f"hardware_unwarp: subject has no {self.image_key!r} image; keys are "
                f"{sorted(subject.keys())}"
            )
        image = subject[self.image_key]
        data = image.data
        if data.ndim != 4 or any(n < 2 for n in data.shape[1:]):
            raise ValueError(
                f"hardware_unwarp is a 3-D resampling and needs (C, I, J, K) with every spatial "
                f"extent >= 2, got {tuple(data.shape)}; apply it before slicing"
            )
        affine = torch.as_tensor(image.affine, dtype=torch.float64)
        field_mm, det = self.displacement_mm(tuple(data.shape[1:]), affine)
        offset_vox = field_mm @ torch.linalg.inv(affine[:3, :3]).T
        image.set_data(self._pull_back(data, offset_vox, det))
        real_dtype = data.real.dtype if torch.is_complex(data) else data.dtype
        subject.add_image(
            tio.ScalarImage(
                tensor=field_mm.permute(3, 0, 1, 2).to(real_dtype), affine=image.affine
            ),
            self.displacement_key,
        )
        return subject
