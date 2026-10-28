"""Algebraic Null-Space Erasure — Orthogonal Marker Removal.

Projects corrupted images onto the orthogonal complement (null-space)
of the marker's known spatial basis, annihilating marker energy without
altering the linearly independent anatomical signal.

Mathematics:
    Given marker spatial basis :math:`M \\in \\mathbb{C}^{N \\times K}` and
    corrupted measurement :math:`Y = X_{anat} + M_{corrupted}`:

    .. math::

        P^{\\perp} = I - M (M^H M + \\epsilon I)^{-1} M^H

        X_{clean} = P^{\\perp} Y

    The Tikhonov term :math:`\\epsilon I` regularises the pseudo-inverse
    for numerical stability.

Reference:
    Standard linear algebra — Moore-Penrose orthogonal projection.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class NullSpaceMarkerEraser(nn.Module):
    """Remove a known marker from corrupted images via null-space projection.

    The marker basis can be a single template or a set of K basis vectors
    (e.g., the marker at different deformation states).

    Args:
        marker_basis: Marker spatial basis ``[1, C, H, W]`` or
            ``[K, C, H, W]`` for multi-basis erasure.
        epsilon: Tikhonov regularisation for pseudo-inverse stability.
    """

    def __init__(
        self,
        marker_basis: Tensor,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon

        # Flatten spatial dims: [K, C*H*W]
        if marker_basis.dim() == 4:
            K = marker_basis.shape[0]
            self.register_buffer("M", marker_basis.reshape(K, -1))  # [K, N]
        else:
            raise ValueError(f"marker_basis must be 4-D [K, C, H, W], got {marker_basis.dim()}-D")

        self._spatial_shape: tuple[int, ...] = tuple(marker_basis.shape[1:])

        logger.info(
            "[NullSpaceMarkerEraser] K=%d basis vectors, N=%d pixels, eps=%.1e",
            self.M.shape[0],
            self.M.shape[1],
            epsilon,
        )

    def forward(self, corrupted_image: Tensor) -> Tensor:
        """Project corrupted image onto the marker's null-space.

        Args:
            corrupted_image: ``[B, C, H, W]`` — complex or real tensor.

        Returns:
            Cleaned image with marker energy removed, same shape.

        forward method for NullSpaceMarkerEraser.

        Executes PyTorch tensor operations.

        Args:
            corrupted_image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = corrupted_image.shape[0]
        spatial = self._spatial_shape  # (C, H, W)
        N = self.M.shape[1]

        # Flatten to [B, N]
        Y = corrupted_image.reshape(B, N)

        # Basis vectors as columns A = M^T [N, K]; P = A (A^H A + eps I)^-1 A^H. For a real
        # basis this equals the earlier row-space form; for a complex one the conjugates
        # make it a true orthogonal projector (idempotent).
        A = self.M.t().to(Y.dtype)  # [N, K]
        a_h = A.conj().t()  # [K, N]
        gram = a_h @ A  # [K, K]
        gram_reg = gram + self.epsilon * torch.eye(
            gram.shape[0], device=gram.device, dtype=gram.dtype
        )
        coeffs = torch.linalg.solve_ex(gram_reg, a_h @ Y.t())[0]  # _ex: no info-check host sync
        y_clean = Y - (A @ coeffs).t()  # [B, N]

        return y_clean.reshape(B, *spatial)


__all__ = ["NullSpaceMarkerEraser"]
