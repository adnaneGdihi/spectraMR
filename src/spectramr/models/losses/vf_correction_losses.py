"""Dual-Marker Correction Losses for Physics-Informed Training.

Losses that enforce geometric constraints derived from the synthetic dual-marker system:
- GNL geometric consistency
- EPI ghost suppression
- Super-resolution forward model consistency
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss

logger = logging.getLogger(__name__)


# ======================================================================
# 1. GNL Consistency Loss
# ======================================================================


@register_loss("gnl_consistency", aliases=["GNLConsistencyLoss"])
class GNLConsistencyLoss(nn.Module):
    r"""Penalise marker position error after gradient non-linearity unwarping.

    After applying the GNL deformation correction, the reconstructed
    marker positions should match the known synthetic prior positions.  Any
    residual error indicates an imperfect correction:

    .. math::

        \mathcal{L}_{\text{GNL}} = \lambda \cdot
            \| \mathbf{p}_{\text{corrected}} - \mathbf{p}_{\text{true}} \|_2^2

    Args:
        lambda_gnl: Weight for the GNL consistency term.
    """

    def __init__(self, lambda_gnl: float = 5.0) -> None:
        super().__init__()
        self.lambda_gnl = lambda_gnl

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        marker_mask: torch.Tensor | None = None,
        marker_prior: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute GNL consistency loss.

        Args:
            prediction: Corrected image ``[B, C, H, W]``.
            target: Ground truth ``[B, C, H, W]``.
            marker_mask: Binary mask of marker region ``[1, 1, H, W]``.
            marker_prior: Known marker intensities ``[1, 1, H, W]``.

        Returns:
            Scalar loss.

        forward method for GNLConsistencyLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if marker_mask is None or marker_prior is None:
            return F.l1_loss(prediction, target)

        pred_marker = prediction * marker_mask
        prior_expanded = (marker_prior * marker_mask).expand_as(pred_marker)
        return self.lambda_gnl * F.mse_loss(pred_marker, prior_expanded)


# ======================================================================
# 2. Ghost Suppression Loss
# ======================================================================


@register_loss("ghost_suppression", aliases=["GhostSuppressionLoss"])
class GhostSuppressionLoss(nn.Module):
    r"""Penalise N/2 ghost residual at the marker location.

    In EPI, the ghost appears at a FOV/2 shift from the true marker.
    After correction, no signal should remain at the ghost position:

    .. math::

        \mathcal{L}_{\text{ghost}} = \lambda \cdot
            \| I(\mathbf{r}_{\text{marker}} + \text{FOV}/2) \|_1

    Args:
        lambda_ghost: Weight for ghost suppression.
    """

    def __init__(self, lambda_ghost: float = 10.0) -> None:
        super().__init__()
        self.lambda_ghost = lambda_ghost

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        marker_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute ghost suppression loss.

        Args:
            prediction: Corrected image ``[B, C, H, W]``.
            target: Ground truth ``[B, C, H, W]``.
            marker_mask: Binary mask of marker ``[1, 1, H, W]``.

        Returns:
            Scalar loss.

        forward method for GhostSuppressionLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if marker_mask is None:
            return F.l1_loss(prediction, target)

        H = prediction.shape[-2]
        ghost_mask = torch.roll(marker_mask, H // 2, dims=-2)
        ghost_signal = (prediction * ghost_mask).abs()
        return self.lambda_ghost * ghost_signal.mean()


# ======================================================================
# 3. Super-Resolution Consistency Loss
# ======================================================================


@register_loss(
    "super_resolution_consistency",
    aliases=["SuperResolutionConsistencyLoss"],
)
class SuperResolutionConsistencyLoss(nn.Module):
    r"""Forward model consistency for multi-frame super-resolution.

    Ensures the reconstructed HR image, when shifted and downsampled
    according to the known sub-pixel translations, reproduces each
    observed LR frame:

    .. math::

        \mathcal{L}_{\text{SR}} = \frac{1}{N} \sum_{n=1}^{N}
            \| D \cdot W_n \cdot \hat{\mathbf{x}} - \mathbf{y}_n \|_1

    Args:
        upscale_factor: Super-resolution upscale factor.
        lambda_sr: Weight for SR consistency.
    """

    def __init__(
        self,
        upscale_factor: int = 2,
        lambda_sr: float = 1.0,
    ) -> None:
        super().__init__()
        self.upscale_factor = upscale_factor
        self.lambda_sr = lambda_sr

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        lr_frames: torch.Tensor | None = None,
        shifts: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Compute SR forward consistency loss.

        Args:
            prediction: HR reconstruction ``[1, C, H_hr, W_hr]``.
            target: Not used directly.
            lr_frames: ``[N, C, H_lr, W_lr]`` observed LR frames.
            shifts: ``[N, 2]`` sub-pixel shifts (dy, dx) in HR pixels.

        Returns:
            Scalar loss.

        forward method for SuperResolutionConsistencyLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if lr_frames is None or shifts is None:
            return F.l1_loss(prediction, target)

        num_frames = lr_frames.shape[0]
        s = self.upscale_factor
        total_loss = torch.tensor(0.0, device=prediction.device)
        height, width = prediction.shape[-2], prediction.shape[-1]
        grid_y = torch.linspace(-1, 1, height, device=prediction.device, dtype=prediction.dtype)
        grid_x = torch.linspace(-1, 1, width, device=prediction.device, dtype=prediction.dtype)
        base_y, base_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
        # With align_corners=True one HR pixel spans 2 / (size - 1) normalised units.
        step_y = 2.0 / max(height - 1, 1)
        step_x = 2.0 / max(width - 1, 1)

        for n in range(num_frames):
            gx = base_x + shifts[n, 1] * step_x
            gy = base_y + shifts[n, 0] * step_y
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

            shifted = F.grid_sample(
                prediction,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            simulated_lr = F.avg_pool2d(shifted, kernel_size=s, stride=s)
            total_loss = total_loss + F.l1_loss(simulated_lr, lr_frames[n : n + 1])

        return self.lambda_sr * total_loss / num_frames


__all__ = [
    "GNLConsistencyLoss",
    "GhostSuppressionLoss",
    "SuperResolutionConsistencyLoss",
]
