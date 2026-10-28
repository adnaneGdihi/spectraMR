"""Task 5: PMA-VarNet Global Objective Loss.

Implements the joint multi-contrast reconstruction loss anchoring to the digital marker.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss
from spectramr.models.losses.ssim_loss import SSIMLoss

logger = logging.getLogger(__name__)


@register_loss("pma_global", aliases=["PMAGlobalLoss"])
class PMAGlobalLoss(nn.Module):
    r"""Joint Multi-Contrast Objective Loss for PMA-VarNet.

    Optimizes the network over all contrasts (T1w, T2w, FLAIR) simultaneously.
    Combines:
    1. L1 global anatomy loss
    2. Structural Similarity (SSIM)
    3. L2 Marker Pinning Prior (strictly enforcing synthetic geometry)

    .. math::

        \mathcal{L} = \sum_{c=1}^{N_{con}} \left(
            \|\hat{\mathbf{x}}_c - \mathbf{x}_{c, gt}\|_1 +
            \lambda_1 (1 - \text{SSIM}) +
            \lambda_2 \|\mathbf{M}_{syn} \odot (\hat{\mathbf{x}}_c - \mathbf{m}_{syn, c})\|_2^2
        \right)
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_marker: float = 5.0,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_marker = lambda_marker
        self.ssim_loss = SSIMLoss()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        marker_mask: torch.Tensor | None = None,
        marker_prior: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute the joint multi-contrast loss.

        Args:
            prediction: Reconstructed images `[B, N_con, H, W]`
            target: Ground truth images `[B, N_con, H, W]`
            marker_mask: Binary mask `[1, 1, H, W]` or `[B, 1, H, W]`
            marker_prior: Ideal marker `[B, N_con, H, W]` (or broadcastable)

        Returns:
            Scalar objective loss.

        forward method for PMAGlobalLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch: {prediction.shape} != {target.shape}")

        if prediction.dim() != 4:
            raise ValueError(f"Expected [B, N_con, H, W]; got {tuple(prediction.shape)}")
        n_con = prediction.shape[1]

        # SSIM is computed on magnitudes. Its data range is the global target
        # maximum; SSIMLoss takes it as a Python float, which costs one host sync.
        pred_mag = prediction.abs() if prediction.is_complex() else prediction
        targ_mag = target.abs() if target.is_complex() else target
        ssim_data_range = float(targ_mag.max())

        # Every term is a per-contrast mean over equally sized channels, so the
        # sum over contrasts is n_con times the mean over all of them; SSIMLoss
        # filters channels independently and averages them per image.
        l1 = (prediction - target).abs().mean()
        l_ssim = self.ssim_loss(pred_mag, targ_mag, data_range=ssim_data_range)

        l_marker = prediction.new_zeros((), dtype=l1.dtype)
        if marker_mask is not None and marker_prior is not None:
            prior = marker_prior if marker_prior.shape[1] == 1 else marker_prior[:, :n_con]
            # ``mse_loss`` has no complex kernel, so |z|² is taken explicitly.
            l_marker = ((prediction - prior) * marker_mask).abs().pow(2).mean()

        per_contrast = (
            (self.lambda_l1 * l1) + (self.lambda_ssim * l_ssim) + (self.lambda_marker * l_marker)
        )
        total_loss: torch.Tensor = n_con * per_contrast
        return total_loss


__all__ = ["PMAGlobalLoss"]
