from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss
from spectramr.models.losses.validation import validate_loss_inputs


@register_loss(name="diffusion_mse", aliases=["DiffusionMSELoss"], domain="agnostic")
class DiffusionMSELoss(nn.Module):
    """Mean squared error loss for diffusion model predictions.

    **DOMAIN**: AGNOSTIC (works on any tensor)
    **Input**: [B, C, H, W] or any tensor shape
    **3D Support**: Yes

    Standard MSE for diffusion noise prediction. Domain-agnostic.
    """

    @validate_loss_inputs
    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute MSE loss.

        forward method for DiffusionMSELoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        return F.mse_loss(pred, target)


@register_loss(name="diffusion_velocity", aliases=["VelocityPredictionLoss"], domain="agnostic")
class VelocityPredictionLoss(nn.Module):
    """Velocity prediction loss for diffusion models.

    **DOMAIN**: AGNOSTIC (flow field loss)
    **Input**: [B, C, H, W] velocity field or any tensor
    **3D Support**: Yes

    Smooth L1 loss for flow field prediction. Domain-agnostic.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{VelocityPrediction} = \\text{SmoothL1}(v_{pred}, v_{target})"""

    @validate_loss_inputs
    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute velocity prediction loss.

        forward method for VelocityPredictionLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        return F.smooth_l1_loss(pred, target)


@register_loss(name="huber", aliases=["HuberLoss"], domain="agnostic")
class HuberLoss(nn.Module):
    """Huber loss for robust diffusion model training.

    **DOMAIN**: AGNOSTIC (works on any tensor)
    **Input**: [B, C, H, W] or any tensor shape
    **3D Support**: Yes

    Smooth L1 (Huber) loss for outlier-robust training. Domain-agnostic.
    """

    def __init__(self, delta: float = 1.0):
        """__init__.

        Args:
            delta (float): Description.
        """
        super().__init__()
        self.delta = delta

    @validate_loss_inputs
    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Compute Huber loss.

        forward method for HuberLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        return F.huber_loss(pred, target, delta=self.delta)
