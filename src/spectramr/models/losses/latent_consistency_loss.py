"""Latent Consistency Loss for enforcing content/style disentanglement.

This loss ensures that the content code (anatomy) remains unchanged after
style transfer by re-encoding the swapped output and comparing against
the original content code.
"""

from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


class _DisentangledModel(Protocol):
    """The three submodules the cross-swap reads off the model."""

    content_encoder: Callable[[torch.Tensor], torch.Tensor]
    style_encoder: Callable[[torch.Tensor], torch.Tensor]
    generator: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@register_loss("latent_consistency", aliases=["LatentConsistencyLoss", "latent_swap"])
class LatentConsistencyLoss(nn.Module):
    """Latent consistency loss for disentanglement.

    Enforces that content code z_c remains invariant to style swapping:
        1. Extract z_c1 = E_c(x_a), z_s2 = E_s(x_b)
        2. Generate x_swap = G(z_c1, z_s2)
        3. Re-encode z_c_swap = E_c(x_swap)
        4. Loss: ||z_c1 - z_c_swap||²

    This forces the model to preserve anatomical structure (content)
    regardless of applied style (contrast/field strength).

    Args:
        reduction: Loss reduction method ('mean', 'sum') (default: 'mean')
        norm_type: Norm for distance computation (1, 2) (default: 2)

    Forward Args:
        model: Disentangled model with content_encoder, style_encoder, generator
        x_a: Source image A [B, C, H, W]
        x_b: Source image B [B, C, H, W]

    Returns:
        Latent consistency loss

    Example:
        >>> loss_fn = LatentConsistencyLoss()
        >>> loss = loss_fn(model, img_ulf, img_hf)
    """

    def __init__(self, reduction: str = "mean", norm_type: int = 2):
        """__init__.

        Args:
            reduction (str): Description.
            norm_type (int): Description.
        """
        super().__init__()
        self.reduction = reduction
        self.norm_type = norm_type

    def forward(
        self, model: _DisentangledModel, x_a: torch.Tensor, x_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute latent consistency loss via cross-swapping.

        Args:
            model: Disentangled model with encoders and generator
            x_a: Image A (e.g., ULF) [B, C, H, W]
            x_b: Image B (e.g., HF) [B, C, H, W]

        Returns:
            Scalar loss value

        forward method for LatentConsistencyLoss.

        Executes PyTorch tensor operations.

        Args:
            model (nn.Module): Expected input tensor.
            x_a (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_b (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Extract original content codes
        with torch.no_grad():
            z_c_a = model.content_encoder(x_a)  # Content from A
            z_c_b = model.content_encoder(x_b)  # Content from B

        # Extract style codes (detach to prevent style gradient interference)
        z_s_a = model.style_encoder(x_a).detach()  # Style from A
        z_s_b = model.style_encoder(x_b).detach()  # Style from B

        # Cross-swap: Content A + Style B
        x_swap_ab = model.generator(z_c_a.detach(), z_s_b)

        # Cross-swap: Content B + Style A
        x_swap_ba = model.generator(z_c_b.detach(), z_s_a)

        # Re-encode swapped images to get new content codes
        z_c_swap_ab = model.content_encoder(x_swap_ab)
        z_c_swap_ba = model.content_encoder(x_swap_ba)

        # Consistency loss: Re-encoded content should match original
        if self.norm_type == 1:
            loss_ab = F.l1_loss(z_c_swap_ab, z_c_a, reduction=self.reduction)
            loss_ba = F.l1_loss(z_c_swap_ba, z_c_b, reduction=self.reduction)
        elif self.norm_type == 2:
            loss_ab = F.mse_loss(z_c_swap_ab, z_c_a, reduction=self.reduction)
            loss_ba = F.mse_loss(z_c_swap_ba, z_c_b, reduction=self.reduction)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")

        # Average loss across both swap directions
        total_loss = (loss_ab + loss_ba) / 2.0

        return total_loss
