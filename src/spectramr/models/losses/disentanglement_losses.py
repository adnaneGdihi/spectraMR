"""Disentanglement Losses
======================

Loss functions for Experiment 1.2: Disentangled Representation Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss(name="modality_swap", aliases=["ModalitySwapLoss"])
class ModalitySwapLoss(nn.Module):
    r"""Loss for enforcing disentanglement via modality swapping.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{ModalitySwap} = \lambda_{recon} \|\hat{y} - y\|_1 + \lambda_{swap} \|\hat{y}_{swap} - y_{swap}\|_1"""

    def __init__(self, lambda_swap: float = 1.0, lambda_recon: float = 1.0):
        """__init__.

        Args:
            lambda_swap (float): Description.
            lambda_recon (float): Description.
        """
        super().__init__()
        self.lambda_swap = lambda_swap
        self.lambda_recon = lambda_recon

    def forward(
        self,
        recon_img: torch.Tensor,
        target_img: torch.Tensor,
        swap_img: torch.Tensor | None = None,
        swap_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss.

        Args:
            recon_img: Self-reconstructed image G(E_A(x), E_M(x))
            target_img: Ground truth for reconstruction (x)
            swap_img: Cross-synthesized image G(E_A(x), E_M(y))
            swap_target: Ground truth for swap (y) - wait, research says:
                         "Feed generator anatomy of T1 but style of T2...
                          The generated I_cross must match the ground truth I_T2"

                         Wait, if we feed Anatomy(T1) and Style(T2), the result should look like T2?
                         NO. Anatomy(T1) + Style(T2) = T1 anatomy with T2 contrast.
                         So it should look like T2-weighted image of Subject 1.

                         If we have paired data (T1, T2 of same subject), then:
                         Anatomy(T1) == Anatomy(T2) (ideally)
                         Style(T2) == T2 style
                         So G(E_A(T1), E_M(T2)) should approximate T2 image of that subject.

                         So swap_target IS the T2 image of the SAME subject.

        Returns:
            Dictionary of losses.

        forward method for ModalitySwapLoss.

        Executes PyTorch tensor operations.

        Args:
            recon_img (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_img (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            swap_img (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            swap_target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        losses = {}

        # 1. Self-Reconstruction Loss
        losses["loss_recon"] = self.lambda_recon * F.l1_loss(recon_img, target_img)

        # 2. Swap Loss (if paired data available)
        if swap_img is not None and swap_target is not None:
            losses["loss_swap"] = self.lambda_swap * F.l1_loss(swap_img, swap_target)

        return losses
