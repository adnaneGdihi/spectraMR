"""Deep supervision loss for improved gradient flow.

This module provides the DeepSupervisionLoss class that computes loss on
intermediate layer outputs to improve gradient flow during training.

Formerly located in composite.py, extracted during Phase 5 cleanup.
"""

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss(name="deep_supervision", aliases=["DeepSupervisionLoss"])
class DeepSupervisionLoss(nn.Module):
    r"""Computes loss on intermediate layer outputs to improve gradient flow.

    This loss encourages intermediate representations in the network to be
    closer to the target, which can help with gradient flow in deep networks
    and prevent vanishing gradients.

    Args:
        base_loss_fn: Base loss function to apply at each level (default: L1Loss)
        weight: Weight multiplier for the deep supervision loss (default: 0.1)

    Example:
        >>> deep_sup = DeepSupervisionLoss(base_loss_fn=nn.L1Loss(), weight=0.2)
        >>> intermediate = [layer1_out, layer2_out, layer3_out]  # From decoder
        >>> target = ground_truth_image
        >>> loss = deep_sup(intermediate, target)
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{DeepSupervision} = w \sum_{l} \mathcal{L}_{base}(\hat{y}^{(l)}, y)"""

    def __init__(self, base_loss_fn: nn.Module | None = None, weight: float = 0.1) -> None:
        """__init__.

        Args:
            base_loss_fn (nn.Module): Description.
            weight (float): Description.
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn if base_loss_fn is not None else nn.L1Loss()
        self.weight = weight

    def forward(
        self,
        intermediate_outputs: list[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deep supervision loss.

        Args:
            intermediate_outputs: List of intermediate predictions from network layers
                                 Each tensor should have shape [B, C, H, W]
            target: Ground truth target with shape [B, C, H, W]

        Returns:
            Total deep supervision loss (scalar tensor)

        forward method for DeepSupervisionLoss.

        Executes PyTorch tensor operations.

        Args:
            intermediate_outputs (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if not intermediate_outputs:
            return torch.tensor(0.0, device=target.device, dtype=target.dtype)

        total_loss = torch.tensor(0.0, device=target.device, dtype=target.dtype)

        for output in intermediate_outputs:
            # Resize target to match output spatial dimensions if needed
            if output.shape[-2:] != target.shape[-2:]:
                target_resized = nn.functional.interpolate(
                    target,
                    size=output.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            else:
                target_resized = target

            # Compute loss at this level. Accumulate the tensor directly — a
            # ``.item()`` here would detach every level from the graph and make
            # the supervision signal contribute zero gradient.
            level_loss = self.base_loss_fn(output, target_resized)
            total_loss = total_loss + level_loss

        # Apply weight multiplier
        return total_loss * self.weight


__all__ = ["DeepSupervisionLoss"]
