import logging
from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.metrics_aware_loss import MetricsAwareLossMixin
from spectramr.models.losses.registry import register_loss

logger = logging.getLogger(__name__)

# Optional import for LPIPS
try:
    import lpips as _lpips_available  # type: ignore[import-untyped]
except ImportError:
    _lpips_available = None


@register_loss(name="lpips", aliases=["LPIPSLoss"], domain="image")
class LPIPSLoss(MetricsAwareLossMixin, nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) loss with optional metrics.

    **DOMAIN**: IMAGE-SPACE ONLY
    **Input**: [B, 3, H, W] RGB (auto-converts grayscale to RGB)
    **3D Support**: No (2D patch-based)

    Pre-trained AlexNet backbone. Do NOT use with k-space [B, 2, H, W].
    Input auto-converted to RGB if necessary.

    The LPIPS network is a frozen metric: its parameters never take gradients
    and it stays in eval mode whatever mode the owning module is put in, since
    ``lpips.LPIPS`` carries dropout on its linear heads.
    """

    def __init__(self, net: str = "alex", verbose: bool = False, compute_metrics: bool = False):
        """__init__.

        Args:
            net (str): Description.
            verbose (bool): Description.
            compute_metrics (bool): Description.
        """
        super().__init__()
        if _lpips_available is None:
            raise ImportError("lpips package is not installed. Please install it to use LPIPSLoss.")
        self.loss_fn = _lpips_available.LPIPS(net=net, verbose=verbose)
        self.loss_fn.requires_grad_(False)
        self.loss_fn.eval()
        self.compute_metrics_flag = compute_metrics

    def train(self, mode: bool = True) -> "LPIPSLoss":
        """Set this module's mode while keeping the LPIPS network in eval mode."""
        super().train(mode)
        self.loss_fn.eval()
        return self

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute LPIPS perceptual distance.

        forward method for LPIPSLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        loss_value: torch.Tensor = self.loss_fn(pred, target).mean()
        if self.compute_metrics_flag:
            metrics = self.compute_metrics(pred, target, loss_value)
            return loss_value, metrics
        return loss_value

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_value: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute LPIPS-specific metrics."""
        # Ensure inputs are in [-1, 1] range for LPIPS
        pred_clipped = torch.clamp(pred, -1, 1)
        target_clipped = torch.clamp(target, -1, 1)

        return {
            "loss": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "lpips": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "pred_mean": pred.mean().item(),
            "pred_std": pred.std().item(),
            "target_mean": target.mean().item(),
            "target_std": target.std().item(),
            "pred_range": (pred.max().item() - pred.min().item()),
            "target_range": (target.max().item() - target.min().item()),
            "pred_clipped_ratio": (
                (pred_clipped != pred).float().mean().item()
            ),  # Ratio of values outside [-1, 1]
            "target_clipped_ratio": ((target_clipped != target).float().mean().item()),
        }


def create_lpips_loss(net: str = "alex", verbose: bool = False) -> nn.Module:
    """Build :class:`LPIPSLoss`; raises ImportError when ``lpips`` is not installed.

    An L1 stand-in would train a different objective under the LPIPS name
    (non-negotiable 3), so a missing package is reported, never substituted.
    """
    return LPIPSLoss(net=net, verbose=verbose)
