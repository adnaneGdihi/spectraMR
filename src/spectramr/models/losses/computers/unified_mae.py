"""Unified MAE Loss Computer for Masked Autoencoder Pretraining.

SSOT Pattern: All loss weighting and initialization centralized.
"""

import logging
from typing import Any

import torch

from spectramr.models.losses.computers.base import BaseLossComputer, LossOutput

logger = logging.getLogger(__name__)


class UnifiedMAELossComputer(BaseLossComputer):
    """Unified loss computer for MAE (Masked Autoencoder) pretraining.

    Implements the MAE reconstruction loss with optional L1 and MSE components.
    Follows SSOT pattern with centralized loss weighting.

    Paper: He et al. "Masked Autoencoders Are Scalable Vision Learners"
    https://arxiv.org/abs/2111.06377
    """

    def __init__(self, config: Any, device: torch.device | str = "cpu"):
        """Initialize Unified MAE Loss Computer.

        Args:
            config: Configuration object with loss weights and settings.
            device: Device to place loss tensors on.
        """
        super().__init__(config, torch.device(device))

    def _initialize_losses(self) -> None:
        """Initialize loss components from config.

        MAE uses:
        - Reconstruction loss (MSE or L1)
        - Optional perceptual loss
        """
        # MSE loss is the primary loss for MAE
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.l1_loss = torch.nn.L1Loss(reduction="mean")

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        iteration: int = 0,
        **kwargs: Any,
    ) -> LossOutput:
        """Standard compute method required by BaseLossComputer."""
        return self.compute_forward_loss(pred, target, **kwargs)

    def compute_forward_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> LossOutput:
        """Compute MAE reconstruction loss (forward pass only, no backward).

        Args:
            pred: Reconstructed image tensor [B, C, H, W]
            target: Target image tensor [B, C, H, W]
            **kwargs: Additional optional arguments (mask, etc.)

        Returns:
            LossOutput with total loss and component dict.
        """
        # Reconstruction loss (primary)
        mse_loss = self.mse_loss(pred, target)

        # Optional L1 regularization
        l1_loss = self.l1_loss(pred, target)

        # Weighted combination - use safe weight lookup
        mse_weight = 1.0
        l1_weight = 0.1

        try:
            if (
                hasattr(self.config, "losses")
                and self.config.losses
                and self.config.losses.reconstruction
            ):
                recon = self.config.losses.reconstruction
                mse_weight = recon.lambda_l2 if hasattr(recon, "lambda_l2") else 1.0
                l1_weight = recon.lambda_l1 if hasattr(recon, "lambda_l1") else 0.1
        except Exception as _exc:
            logger.debug("Suppressed exception: %s", _exc)

        weighted_mse = mse_loss * mse_weight
        weighted_l1 = l1_loss * l1_weight

        total_loss = weighted_mse + weighted_l1

        # Stack components
        components = {
            "reconstruction_loss": weighted_mse,
            "l1_loss": weighted_l1,
        }

        # Detached tensors, not ``float()``: two device syncs per step for a dict
        # no training-path caller reads.
        return LossOutput(
            total=total_loss,
            components=components,
            metrics={
                "reconstruction": mse_loss.detach(),
                "l1": l1_loss.detach(),
            },
        )

    def compute_validation_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> dict[str, float]:
        """Compute validation metrics for MAE reconstruction.

        Args:
            pred: Reconstructed image tensor
            target: Target image tensor
            device: Device to compute metrics on

        Returns:
            Dictionary of metric values (PSNR, SSIM, MAE, etc.)
        """
        from spectramr.core.metrics.evaluation import EvaluationMetrics

        eval_metrics = EvaluationMetrics(device=str(device), config=self.config)
        return eval_metrics.compute_all(pred, target)


__all__ = ["UnifiedMAELossComputer"]
