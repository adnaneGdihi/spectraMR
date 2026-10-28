"""Heteroscedastic Uncertainty-Aware Loss for MRI Reconstruction.

Implements the Kendall & Gal (2017) heteroscedastic formulation that learns
per-pixel aleatoric uncertainty during training. The network outputs both
a prediction ŷ and a log-variance map log(σ²), enabling calibrated spatial
uncertainty maps that flag regions where ULF signal is insufficient for
robust 3T synthesis.

Mathematical Formulation:
    L_UQ = (1/N) Σᵢ [ (ŷᵢ - y_3T,i)² / (2σᵢ²) + ½ log(σᵢ²) ]

The first term down-weights residuals in high-uncertainty regions (learned
attenuation). The second term (log-barrier) prevents trivial σ → ∞ collapse.

The formulation is derived from the negative log-likelihood of a Gaussian
with input-dependent variance:
    p(y|x) = N(y; f_θ(x), σ²_θ(x))

References:
    - Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning
      for Computer Vision?", NeurIPS 2017.
    - Nix & Weigend, "Estimating the mean and variance of the target probability
      distribution", IEEE ICNN 1994.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss(name="uncertainty", aliases=["UncertaintyAwareLoss"])
class UncertaintyAwareLoss(nn.Module):
    """Heteroscedastic uncertainty-aware loss (Kendall & Gal 2017).

    Learns per-pixel aleatoric uncertainty during training by jointly optimizing
    the reconstruction quality and a log-variance map. Supports both L1 and L2
    base losses, and provides a fallback path for backward compatibility when
    no uncertainty map is provided.

    Args:
        base_loss: Callable that computes element-wise loss (pred, target) → scalar.
            Used as fallback when uncertainty is None.
        uncertainty_weight: Multiplicative weight for the log-barrier regularizer
            (the ½ log(σ²) term). Default 1.0 for proper NLL; reduce for weaker
            uncertainty penalty.
        base_loss_type: Type of per-pixel loss for the heteroscedastic term.
            'l2' (default) uses squared error / (2σ²), matching Gaussian NLL.
            'l1' uses absolute error / σ, matching Laplacian NLL.
        min_log_var: Lower clamp for log-variance (numerical stability).
            Default -10.0 → σ_min ≈ 0.007.
        max_log_var: Upper clamp for log-variance (prevents collapse to trivial
            infinite uncertainty). Default 10.0 → σ_max ≈ 148.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{Uncertainty} = \\frac{1}{2} e^{-\\log(\\sigma^2)} (\\hat{y} - y)^2 + \\frac{1}{2} \\log(\\sigma^2)"""

    def __init__(
        self,
        base_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        uncertainty_weight: float = 1.0,
        base_loss_type: str = "l2",
        min_log_var: float = -10.0,
        max_log_var: float = 10.0,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else F.l1_loss
        self.uncertainty_weight = uncertainty_weight
        self.base_loss_type = base_loss_type.lower()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        if self.base_loss_type not in ("l1", "l2"):
            raise ValueError(f"base_loss_type must be 'l1' or 'l2', got '{base_loss_type}'")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute heteroscedastic uncertainty-aware loss.

        Args:
            pred: Model predictions [B, C, H, W] (image or k-space).
            target: Ground truth targets [B, C, H, W].
            uncertainty: Log-variance map log(σ²) [B, C, H, W] or [B, 1, H, W].
                If None, falls back to base_loss only (backward compat).

        Returns:
            Scalar loss value.

        forward method for UncertaintyAwareLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            uncertainty (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Fallback: no uncertainty map → use base loss directly
        if uncertainty is None:
            return self.base_loss(pred, target)

        # Clamp log-variance for numerical stability
        log_var = torch.clamp(uncertainty, min=self.min_log_var, max=self.max_log_var)

        # Broadcast log_var to match pred shape if needed (e.g., [B,1,H,W] → [B,C,H,W])
        if log_var.shape != pred.shape:
            log_var = log_var.expand_as(pred)

        if self.base_loss_type == "l2":
            # Gaussian NLL: (y - ŷ)² / (2σ²) + ½ log(σ²)
            # With log_var = log(σ²):
            #   precision = exp(-log_var) = 1/σ²
            #   loss = 0.5 * exp(-log_var) * (pred - target)² + 0.5 * log_var
            precision = torch.exp(-log_var)
            sq_error = (pred - target) ** 2
            nll = 0.5 * precision * sq_error + 0.5 * self.uncertainty_weight * log_var

        else:  # l1 → Laplacian NLL
            # Laplacian NLL: |y - ŷ| / σ + log(σ)
            # With log_var = log(σ²), log(σ) = 0.5 * log_var, σ = exp(0.5 * log_var)
            #   loss = exp(-0.5 * log_var) * |pred - target| + 0.5 * log_var
            inv_sigma = torch.exp(-0.5 * log_var)
            abs_error = torch.abs(pred - target)
            nll = inv_sigma * abs_error + 0.5 * self.uncertainty_weight * log_var

        return nll.mean()

    def extra_repr(self) -> str:
        return (
            f"base_loss_type={self.base_loss_type}, "
            f"uncertainty_weight={self.uncertainty_weight}, "
            f"log_var_range=[{self.min_log_var}, {self.max_log_var}]"
        )
