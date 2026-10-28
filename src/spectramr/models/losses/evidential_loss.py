"""Evidential Deep Learning Loss for Epistemic Uncertainty.

Implements Pillar 9 of the ultra-low field synthesis paradigms. This loss trains
a network to output the parameters of a Normal-Inverse-Gamma (NIG) distribution:
(gamma, nu, alpha, beta) rather than a single deterministic scalar.

This enables the generator to quantify both aleatoric and epistemic uncertainty,
flagging regions where the 64mT signal is insufficient for robust 3T reconstruction.

References:
    Amini et al., "Deep Evidential Regression," NeurIPS 2020.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from spectramr.models.losses.registry import register_loss


@register_loss("evidential", aliases=["nig_loss", "evidential_regression"])
class EvidentialLoss(nn.Module):
    """Evidential Loss for Normal-Inverse-Gamma distribution matching.

    Minimizes the Negative Log-Likelihood of the NIG distribution and applies
    an evidence regularizer penalizing overconfidence in erroneous regions.

    Args:
        coeff (float): Regularization coefficient (lambda) for the evidence penalty.
            Balances the data fit against uncertainty calibration.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{Evidential} = \\mathbb{E}[\\text{NLL}] + \\lambda \\cdot \\text{KL}(\\text{NIG} || \\text{NIG}_0)"""

    def __init__(self, coeff: float = 1.0) -> None:
        super().__init__()
        self.coeff = coeff

    def nll_loss(
        self, y_true: Tensor, gamma: Tensor, nu: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        """Negative log-likelihood of the NIG distribution."""
        # \Gamma(alpha) / \Gamma(alpha + 0.5) simplified via log-gamma
        omega = 2 * beta * (1 + nu)

        nll = (
            0.5 * torch.log(torch.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log(omega + nu * (y_true - gamma) ** 2)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        return nll

    def nig_regularizer(
        self, y_true: Tensor, gamma: Tensor, nu: Tensor, alpha: Tensor, beta: Tensor
    ) -> Tensor:
        """Evidence regularizer penalizing high evidence on high error predictions."""
        # Penalty scales with absolute error |y - \gamma| multiplied by total evidence (2nu + alpha)
        # Note: True Amini paper reg is: |y - gamma| * (2 * nu + alpha)
        error = torch.abs(y_true - gamma)
        reg = error * (2 * nu + alpha)
        return reg

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        nig_params: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the evidential loss.

        Args:
            prediction: Stacked evidential parameters [B, 4, D, H, W] or [B, 4, H, W]
                where channel 0: gamma (prediction mean)
                channel 1: nu (virtual observation count for mean) > 0
                channel 2: alpha (virtual observation count for variance) > 1
                channel 3: beta (sum of squared errors) > 0
            target: Ground truth 3T volume [B, 1, D, H, W] or [B, 1, H, W]

        Returns:
            Scalar loss value balancing NLL and regularization.

        forward method for EvidentialLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # The strategy hands the structural losses gamma alone and the NIG tensor here.
        if nig_params is not None:
            prediction = nig_params
        if prediction.shape[1] != 4:
            raise ValueError(
                f"EvidentialLoss requires exactly 4 predicted channels (gamma, nu, alpha, beta), "
                f"but got {prediction.shape[1]} and no nig_params."
            )

        gamma = prediction[:, 0:1]
        nu = prediction[:, 1:2]
        alpha = prediction[:, 2:3]
        beta = prediction[:, 3:4]

        # Enforce strict positivity constraints (usually done in generator via softplus,
        # but we clip here for numerical stability just in case).
        nu = torch.clamp(nu, min=1e-4)
        alpha = torch.clamp(alpha, min=1.0001)  # Alpha must be > 1
        beta = torch.clamp(beta, min=1e-4)

        nll = self.nll_loss(target, gamma, nu, alpha, beta)
        reg = self.nig_regularizer(target, gamma, nu, alpha, beta)

        loss = torch.mean(nll + self.coeff * reg)
        return loss


__all__ = ["EvidentialLoss"]
