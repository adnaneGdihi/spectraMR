#!/usr/bin/env python3
"""KL Divergence loss functions for VAE training.

This module implements KL divergence losses with various weighting
schemes for stable VAE training.
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.losses.registry import register_loss


@register_loss(name="kl", aliases=["kl_divergence", "KLDivergenceLoss"])
class KLDivergenceLoss(nn.Module):
    """KL Divergence loss for VAE training.

    **DOMAIN**: LATENT SPACE
    **Input**: mu [B, latent_dim], log_var [B, latent_dim]
    **3D Support**: No (flat embeddings)

    Computes KL divergence between learned latent distribution and standard
    normal prior, with optional annealing and capacity constraints (β-VAE).
    """

    annealing_weight: torch.Tensor

    def __init__(
        self,
        beta: float = 1.0,
        capacity: float | None = None,
        limit_capacity: bool = False,
        annealing_steps: int | None = None,
        annealing_start: float = 0.0,
        annealing_end: float = 1.0,
    ):
        """Initialize KL divergence loss.

        Args:
            beta: Weight for KL loss (β-VAE)
            capacity: Target capacity C (Burgess et al. 2018)
            limit_capacity: Whether to enforce capacity constraint |KL - C|
            annealing_steps: Number of steps for KL annealing
            annealing_start: Starting annealing weight
            annealing_end: Final annealing weight

        """
        super().__init__()

        self.beta = beta
        self.capacity = capacity
        self.limit_capacity = limit_capacity
        self.annealing_steps = annealing_steps
        self.annealing_start = annealing_start
        self.annealing_end = annealing_end

        self.current_step = 0
        self.register_buffer("annealing_weight", torch.tensor(annealing_start))

    def forward(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            mu: Mean of latent distribution [B, latent_dim]
            log_var: Log variance of latent distribution [B, latent_dim]

        Returns:
            KL divergence loss

        forward method for KLDivergenceLoss.

        Executes PyTorch tensor operations.

        Args:
            mu (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            log_var (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # [FIX] Clamp values to prevent numerical instability
        # log_var: prevent exp() overflow (exp(20) ≈ 5e8, exp(-20) ≈ 2e-9)
        log_var_clamped = torch.clamp(log_var, min=-20.0, max=20.0)
        # mu: prevent extreme mu^2 values
        mu_clamped = torch.clamp(mu, min=-50.0, max=50.0)

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(
            1 + log_var_clamped - mu_clamped.pow(2) - log_var_clamped.exp(), dim=1
        )
        kl_mean = torch.mean(kl_loss)

        if self.limit_capacity and self.capacity is not None:
            # Burgess et al. (2018): beta * |KL - C|
            final_loss = self.beta * torch.abs(kl_mean - self.capacity)
        else:
            # Standard β-VAE: beta * KL
            final_loss = self.beta * kl_mean

        # Apply annealing if enabled. The buffer multiplies as a 0-dim
        # tensor — no `.item()` GPU sync in the per-step loss path.
        if self.annealing_steps is not None:
            self._update_annealing_weight()
            final_loss = self.annealing_weight * final_loss

        return final_loss

    def _update_annealing_weight(self) -> None:
        """Update annealing weight based on current step."""
        if self.annealing_steps is not None and self.current_step < self.annealing_steps:
            progress = self.current_step / self.annealing_steps
            weight = self.annealing_start + progress * (self.annealing_end - self.annealing_start)
            self.annealing_weight.data.fill_(weight)
            self.current_step += 1
        elif self.annealing_steps is not None:
            self.annealing_weight.data.fill_(self.annealing_end)

    def reset_annealing(self) -> None:
        """Reset annealing to initial state."""
        self.current_step = 0
        self.annealing_weight.data.fill_(self.annealing_start)

    def get_current_weight(self) -> float:
        """Get current annealing weight."""
        return self.annealing_weight.item()


@register_loss(name="vq_kl", aliases=["VQKLLoss"])
class VQKLLoss(nn.Module):
    """KL divergence loss for VQ-VAE training.

    Combines VQ loss with optional KL regularization for
    improved latent space properties.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{VQKLLoss} = \\mathcal{L}_{VQ} + \\lambda_{KL} \\text{KL}(q(z) || p(z))"""

    def __init__(
        self,
        commitment_cost: float = 0.25,
        kl_weight: float = 0.1,
        use_kl: bool = True,
    ):
        """Initialize VQ-KL loss.

        Args:
            commitment_cost: Weight for VQ commitment loss
            kl_weight: Weight for KL regularization
            use_kl: Whether to include KL regularization

        """
        super().__init__()

        self.commitment_cost = commitment_cost
        self.kl_weight = kl_weight
        self.use_kl = use_kl

    def forward(
        self,
        quantized: torch.Tensor,
        z: torch.Tensor,
        mu: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute VQ-KL loss.

        Args:
            quantized: Quantized latent vectors
            z: Original latent vectors
            mu: Optional mean for KL regularization
            log_var: Optional log variance for KL regularization

        Returns:
            Combined VQ-KL loss

        forward method for VQKLLoss.

        Executes PyTorch tensor operations.

        Args:
            quantized (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mu (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            log_var (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # VQ loss (van den Oord et al. 2017): codebook term ||sg[z_e]-e||^2 updates
        # the codebook (stop-grad on encoder z), commitment term b*||z_e-sg[e]||^2
        # updates the encoder (stop-grad on codebook ``quantized``).
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        total_loss = vq_loss

        # Add KL regularization if provided
        if self.use_kl and mu is not None and log_var is not None:
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            kl_loss = self.kl_weight * torch.mean(kl_loss)
            total_loss = total_loss + kl_loss

        return total_loss


def kl_divergence_loss(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute KL divergence loss (functional version).

    Args:
        mu: Mean of latent distribution [B, latent_dim]
        log_var: Log variance of latent distribution [B, latent_dim]
        beta: Weight for KL loss

    Returns:
        KL divergence loss

    """
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return beta * torch.mean(kl_loss)


def vq_loss(
    quantized: torch.Tensor,
    z: torch.Tensor,
    commitment_cost: float = 0.25,
) -> torch.Tensor:
    """Compute VQ loss (functional version).

    Args:
        quantized: Quantized latent vectors
        z: Original latent vectors
        commitment_cost: Weight for commitment loss

    Returns:
        VQ loss

    """
    # Codebook term updates the codebook (stop-grad on encoder ``z``);
    # commitment term updates the encoder (stop-grad on ``quantized``).
    codebook_loss = F.mse_loss(quantized, z.detach())
    commitment_loss = F.mse_loss(quantized.detach(), z)
    return codebook_loss + commitment_cost * commitment_loss
