#!/usr/bin/env python3
"""Latent-Specific Loss Functions Module
=====================================

This module implements loss functions specifically designed for latent space
models including VAE, VQ-VAE, and latent diffusion models.
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle

from spectramr.models.losses.kl_divergence import KLDivergenceLoss
from spectramr.models.losses.registry import register_loss
from spectramr.models.losses.vq_losses import VQLoss  # Import canonical VQLoss

_LATENT_REGULARIZATION_TYPES = ("l2", "sparsity", "diversity")


@register_loss(name="latent_diffusion", aliases=["LatentDiffusionLoss"])
class LatentDiffusionLoss(nn.Module):
    """Loss function for latent diffusion models.

    This combines denoising matching loss with optional auxiliary losses
    for latent space regularization.

    Args:
        prediction_type: Type of prediction ('epsilon', 'x0', 'v')
        aux_weight: Weight for auxiliary loss
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{LatentDiffusion} = \\| \\epsilon_\\theta(z_t, t) - \\epsilon \\|_2^2"""

    def __init__(self, prediction_type: str = "epsilon", aux_weight: float = 0.0):
        """__init__.

        Args:
            prediction_type (str): Description.
            aux_weight (float): Description.
        """
        super().__init__()
        self.prediction_type = prediction_type
        self.aux_weight = aux_weight

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        aux_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute latent diffusion loss.

        Args:
            predicted: Model prediction
            target: Target value (noise, x0, or v depending on prediction_type)
            aux_loss: Optional auxiliary loss

        Returns:
            Tuple of (total_loss, denoising_loss)

        forward method for LatentDiffusionLoss.

        Executes PyTorch tensor operations.

        Args:
            predicted (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            aux_loss (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Denoising matching loss
        denoising_loss = F.mse_loss(predicted, target)

        # Total loss
        total_loss = denoising_loss
        if aux_loss is not None and self.aux_weight > 0:
            total_loss = total_loss + self.aux_weight * aux_loss

        return total_loss, denoising_loss


@register_loss(name="perceptual_latent", aliases=["PerceptualLatentLoss"])
class PerceptualLatentLoss(nn.Module):
    r"""Perceptual loss computed in latent space.

    This loss uses a pre-trained encoder to compute perceptual similarity
    in the latent space rather than pixel space.

    Args:
        encoder: Pre-trained encoder network
        layers: Layers to extract features from
        weights: Weights for each layer
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{PerceptualLatent} = \sum_l w_l \| \Phi_l(z_1) - \Phi_l(z_2) \|_2^2"""

    def __init__(
        self,
        encoder: nn.Module,
        layers: tuple[str, ...] = ("conv1", "conv2", "conv3"),
        weights: tuple[float, ...] | None = None,
    ):
        """__init__.

        Args:
            encoder (nn.Module): Description.
            layers (tuple[str, ...]): Description.
            weights (Optional[tuple[float, ...]]): Description.
        """
        super().__init__()
        self.encoder = encoder
        self.layers = layers

        if weights is None:
            weights = tuple(1.0 / len(layers) for _ in layers)
        if len(weights) != len(layers):
            raise ValueError(f"got {len(weights)} weights for {len(layers)} layers")
        self.weights = weights

        # Register hooks to extract features
        self.features: dict[str, torch.Tensor] = {}
        self.hooks: list[RemovableHandle] = []

        for layer_name in layers:
            layer = getattr(self.encoder, layer_name, None)
            if not isinstance(layer, nn.Module):
                raise ValueError(
                    f"PerceptualLatentLoss: encoder has no submodule {layer_name!r} to hook"
                )
            hook = layer.register_forward_hook(self._get_features_hook(layer_name))
            self.hooks.append(hook)

    def _get_features_hook(
        self, layer_name: str
    ) -> Callable[[nn.Module, tuple[Any, ...], torch.Tensor], None]:
        """Create hook function for feature extraction."""

        def hook(module: nn.Module, input: tuple[Any, ...], output: torch.Tensor) -> None:
            """Store ``output`` under ``layer_name``."""
            self.features[layer_name] = output

        return hook

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between two inputs.

        Args:
            input1: First input tensor
            input2: Second input tensor

        Returns:
            Perceptual loss

        forward method for PerceptualLatentLoss.

        Executes PyTorch tensor operations.

        Args:
            input1 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            input2 (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Clear features
        self.features.clear()

        # Forward pass for first input
        _ = self.encoder(input1)

        # Extract features
        features1 = {layer: self.features[layer] for layer in self.layers}

        # Clear features
        self.features.clear()

        # Forward pass for second input
        _ = self.encoder(input2)

        # Extract features
        features2 = {layer: self.features[layer] for layer in self.layers}

        # Stays a tensor so the term trains the encoder rather than only reporting a value.
        loss = input1.new_zeros(())
        for weight, layer in zip(self.weights, self.layers, strict=True):
            loss = loss + weight * F.mse_loss(features1[layer], features2[layer])

        return loss


@register_loss(name="latent_regularization", aliases=["LatentRegularizationLoss"])
class LatentRegularizationLoss(nn.Module):
    r"""Regularization loss for latent space.

    This loss encourages certain properties in the latent space such as
    smoothness, diversity, or specific distributions.

    Args:
        regularization_type: Type of regularization
            ('l2', 'sparsity', 'diversity')
        weight: Weight for regularization term
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{LatentReg} = \lambda \mathbb{E}[ \|z\|_2^2 ]"""

    def __init__(self, regularization_type: str = "l2", weight: float = 1e-4):
        """__init__.

        Args:
            regularization_type (str): Description.
            weight (float): Description.
        """
        super().__init__()
        if regularization_type not in _LATENT_REGULARIZATION_TYPES:
            raise ValueError(
                f"regularization_type must be one of {_LATENT_REGULARIZATION_TYPES}, "
                f"got {regularization_type!r}"
            )
        self.regularization_type = regularization_type
        self.weight = weight

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss.

        Args:
            latent: Latent representation [B, latent_dim, ...]

        Returns:
            Regularization loss

        forward method for LatentRegularizationLoss.

        Executes PyTorch tensor operations.

        Args:
            latent (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.regularization_type == "l2":
            # L2 regularization on latent codes
            return self.weight * torch.mean(latent.pow(2))

        if self.regularization_type == "sparsity":
            # L1 sparsity regularization
            return self.weight * torch.mean(torch.abs(latent))

        if self.regularization_type == "diversity":
            # Mean inverse pairwise distance over the B(B-1) off-diagonal pairs; an
            # off-diagonal weight replaces boolean-mask indexing, which syncs on CUDA.
            n = latent.size(0)
            if n < 2:
                raise ValueError("diversity regularization needs at least 2 latents per batch")
            latent_flat = latent.reshape(n, -1)
            distances = torch.cdist(latent_flat, latent_flat, p=2)
            off_diag = 1.0 - torch.eye(n, device=latent.device, dtype=distances.dtype)
            inv = off_diag / (distances + 1e-8)
            return self.weight * inv.sum() / (n * (n - 1))

        raise ValueError(f"unknown regularization_type {self.regularization_type!r}")


# Convenience functions for creating losses
def create_beta_kl_loss(
    beta: float = 1.0,
    reduction: str = "mean",
) -> KLDivergenceLoss:
    """Create beta-weighted KL divergence loss.

    Now uses KLDivergenceLoss from spectramr.models.losses.kl_divergence

    Args:
        beta: Weighting factor for KL divergence
        reduction: Reduction method - Ignored for compatibility as KLDivergenceLoss handles it differently,
                    but logic is similar. KLDivergenceLoss uses 'mean' by default.

    Returns:
        KLDivergenceLoss instance

    """
    return KLDivergenceLoss(beta=beta)


def create_annealed_kl_loss(
    beta_start: float = 0.0,
    beta_end: float = 1.0,
    annealing_steps: int = 10000,
) -> KLDivergenceLoss:
    """Create annealed KL divergence loss.

    Now uses KLDivergenceLoss from spectramr.models.losses.kl_divergence

    Args:
        beta_start: Initial beta value
        beta_end: Final beta value
        annealing_steps: Number of annealing steps

    Returns:
        KLDivergenceLoss instance

    """
    return KLDivergenceLoss(
        beta=1.0,  # The class handles annealing internally scaling this
        annealing_start=beta_start,
        annealing_end=beta_end,
        annealing_steps=annealing_steps,
    )


def create_vq_loss(
    commitment_weight: float = 0.25,
    codebook_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
) -> VQLoss:
    """Create VQ loss for VQ-VAE models.

    Args:
        commitment_weight: Weight for commitment loss
        codebook_weight: Weight for codebook loss
        reconstruction_weight: Weight for reconstruction loss

    Returns:
        VQLoss instance

    """
    return VQLoss(
        commitment_weight=commitment_weight,
        codebook_weight=codebook_weight,
        reconstruction_weight=reconstruction_weight,
    )


def create_latent_diffusion_loss(
    prediction_type: str = "epsilon",
    aux_weight: float = 0.0,
) -> LatentDiffusionLoss:
    """Create latent diffusion loss.

    Args:
        prediction_type: Type of prediction
        aux_weight: Weight for auxiliary loss

    Returns:
        LatentDiffusionLoss instance

    """
    return LatentDiffusionLoss(prediction_type=prediction_type, aux_weight=aux_weight)
