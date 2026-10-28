#!/usr/bin/env python3
"""VQ Loss functions for VQ-VAE and VQGAN training.

This module implements loss functions specific to vector quantization
and adversarial training for VQ-based models.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.losses.elementary import resolve_elementary_loss
from spectramr.models.losses.registry import register_loss


@register_loss(name="vq", aliases=["VQLoss", "latent_vq", "LatentVQLoss"])
class VQLoss(nn.Module):
    """Unified Vector Quantization loss supporting multiple VQ-VAE variants.

    **DOMAIN**: LATENT SPACE
    **Input**: Quantized embeddings, target embeddings [B, latent_dim]
    **3D Support**: No (codebook is flat)

    Combines codebook loss, commitment loss, and optional reconstruction/perceptual losses.
    Supports both simple API (quantized + targets) and full latent space API
    (reconstructed, target, quantized, encodings, embedded).
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{VQ} = \\mathcal{L}_{recon} + \\| sg[z] - z_q \\|_2^2 + \\beta \\| z - sg[z_q] \\|_2^2"""

    def __init__(
        self,
        commitment_cost: float | None = None,
        commitment_weight: float | None = None,
        codebook_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        perceptual_loss_fn: nn.Module | None = None,
    ):
        """Initialize unified VQ loss.

        Args:
            commitment_cost: Weight for commitment loss (alias: commitment_weight).
                Defaults to 0.25 if both not provided.
            commitment_weight: Weight for commitment loss (preferred parameter name).
                Defaults to 0.25 if both not provided.
            codebook_weight: Weight for codebook reconstruction loss. Default: 1.0
            reconstruction_weight: Weight for reconstruction loss (full latent API).
                Default: 1.0
            perceptual_weight: Weight for perceptual loss. Default: 0.0
            perceptual_loss_fn: Perceptual loss function. Default: None

        """
        super().__init__()

        # Handle commitment_cost vs commitment_weight naming
        if commitment_weight is not None:
            self.commitment_weight = commitment_weight
        elif commitment_cost is not None:
            self.commitment_weight = commitment_cost
        else:
            self.commitment_weight = 0.25

        self.codebook_weight = codebook_weight
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss_fn = perceptual_loss_fn

    def forward(
        self,
        *args: torch.Tensor,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, dict[str, float]]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Compute VQ loss.

        Supports two APIs:
        1. Simple API: forward(quantized, targets, inputs=None)
           Returns: (total_loss, loss_dict)
        2. Latent space API: forward(reconstructed, target, quantized, encodings, embedded)
           Returns: (total_loss, reconstruction_loss, commitment_loss, codebook_loss)

        Args:
            *args: Variable arguments (see above)
            **kwargs: Keyword arguments

        Returns:
            Tuple based on API used

        forward method for VQLoss.

        Executes PyTorch tensor operations.

        Args:
            None

        Returns:
            tuple[torch.Tensor, dict] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # API detection: check number of positional arguments
        if len(args) in (2, 3):
            # Simple API: (quantized, targets) or (quantized, targets, inputs)
            quantized = args[0]
            targets = args[1]
            inputs = args[2] if len(args) > 2 else kwargs.get("inputs")
            return self._forward_simple(quantized, targets, inputs)
        elif len(args) >= 5:
            # Latent space API: (reconstructed, target, quantized, encodings, embedded)
            reconstructed = args[0]
            target = args[1]
            quantized = args[2]
            encodings = args[3]
            embedded = args[4]
            return self._forward_latent(reconstructed, target, quantized, encodings, embedded)
        else:
            raise ValueError(
                f"VQLoss forward() expects either 2-3 args (simple API) "
                f"or 5 args (latent space API), got {len(args)}"
            )

    def _forward_simple(
        self,
        quantized: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Simple API: compute VQ loss from quantized embeddings.

        Args:
            quantized: Quantized latent vectors
            targets: Target latent vectors (before quantization)
            inputs: Optional original inputs for perceptual loss

        Returns:
            Tuple of (total_loss, loss_dict)

        """
        # VQ-VAE (van den Oord et al. 2017) gradient routing.
        # ``quantized`` = codebook vectors e, ``targets`` = encoder output z_e.
        #   codebook term   ||sg[z_e] - e||^2  -> updates the codebook e (weight 1.0)
        #   commitment term b*||z_e - sg[e]||^2 -> updates the encoder z_e (weight b)
        # The stop-gradient must sit on the OPPOSITE side from the term's name,
        # otherwise codebook_weight/commitment_weight land on the wrong tensors.
        codebook_loss = F.mse_loss(quantized, targets.detach())

        # Commitment loss
        commitment_loss = F.mse_loss(quantized.detach(), targets)

        # Total VQ loss
        vq_loss = self.codebook_weight * codebook_loss + self.commitment_weight * commitment_loss

        total_loss = vq_loss
        loss_dict = {
            "vq_loss": vq_loss.item(),
            "codebook_loss": codebook_loss.item(),
            "commitment_loss": commitment_loss.item(),
        }

        # Add perceptual loss if provided
        if (
            self.perceptual_weight > 0
            and self.perceptual_loss_fn is not None
            and inputs is not None
        ):
            perceptual_loss = self.perceptual_loss_fn(inputs, quantized)
            total_loss = total_loss + self.perceptual_weight * perceptual_loss
            loss_dict["perceptual_loss"] = perceptual_loss.item()

        return total_loss, loss_dict

    def _forward_latent(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        quantized: torch.Tensor,
        encodings: torch.Tensor,
        embedded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Latent space API: compute VQ loss with reconstruction from full latent space.

        This API is used by latent space models (VQ-VAE, etc.) that need
        explicit control over reconstruction, quantized, and embedding terms.

        Args:
            reconstructed: Reconstructed output [B, C, H, W]
            target: Target output [B, C, H, W]
            quantized: Quantized latent codes [B, latent_dim, H', W']
            encodings: One-hot encodings [B, num_embeddings, H', W']
            embedded: Embedded codes from codebook [B, latent_dim, H', W']

        Returns:
            Tuple of (total_loss, reconstruction_loss, commitment_loss, codebook_loss)

        """
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, target)

        # Commitment loss (encourage encoder to commit to codebook entries)
        commitment_loss = F.mse_loss(quantized.detach(), embedded)

        # Codebook loss (encourage codebook to match encoder outputs)
        codebook_loss = F.mse_loss(quantized, embedded.detach())

        # Total loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.commitment_weight * commitment_loss
            + self.codebook_weight * codebook_loss
        )

        return total_loss, reconstruction_loss, commitment_loss, codebook_loss


@register_loss(name="vqgan", aliases=["VQGANLoss"])
class VQGANLoss(nn.Module):
    """VQGAN loss combining VQ loss with adversarial and reconstruction losses.

    Implements the full VQGAN training objective with discriminator losses.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{VQGAN} = \\mathcal{L}_{recon} + \\| sg[z] - z_q \\|_2^2 + \\beta \\| z - sg[z_q] \\|_2^2 + \\lambda_{adv} \\mathcal{L}_{GAN}"""

    def __init__(
        self,
        commitment_cost: float = 0.25,
        codebook_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        adversarial_weight: float = 0.1,
        reconstruction_weight: float = 1.0,
        perceptual_loss_fn: nn.Module | None = None,
        discriminator: nn.Module | None = None,
    ):
        """Initialize VQGAN loss.

        Args:
            commitment_cost: Weight for VQ commitment loss
            codebook_weight: Weight for codebook reconstruction loss
            perceptual_weight: Weight for perceptual loss
            adversarial_weight: Weight for adversarial loss
            reconstruction_weight: Weight for pixel reconstruction loss
            perceptual_loss_fn: Perceptual loss function
            discriminator: Discriminator network for adversarial loss

        """
        super().__init__()

        self.commitment_cost = commitment_cost
        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_loss_fn = perceptual_loss_fn
        self.discriminator = discriminator

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        reconstructions: torch.Tensor,
        targets: torch.Tensor,
        quantized: torch.Tensor,
        latent_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute VQGAN loss.

        Args:
            reconstructions: Reconstructed images
            targets: Target images
            quantized: Quantized latent vectors
            latent_targets: Target latent vectors

        Returns:
            Tuple of (total_loss, loss_dict)

        forward method for VQGANLoss.

        Executes PyTorch tensor operations.

        Args:
            reconstructions (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            targets (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            quantized (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            latent_targets (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, dict]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        loss_dict: dict[str, float] = {}

        # Reconstruction loss (L1/L2)
        recon_loss = F.l1_loss(reconstructions, targets)
        total_loss = self.reconstruction_weight * recon_loss
        loss_dict["reconstruction_loss"] = recon_loss.item()

        # Same routing as VQLoss: the codebook term moves the codebook vectors
        # ``quantized``, the commitment term moves the encoder ``latent_targets``.
        codebook_loss = F.mse_loss(quantized, latent_targets.detach())
        commitment_loss = F.mse_loss(quantized.detach(), latent_targets)
        vq_loss = self.codebook_weight * codebook_loss + self.commitment_cost * commitment_loss
        total_loss = total_loss + vq_loss
        loss_dict.update(
            {
                "vq_loss": vq_loss.item(),
                "codebook_loss": codebook_loss.item(),
                "commitment_loss": commitment_loss.item(),
            },
        )

        # Perceptual loss
        if self.perceptual_weight > 0 and self.perceptual_loss_fn is not None:
            perceptual_loss = self.perceptual_loss_fn(reconstructions, targets)
            total_loss = total_loss + self.perceptual_weight * perceptual_loss
            loss_dict["perceptual_loss"] = perceptual_loss.item()

        # Adversarial loss (if discriminator provided)
        if self.adversarial_weight > 0 and self.discriminator is not None:
            # Generator loss
            disc_fake = self.discriminator(reconstructions)
            gen_loss = self.bce_loss(disc_fake, torch.ones_like(disc_fake))
            total_loss = total_loss + self.adversarial_weight * gen_loss
            loss_dict["generator_loss"] = gen_loss.item()

        return total_loss, loss_dict

    def compute_discriminator_loss(
        self,
        reconstructions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator loss for adversarial training.

        Args:
            reconstructions: Generated images
            targets: Real target images

        Returns:
            Discriminator loss

        """
        if self.discriminator is None:
            return torch.tensor(0.0, device=reconstructions.device)

        # Real images
        disc_real = self.discriminator(targets)
        real_loss = self.bce_loss(disc_real, torch.ones_like(disc_real))

        # Fake images
        disc_fake = self.discriminator(reconstructions.detach())
        fake_loss = self.bce_loss(disc_fake, torch.zeros_like(disc_fake))

        total: torch.Tensor = real_loss + fake_loss
        return total


def vq_reconstruction_loss(
    reconstructions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Compute reconstruction loss for VQ models.

    Args:
        reconstructions: Reconstructed images
        targets: Target images
        loss_type: Type of reconstruction loss ('l1', 'l2', 'huber')

    Returns:
        Reconstruction loss

    """
    loss: torch.Tensor = resolve_elementary_loss(loss_type)(reconstructions, targets)
    return loss


def codebook_perplexity(encodings: torch.Tensor) -> torch.Tensor:
    """Compute codebook perplexity from encodings.

    Args:
        encodings: One-hot encodings [B, N, H, W] or [B*N*H*W, num_embeddings]

    Returns:
        Perplexity value

    """
    if encodings.dim() == 4:
        # Convert from [B, N, H, W] to [B*N*H*W, N]
        encodings = encodings.permute(0, 2, 3, 1)
        encodings = encodings.reshape(-1, encodings.shape[-1])

    # Compute average probabilities
    avg_probs = torch.mean(encodings, dim=0)

    # Compute perplexity
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

    return perplexity
