#!/usr/bin/env python3
"""VAE implementation for MRI super-resolution.

This module implements a complete Variational Autoencoder that combines
encoder and decoder with reparameterization trick and KL divergence loss.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IModel
from spectramr.models.registry import register_model

from .decoder import VAEDecoder
from .encoder import VAEEncoder


# ``forward`` returns ``(reconstruction, mu, log_var)``: the tensor a loss sees is the
# decoded image. ``output_domain="latent"`` (until 2026-09-03) failed the two arms
# that declare an image-domain loss on it, which is the only loss a VAE has.
@register_model(name="vae", training_mode="vae", spatial_dims=(2,), output_domain="image")
class VAE(nn.Module, IModel):
    """Variational Autoencoder for MRI super-resolution.

    Combines encoder and decoder with reparameterization trick
    and provides KL divergence loss computation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (64, 128, 256, 512),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        deterministic: bool = False,
    ):
        """Initialize VAE.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for encoder/decoder
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            deterministic: decode ``z = mu`` instead of a reparameterised sample, the
                deterministic-autoencoder comparator. Pair it with a zero KL weight;
                ``VAETrainingStrategy`` refuses it otherwise.

        """
        super().__init__()
        if not isinstance(deterministic, bool):
            raise TypeError(f"vae deterministic must be a bool, got {deterministic!r}.")
        self.deterministic = deterministic

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        # Create encoder and decoder
        self.encoder = VAEEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        self.decoder = VAEDecoder(
            out_channels=out_channels,
            latent_dim=latent_dim,
            hidden_dims=tuple(reversed(hidden_dims)),
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        # Weight initialization handled by centralized weight_initialization module
        pass

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of (reconstruction, mu, log_var)

        forward method for VAE.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode
        mu, log_var = self.encoder(x)

        # Reparameterize (the deterministic comparator decodes the mean)
        z = mu if self.deterministic else self.encoder.reparameterize(mu, log_var)

        # Decode
        reconstruction = self.decoder(z)
        if reconstruction.shape[-2:] != x.shape[-2:]:
            # The decoder expands a 2x2 seed by one stride-2 stage per hidden dim, so
            # it renders 2 * 2**len(hidden_dims) pixels a side whatever it was fed. An
            # arm whose patch does not match trips the strategy's shape guard one
            # step later with no hint of the cause; say it here (2026-09-03).
            raise ValueError(
                f"VAE decodes {tuple(reconstruction.shape[-2:])} for an input of "
                f"{tuple(x.shape[-2:])}: the decoder renders 2 * 2**len(hidden_dims) = "
                f"{2 * 2 ** len(self.hidden_dims)} pixels a side. Give the arm a matching "
                "data.patch_size or one hidden dim per halving of the patch."
            )

        return reconstruction, mu, log_var

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder.decode(z)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        return self.encoder.reparameterize(mu, log_var)

    def sample(
        self,
        num_samples: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample from latent space.

        Args:
            num_samples: Number of samples to generate
            device: Device to place samples on

        Returns:
            Generated samples [num_samples, out_channels, H, W]

        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Decode
        return self.decode(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input through VAE."""
        reconstruction, _, _ = self.forward(x)
        return reconstruction

    def compute_kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            mu: Mean of latent distribution [B, latent_dim]
            log_var: Log variance of latent distribution [B, latent_dim]

        Returns:
            KL divergence loss

        """
        # KL divergence: 0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return torch.mean(kl_loss)

    def get_latent_distribution(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get latent distribution parameters."""
        return self.encode(x)

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "VAE"


def create_vae(
    in_channels: int = 1,
    out_channels: int = 1,
    latent_dim: int = 128,
    hidden_dims: tuple[int, ...] = (64, 128, 256, 512),
) -> VAE:
    """Factory function for creating VAEs.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        latent_dim: Dimension of latent space
        hidden_dims: Hidden dimensions for encoder/decoder

    Returns:
        Configured VAE

    """
    return VAE(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
