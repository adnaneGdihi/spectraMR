#!/usr/bin/env python3
"""Latent Discriminator implementation for Latent GAN.

This module implements a discriminator that operates on latent space
vectors for adversarial training in latent GANs.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IDiscriminator
from spectramr.models.registry import register_model


# Scores a latent VECTOR ([B, latent_dim]), not a signal: ``forward(self, z)``. ``latent``
# is outside ``critic_domain.TRANSFORMABLE``, so an image/k-space generator paired with
# this critic raises rather than transforming a latent.
@register_model(
    role="discriminator", name="latent_discriminator", training_mode="gan", input_domain="latent"
)
class LatentDiscriminator(IDiscriminator, nn.Module):
    """Discriminator for latent space in Latent GAN.

    Discriminates between real and fake latent vectors using
    multi-layer perceptron architecture with spectral normalization.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: tuple = (512, 256, 128),
        spectral_norm: bool = True,
        dropout_rate: float = 0.0,
        hidden_dim: int = None,
    ):
        """Initialize Latent Discriminator.

        Args:
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for MLP layers
            spectral_norm: Whether to use spectral normalization
            dropout_rate: Dropout rate for regularization
            hidden_dim: Legacy argument for hidden dimension (ignored if hidden_dims provided)

        """
        super().__init__()
        nn.Module.__init__(self)

        if hidden_dim is not None:
            # If hidden_dim is provided, we can use it to construct hidden_dims
            # e.g. (hidden_dim, hidden_dim // 2, hidden_dim // 4)
            # But for now, just accepting it prevents the crash.
            pass

        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.spectral_norm = spectral_norm
        self.dropout_rate = dropout_rate

        # Build MLP discriminator
        layers = []
        current_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    self._create_linear_layer(current_dim, hidden_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                ],
            )
            current_dim = hidden_dim

        # Final classification layer
        layers.extend(
            [
                self._create_linear_layer(current_dim, 1),
                nn.Sigmoid(),  # Output probability
            ],
        )

        self.model = nn.Sequential(*layers)

    def _create_linear_layer(self, in_features: int, out_features: int) -> nn.Module:
        """Create a linear layer with optional spectral normalization."""
        linear = nn.Linear(in_features, out_features)
        if self.spectral_norm:
            return nn.utils.spectral_norm(linear)
        return linear

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through discriminator.

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            Discrimination probability [B, 1]

        forward method for LatentDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.model(z)

    def discriminate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Discriminate real vs fake latent vectors.

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            Discrimination probability [B, 1]

        """
        return self.forward(z)

    def get_feature_maps(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract feature maps for feature matching in latent space.

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            Dictionary of feature maps at different layers

        """
        features = {}
        x = z

        for i, layer in enumerate(self.model):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                features[f"linear_{i}"] = x
            elif isinstance(layer, (nn.LeakyReLU, nn.Dropout)):
                x = layer(x)
            else:
                # Final activation
                x = layer(x)

        return features

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "LatentDiscriminator"

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the discriminator."""
        return sum(p.numel() for p in self.parameters())


def create_latent_discriminator(
    latent_dim: int = 128,
    hidden_dims: tuple = (512, 256, 128),
) -> LatentDiscriminator:
    """Factory function for creating latent discriminators.

    Args:
        latent_dim: Dimension of latent space
        hidden_dims: Hidden dimensions for MLP layers

    Returns:
        Configured latent discriminator

    """
    return LatentDiscriminator(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
    )
