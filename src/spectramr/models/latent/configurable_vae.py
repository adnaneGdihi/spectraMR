"""Configurable VAE using Flexible Latent Architecture
====================================================

Example VAE implementation using the new configurable system.
"""

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.latent.configurable_architecture import (
    DecoderConfig,
    EncoderConfig,
    FlexibleDecoder,
    FlexibleEncoder,
    LatentConfig,
)
from spectramr.models.registry import register_model


@register_model(name="configurable_vae", training_mode="vae")
class ConfigurableVAE(IGenerator, nn.Module):
    """Variational Autoencoder with fully configurable architecture.

    Example usage:
        >>> # Create a simple VAE
        >>> vae = ConfigurableVAE.from_presets("simple")

        >>> # Create a ResNet VAE with attention
        >>> vae = ConfigurableVAE.from_presets("resnet_attention")

        >>> # Custom configuration
        >>> encoder_cfg = EncoderConfig(
        ...     arch_type="resnet",
        ...     hidden_dims=[64, 128, 256, 512],
        ...     use_attention=True,
        ... )
        >>> decoder_cfg = DecoderConfig(
        ...     arch_type="resnet",
        ...     hidden_dims=[512, 256, 128, 64],
        ...     use_attention=True,
        ... )
        >>> latent_cfg = LatentConfig(latent_dim=512)
        >>> vae = ConfigurableVAE(encoder_cfg, decoder_cfg, latent_cfg)
    """

    def __init__(
        self,
        encoder_config: EncoderConfig | None = None,
        decoder_config: DecoderConfig | None = None,
        latent_config: LatentConfig | None = None,
        beta: float = 1.0,
    ):
        """__init__.

        Args:
            encoder_config (Optional[EncoderConfig]): Description.
            decoder_config (Optional[DecoderConfig]): Description.
            latent_config (Optional[LatentConfig]): Description.
            beta (float): Description.
        """
        super().__init__()

        # Use defaults if not provided
        self.encoder_config = encoder_config or EncoderConfig()
        self.decoder_config = decoder_config or DecoderConfig()
        self.latent_config = latent_config or LatentConfig()
        self.beta = beta

        # ConfigurableVAE.forward/compute_loss hard-assume a Gaussian latent
        # (it reads ``mu``/``logvar`` and applies the reparameterization +
        # KL terms). Only the gaussian distribution is end-to-end supported;
        # for {vq, categorical, flow} the encoder returns a different dict
        # and the VAE objective does not apply, so fail fast at construction
        # with a clear message rather than a bare KeyError at forward time.
        if self.latent_config.distribution != "gaussian":
            raise NotImplementedError(
                "ConfigurableVAE only supports distribution='gaussian'; got "
                f"{self.latent_config.distribution!r}. The forward/compute_loss "
                "path requires 'mu'/'logvar' latent parameters."
            )

        # Build encoder and decoder
        self.encoder = FlexibleEncoder(self.encoder_config, self.latent_config)
        self.decoder = FlexibleDecoder(self.decoder_config, self.latent_config)

    @classmethod
    def from_presets(cls, preset: str, **kwargs) -> "ConfigurableVAE":
        """Create VAE from preset configurations.

        Args:
            preset: One of "simple", "resnet", "attention", "resnet_attention", "deep"
            **kwargs: Override specific parameters

        Returns:
            Configured VAE
        """
        presets = {
            "simple": {
                "encoder_config": EncoderConfig(
                    arch_type="conv",
                    hidden_dims=[32, 64, 128, 256],
                    use_attention=False,
                    use_residual=False,
                ),
                "decoder_config": DecoderConfig(
                    arch_type="conv",
                    hidden_dims=[256, 128, 64, 32],
                    use_attention=False,
                    use_residual=False,
                ),
                "latent_config": LatentConfig(latent_dim=256),
            },
            "resnet": {
                "encoder_config": EncoderConfig(
                    arch_type="resnet",
                    hidden_dims=[64, 128, 256, 512],
                    use_attention=False,
                    use_residual=True,
                ),
                "decoder_config": DecoderConfig(
                    arch_type="resnet",
                    hidden_dims=[512, 256, 128, 64],
                    use_attention=False,
                    use_residual=True,
                ),
                "latent_config": LatentConfig(latent_dim=512),
            },
            "attention": {
                "encoder_config": EncoderConfig(
                    arch_type="hybrid",
                    hidden_dims=[64, 128, 256, 512],
                    use_attention=True,
                    attention_layers=[2, 3],
                    num_heads=8,
                ),
                "decoder_config": DecoderConfig(
                    arch_type="hybrid",
                    hidden_dims=[512, 256, 128, 64],
                    use_attention=True,
                    attention_layers=[0, 1],
                    num_heads=8,
                ),
                "latent_config": LatentConfig(latent_dim=512),
            },
            "resnet_attention": {
                "encoder_config": EncoderConfig(
                    arch_type="resnet",
                    hidden_dims=[64, 128, 256, 512],
                    use_attention=True,
                    attention_layers=[3],
                    use_residual=True,
                    num_heads=16,
                ),
                "decoder_config": DecoderConfig(
                    arch_type="resnet",
                    hidden_dims=[512, 256, 128, 64],
                    use_attention=True,
                    attention_layers=[0],
                    use_residual=True,
                    num_heads=16,
                ),
                "latent_config": LatentConfig(latent_dim=512),
            },
            "deep": {
                "encoder_config": EncoderConfig(
                    arch_type="resnet",
                    hidden_dims=[32, 64, 128, 256, 512, 1024],
                    use_attention=True,
                    attention_layers=[4, 5],
                    use_residual=True,
                    dropout_rate=0.2,
                ),
                "decoder_config": DecoderConfig(
                    arch_type="resnet",
                    hidden_dims=[1024, 512, 256, 128, 64, 32],
                    use_attention=True,
                    attention_layers=[0, 1],
                    use_residual=True,
                    dropout_rate=0.2,
                ),
                "latent_config": LatentConfig(latent_dim=1024),
            },
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

        config = presets[preset]
        config.update(kwargs)  # Allow overrides

        return cls(**config)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "ConfigurableVAE"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode input to latent distribution."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass through VAE.

        Args:
            x: Input tensor [B, C, H, W]
            return_latent: Whether to return latent parameters

        Returns:
            If return_latent=False: (reconstruction,)
            If return_latent=True: (reconstruction, mu, logvar, z)
        """
        # Encode
        latent_dict = self.encode(x)
        mu = latent_dict["mu"]
        logvar = latent_dict["logvar"]

        # Sample
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decode(z)

        if return_latent:
            return recon, mu, logvar, z
        return (recon,)

    def generate(self, z: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Generate samples from latent space.

        Args:
            z: Latent vector. If None, sample from N(0, I)
            **kwargs: Additional arguments

        Returns:
            Generated samples
        """
        if z is None:
            batch_size = kwargs.get("batch_size", 1)
            device = next(self.parameters()).device
            z = torch.randn(batch_size, self.latent_config.latent_dim, device=device)

        return self.decode(z)

    def compute_loss(
        self,
        x: torch.Tensor,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Input tensor
            reduction: Loss reduction method

        Returns:
            Dictionary with loss components
        """
        recon, mu, logvar, z = self.forward(x, return_latent=True)

        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, x, reduction=reduction)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if reduction == "mean":
            kl_loss = kl_loss / x.size(0)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "beta_kl": self.beta * kl_loss,
        }


def create_configurable_vae(
    preset: str = "simple",
    in_channels: int = 1,
    out_channels: int = 1,
    latent_dim: int = 256,
    **kwargs,
) -> ConfigurableVAE:
    """Factory function to create configurable VAE.

    Args:
        preset: Architecture preset
        in_channels: Input channels
        out_channels: Output channels
        latent_dim: Latent dimensionality
        **kwargs: Additional configuration

    Returns:
        Configured VAE model
    """
    vae = ConfigurableVAE.from_presets(preset, **kwargs)

    # Override dimensions if specified
    vae.encoder_config.in_channels = in_channels
    vae.decoder_config.out_channels = out_channels
    vae.latent_config.latent_dim = latent_dim

    # Rebuild with new config
    vae.encoder = FlexibleEncoder(vae.encoder_config, vae.latent_config)
    vae.decoder = FlexibleDecoder(vae.decoder_config, vae.latent_config)

    return vae


__all__ = ["ConfigurableVAE", "create_configurable_vae"]
