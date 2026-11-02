"""KAN VAE Generator Implementation
===============================

This module implements a VAE with Kolmogorov-Arnold Networks (KAN)
for advanced latent space modeling in medical image super-resolution.

Key Features:
- KAN-based encoder and decoder architectures
- Probabilistic latent space with reparameterization trick
- Configurable latent dimensions and KL divergence weighting
- Support for medical imaging constraints ([-1, 1] output range)
- Integration with the IGenerator interface for factory compatibility

Architecture:
- Encoder: Convolutional layers → KAN layers → μ and σ prediction
- Decoder: KAN layers → Transposed convolutions → Output reconstruction
- Reparameterization: z = μ + σ * ε, ε ~ N(0, I)
- Loss: Reconstruction loss + KL divergence regularization

Usage:
    from spectramr.models.generators.kan_vae_generator import (
        KANVAEGenerator
    )

    model = KANVAEGenerator(
        in_channels=1,
        out_channels=1,
        latent_dim=128,
        kl_weight=0.001
    )
"""

import math

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network Layer

    Implements learnable activation functions using B-spline basis functions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degree: int = 3,
        grid_size: int = 5,
    ):
        """__init__.

        Args:
            input_dim (int): Description.
            output_dim (int): Description.
            degree (int): Description.
            grid_size (int): Description.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.grid_size = grid_size

        # Learnable B-spline coefficients
        self.coefficients = nn.Parameter(
            torch.randn(output_dim, input_dim, grid_size + degree + 1),
        )

        # Grid points for B-spline basis
        self.register_buffer(
            "grid",
            torch.linspace(0, 1, grid_size + 1).repeat(input_dim, 1),
        )

    def b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis functions.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            B-spline basis values

        """
        x.shape[0]
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)

        # Normalize input to [0, 1]
        x_min = self.grid[:, 0:1]
        x_max = self.grid[:, -1:]
        x_normalized = (x - x_min) / (x_max - x_min + 1e-8)

        # Compute B-spline basis
        basis = []
        for i in range(self.grid_size + self.degree + 1):
            if i < self.degree + 1:
                # Left boundary
                basis_val = torch.where(
                    x_normalized < self.grid[:, i : i + 1],
                    torch.ones_like(x_normalized),
                    torch.zeros_like(x_normalized),
                )
            elif i > self.grid_size:
                # Right boundary
                basis_val = torch.where(
                    x_normalized >= self.grid[:, i - self.degree - 1 : i - self.degree],
                    torch.ones_like(x_normalized),
                    torch.zeros_like(x_normalized),
                )
            else:
                # Interior
                basis_val = torch.zeros_like(x_normalized)
                for j in range(self.degree + 1):
                    if i - j >= 0 and i - j <= self.grid_size:
                        coeff = math.comb(self.degree, j) * (-1) ** j
                        grid_val = self.grid[:, i - j : i - j + 1]
                        basis_val += (
                            coeff
                            * torch.maximum(
                                torch.zeros_like(x_normalized),
                                x_normalized - grid_val,
                            )
                            ** self.degree
                        )

            basis.append(basis_val)

        return torch.cat(basis, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through KAN layer.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)

        forward method for KANLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        basis = self.b_spline_basis(x)  # (batch_size, input_dim, num_basis)
        # Apply coefficients: (output_dim, input_dim, num_basis) * (batch_size,
        # input_dim, num_basis)
        output = torch.einsum("oik,bik->bo", self.coefficients, basis)
        return output


class KANEncoder(nn.Module):
    """KAN-based VAE Encoder

    Maps input images to latent distribution parameters (μ, logσ).
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (32, 64, 128),
        kan_hidden_dim: int = 64,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int): Description.
            hidden_dims (tuple[int, ...]): Description.
            kan_hidden_dim (int): Description.
        """
        super().__init__()

        # Convolutional encoder
        self.encoder_layers = nn.ModuleList()
        current_channels = in_channels

        for hidden_dim in hidden_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_channels,
                        hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                ),
            )
            current_channels = hidden_dim

        # Calculate flattened dimension
        self.flattened_dim = 8192

        # KAN layers for feature processing
        self.kan_encoder = nn.Sequential(
            nn.Linear(self.flattened_dim, kan_hidden_dim),
            KANLayer(kan_hidden_dim, kan_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Latent distribution parameters
        self.fc_mu = nn.Linear(kan_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(kan_hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            Tuple of (mu, logvar) tensors, each of shape (batch_size, latent_dim)

        forward method for KANEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Convolutional encoding
        for layer in self.encoder_layers:
            x = layer(x)

        # Force spatial size to 8x8 to match flattened_dim=8192
        x = nn.functional.adaptive_avg_pool2d(x, (8, 8))

        # Flatten
        x = x.view(x.size(0), -1)

        # KAN processing
        x = self.kan_encoder(x)

        # Latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class KANDecoder(nn.Module):
    """KAN-based VAE Decoder

    Maps latent vectors back to reconstructed images.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 1,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        kan_hidden_dim: int = 64,
    ):
        """__init__.

        Args:
            latent_dim (int): Description.
            out_channels (int): Description.
            hidden_dims (tuple[int, ...]): Description.
            kan_hidden_dim (int): Description.
        """
        super().__init__()

        # KAN layers for latent processing
        self.kan_decoder = nn.Sequential(
            nn.Linear(latent_dim, kan_hidden_dim),
            KANLayer(kan_hidden_dim, kan_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        hidden_dims[0]

        # First layer from KAN output
        self.reversed_hidden_dims = list(reversed(hidden_dims))
        self.decoder_input = nn.Linear(
            kan_hidden_dim,
            self.reversed_hidden_dims[0] * 4 * 4,
        )

        # Convolutional decoder
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_hidden_dims) - 1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        reversed_hidden_dims[i],
                        reversed_hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(reversed_hidden_dims[i + 1]),
                    nn.ReLU(inplace=True),
                ),
            )

        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                reversed_hidden_dims[-1],
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Tanh(),  # Output in [-1, 1] range
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed image.

        Args:
            z: Latent tensor of shape (batch_size, latent_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, out_channels, H, W)

        forward method for KANDecoder.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # KAN processing
        x = self.kan_decoder(z)

        # Decoder input
        x = self.decoder_input(x)
        x = x.view(x.size(0), self.reversed_hidden_dims[0], 4, 4)

        # Convolutional decoding
        for layer in self.decoder_layers:
            x = layer(x)

        # Final output
        x = self.final_layer(x)
        return x


@register_model(
    name="fourier_kan_vae",
    training_mode="vae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
@register_model(
    name="wavelet_kan_vae",
    training_mode="vae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class KANVAEGenerator(nn.Module, IGenerator):
    """KAN-based Variational Autoencoder Generator

    Implements the IGenerator interface for compatibility with the model factory.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 128,
        kl_weight: float = 0.001,
        hidden_dims: tuple[int, ...] = (32, 64, 128),
        kan_hidden_dim: int = 64,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            latent_dim (int): Description.
            kl_weight (float): Description.
            hidden_dims (tuple[int, ...]): Description.
            kan_hidden_dim (int): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Encoder and decoder
        self.encoder = KANEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            kan_hidden_dim=kan_hidden_dim,
        )

        self.decoder = KANDecoder(
            latent_dim=latent_dim,
            out_channels=out_channels,
            hidden_dims=hidden_dims,
            kan_hidden_dim=kan_hidden_dim,
        )

        # Initialize weights - handled by centralized weight initialization
        # self.apply(self._init_weights)
        self._last_aux: dict[str, torch.Tensor] | None = None

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE.

        Args:
            mu: Mean tensor
            logvar: Log variance tensor

        Returns:
            Sampled latent vector

        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through VAE for generation.

        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            Reconstructed tensor of shape (batch_size, out_channels, H, W)

        forward method for KANVAEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decoder(z)
        # The strategy reads the posterior here, so the KL acts on a bare-tensor forward (#2362).
        self._last_aux = {"mu": mu, "logvar": logvar, "z": z}

        # Resize to original input size if mismatched
        if reconstruction.shape[2:] != x.shape[2:]:
            reconstruction = nn.functional.interpolate(
                reconstruction, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        return reconstruction

    def last_aux(self) -> dict[str, torch.Tensor] | None:
        """The posterior ``(mu, logvar, z)`` of the most recent forward, or None before one."""
        return self._last_aux

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution.

        Args:
            x: Input tensor

        Returns:
            Tuple of (mu, logvar)

        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent tensor

        Returns:
            Reconstructed tensor

        """
        return self.decoder(z)

    def sample(self, num_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """Sample from the latent space.

        Args:
            num_samples: Number of samples to generate
            device: Device to place samples on

        Returns:
            Generated samples

        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    @property
    def name(self) -> str:
        """Get model name."""
        return "kan_vae"

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape.

        Args:
            input_shape: Input tensor shape (C, H, W)

        Returns:
            Output tensor shape (C, H, W)

        """
        return (self.out_channels, input_shape[1], input_shape[2])

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate method for IGenerator interface.

        For VAE, this performs reconstruction of the input.

        Args:
            x: Input tensor
            **kwargs: Additional arguments

        Returns:
            Generated/reconstructed tensor

        """
        reconstruction = self.forward(x)
        return reconstruction

    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            mu: Mean tensor
            logvar: Log variance tensor

        Returns:
            KL divergence loss

        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

    def get_latent_dim(self) -> int:
        """Get latent dimension."""
        return self.latent_dim


def create_kan_vae_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    latent_dim: int = 128,
    kl_weight: float = 0.001,
    **kwargs,
) -> KANVAEGenerator:
    """Factory function for creating KAN VAE generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        latent_dim: Dimension of latent space
        kl_weight: Weight for KL divergence loss
        **kwargs: Additional arguments

    Returns:
        KANVAEGenerator instance

    """
    return KANVAEGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_dim=latent_dim,
        kl_weight=kl_weight,
        **kwargs,
    )
