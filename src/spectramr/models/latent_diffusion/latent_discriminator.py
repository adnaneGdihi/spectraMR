#!/usr/bin/env python3
"""Latent Discriminator Module
===========================

This module implements discriminator architectures designed for latent space
discrimination in latent GAN models.
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.registry import register_model


@register_model(
    role="discriminator",
    name="ldm_latent_discriminator",
    training_mode="gan",
    input_domain="latent",
)
class LatentDiscriminator(nn.Module):
    """Discriminator for latent space in latent GAN models.

    This discriminator operates on latent representations. It can handle
    both latent vectors (2D) and latent images (4D).

    Args:
        in_channels: Number of input channels (for 4D)
            or latent dimension (for 2D)
        base_channels: Base number of channels for the first layer
        channel_mult: Multipliers for channel progression
        num_layers: Number of convolutional layers
        use_spectral_norm: Whether to use spectral normalization
        dropout: Dropout probability
        input_type: 'vector' for 2D latent vectors,
            'image' for 4D latent images

    """

    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int | None = None,
        base_channels: int = 64,
        channel_mult: tuple[int, ...] = (1, 2, 4, 8),
        num_layers: int = 4,
        use_spectral_norm: bool = True,
        dropout: float = 0.1,
        input_type: str = "vector",  # 'vector' or 'image'
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int | None): Description.
            base_channels (int): Description.
            channel_mult (tuple[int, ...]): Description.
            num_layers (int): Description.
            use_spectral_norm (bool): Description.
            dropout (float): Description.
            input_type (str): Description.
        """
        super().__init__()

        # Allow callers to specify latent_dim alias for in_channels
        if latent_dim is not None and in_channels == 4:
            in_channels = latent_dim

        self.in_channels = in_channels
        self.use_spectral_norm = use_spectral_norm
        self.input_type = input_type

        if input_type == "vector":
            # For latent vectors: simple MLP discriminator
            self.discriminator = self._build_vector_discriminator(in_channels)
        else:
            # For latent images: convolutional discriminator
            self.discriminator = self._build_image_discriminator(
                in_channels,
                base_channels,
                channel_mult,
                num_layers,
                dropout,
            )

    def _build_vector_discriminator(self, latent_dim: int) -> nn.Sequential:
        """Build MLP discriminator for latent vectors."""
        layers = []

        # First linear layer
        linear1 = nn.Linear(latent_dim, 512)
        if self.use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
        layers.extend([linear1, nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.1)])

        # Second linear layer
        linear2 = nn.Linear(512, 256)
        if self.use_spectral_norm:
            linear2 = nn.utils.spectral_norm(linear2)
        layers.extend([linear2, nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.1)])

        # Output layer
        linear3 = nn.Linear(256, 1)
        if self.use_spectral_norm:
            linear3 = nn.utils.spectral_norm(linear3)
        layers.append(linear3)

        return nn.Sequential(*layers)

    def _build_image_discriminator(
        self,
        in_channels: int,
        base_channels: int,
        channel_mult: tuple[int, ...],
        num_layers: int,
        dropout: float,
    ) -> nn.Sequential:
        """Build convolutional discriminator for latent images."""
        layers = []
        current_channels = in_channels

        for _i, mult in enumerate(channel_mult):
            out_channels = base_channels * mult

            # Convolutional layer
            conv = nn.Conv2d(
                current_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )

            # Apply spectral normalization if requested
            if self.use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)

            layers.append(conv)

            # Batch normalization (not used with spectral norm)
            if not self.use_spectral_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            # Activation
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))

            current_channels = out_channels

        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        # Flatten
        layers.append(nn.Flatten())

        # Final classification layer
        final_conv = nn.Linear(current_channels, 1)
        if self.use_spectral_norm:
            final_conv = nn.utils.spectral_norm(final_conv)
        layers.append(final_conv)

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the latent discriminator.

        Args:
            x: Input latent tensor [B, latent_dim] for vectors
                or [B, C, H, W] for images

        Returns:
            Discrimination score [B, 1]

        forward method for LatentDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.input_type == "vector":
            # For vector inputs with MLP discriminator, no reshaping needed
            pass  # x is already [B, latent_dim]
        # For image inputs with conv discriminator, ensure 4D
        elif x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # [B, latent_dim, 1, 1]

        return self.discriminator(x)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        # Output is always [B, 1] for discrimination score
        return (input_shape[0], 1)


@register_model(
    role="discriminator",
    name="ldm_patch_latent_discriminator",
    training_mode="gan",
    input_domain="latent",
)
class PatchLatentDiscriminator(nn.Module):
    """Patch-based discriminator for latent space.

    This discriminator provides patch-level discrimination scores,
    similar to PatchGAN but operating in latent space.

    Args:
        in_channels: Number of input channels (latent dimension)
        base_channels: Base number of channels for the first layer
        use_spectral_norm: Whether to use spectral normalization
        dropout: Dropout probability

    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        use_spectral_norm: bool = True,
        dropout: float = 0.1,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            base_channels (int): Description.
            use_spectral_norm (bool): Description.
            dropout (float): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.use_spectral_norm = use_spectral_norm

        # Build patch discriminator layers
        layers = []

        # First layer
        conv1 = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        if use_spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
        layers.extend(
            [
                conv1,
                nn.LeakyReLU(0.2, inplace=True),
            ],
        )

        # Second layer
        conv2 = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        if use_spectral_norm:
            conv2 = nn.utils.spectral_norm(conv2)
        layers.extend(
            [
                conv2,
                nn.LeakyReLU(0.2, inplace=True),
            ],
        )

        # Third layer
        conv3 = nn.Conv2d(
            base_channels * 2,
            base_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        if use_spectral_norm:
            conv3 = nn.utils.spectral_norm(conv3)
        layers.extend(
            [
                conv3,
                nn.LeakyReLU(0.2, inplace=True),
            ],
        )

        # Fourth layer (output layer)
        conv4 = nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            conv4 = nn.utils.spectral_norm(conv4)
        layers.append(conv4)

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the patch latent discriminator.

        Args:
            x: Input latent tensor [B, C, H, W]

        Returns:
            Patch discrimination scores [B, 1, H', W']

        forward method for PatchLatentDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.discriminator(x)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(
        self,
        input_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        """Get output shape for given input shape."""
        # Calculate output spatial dimensions after 4 strided convolutions
        # Each conv: stride=2, except last which has stride=1
        h_out = input_shape[2] // 16  # 2^4 = 16
        w_out = input_shape[3] // 16
        return (input_shape[0], 1, h_out, w_out)


@register_model(
    role="discriminator",
    name="ldm_multiscale_latent_discriminator",
    training_mode="gan",
    input_domain="latent",
)
class MultiScaleLatentDiscriminator(nn.Module):
    """Multi-scale discriminator for latent space.

    This discriminator uses multiple scales to provide richer discrimination
    signals for latent GAN training.

    Args:
        in_channels: Number of input channels (latent dimension)
        base_channels: Base number of channels for the first layer
        num_scales: Number of scales to use
        use_spectral_norm: Whether to use spectral normalization

    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_scales: int = 3,
        use_spectral_norm: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            base_channels (int): Description.
            num_scales (int): Description.
            use_spectral_norm (bool): Description.
        """
        super().__init__()

        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        for _i in range(num_scales):
            # Create discriminator for this scale
            discriminator = LatentDiscriminator(
                in_channels=in_channels,
                base_channels=base_channels,
                channel_mult=(1, 2, 4),
                num_layers=3,
                use_spectral_norm=use_spectral_norm,
                dropout=0.1,
            )
            self.discriminators.append(discriminator)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass through multi-scale discriminators.

        Args:
            x: Input latent tensor [B, C, H, W]

        Returns:
            Tuple of discrimination scores from each scale

        forward method for MultiScaleLatentDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, ...]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        outputs = []
        current_x = x

        for i, discriminator in enumerate(self.discriminators):
            # Apply average pooling for coarser scales
            if i > 0:
                current_x = F.avg_pool2d(current_x, kernel_size=2, stride=2)

            output = discriminator(current_x)
            outputs.append(output)

        return tuple(outputs)

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shapes(
        self,
        input_shape: tuple[int, ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Get output shapes for all scales."""
        shapes = []
        current_shape = input_shape

        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                # Account for average pooling
                current_shape = (
                    current_shape[0],
                    current_shape[1],
                    current_shape[2] // 2,
                    current_shape[3] // 2,
                )

            shape = discriminator.get_output_shape(current_shape)
            shapes.append(shape)

        return tuple(shapes)


# Convenience functions for creating latent discriminators
def create_latent_discriminator(
    in_channels: int = 4,
    discriminator_type: str = "standard",
    **kwargs,
) -> nn.Module:
    """Create a latent discriminator.

    Args:
        in_channels: Number of input channels
        discriminator_type: Type of discriminator
            ("standard", "patch", "multiscale")
        **kwargs: Additional arguments for the discriminator

    Returns:
        Configured discriminator instance

    """
    if discriminator_type == "patch":
        return PatchLatentDiscriminator(in_channels=in_channels, **kwargs)
    if discriminator_type == "multiscale":
        return MultiScaleLatentDiscriminator(in_channels=in_channels, **kwargs)
    if discriminator_type == "standard":
        return LatentDiscriminator(in_channels=in_channels, **kwargs)
    raise ValueError(
        f"Unknown discriminator_type {discriminator_type!r}; "
        "expected one of ('standard', 'patch', 'multiscale')."
    )


def create_standard_latent_discriminator(
    in_channels: int = 4,
    base_channels: int = 64,
    **kwargs,
) -> LatentDiscriminator:
    """Create a standard latent discriminator.

    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels
        **kwargs: Additional arguments

    Returns:
        Standard latent discriminator

    """
    return LatentDiscriminator(
        in_channels=in_channels,
        base_channels=base_channels,
        **kwargs,
    )


def create_patch_latent_discriminator(
    in_channels: int = 4,
    base_channels: int = 64,
    **kwargs,
) -> PatchLatentDiscriminator:
    """Create a patch-based latent discriminator.

    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels
        **kwargs: Additional arguments

    Returns:
        Patch latent discriminator

    """
    return PatchLatentDiscriminator(
        in_channels=in_channels,
        base_channels=base_channels,
        **kwargs,
    )


def create_multiscale_latent_discriminator(
    in_channels: int = 4,
    base_channels: int = 64,
    num_scales: int = 3,
    **kwargs,
) -> MultiScaleLatentDiscriminator:
    """Create a multi-scale latent discriminator.

    Args:
        in_channels: Number of input channels
        base_channels: Base number of channels
        num_scales: Number of scales
        **kwargs: Additional arguments

    Returns:
        Multi-scale latent discriminator

    """
    return MultiScaleLatentDiscriminator(
        in_channels=in_channels,
        base_channels=base_channels,
        num_scales=num_scales,
        **kwargs,
    )
