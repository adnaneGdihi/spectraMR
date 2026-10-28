#!/usr/bin/env python3
"""3D UNet Generator Implementation
===============================

Concrete implementation of IGenerator for 3D UNet-based generators.
Supports volumetric medical imaging with full 3D convolutions.
Follows SOLID principles with single responsibility and dependency injection.
"""

from itertools import pairwise
from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.convolutional import DoubleConv3D
from spectramr.models.blocks.unet import Down3D, OutConv3D, Up3D
from spectramr.models.registry import register_model

from .base_3d_generator import Base3DGenerator, standardize_3d_params


@register_model(name="unet3d", training_mode="reconstruction")
class UNet3DGenerator(Base3DGenerator):
    """3D UNet-based generator implementation following SOLID principles.

    Single Responsibility: Only handles 3D UNet generation for volumetric data
    Open/Closed: Can be extended without modification
    Liskov Substitution: Fully substitutable for any IGenerator
    Interface Segregation: Only implements needed interfaces
    Dependency Inversion: Depends on abstractions, not concretions
    """

    __expected_kwargs__: set[str] = {
        "depth",
        "height",
        "width",
        "latent_shape",
        "output_dims",
        "device",
    }

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (32, 64, 128, 256),
        trilinear: bool = True,
        output_activation: nn.Module | None = None,
        **kwargs,
    ):
        # Standardize parameters for backward compatibility while allowing
        # callers to override the default spatial dimensions.
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (tuple[int, ...]): Description.
            trilinear (bool): Description.
            output_activation (Optional[nn.Module]): Description.
        """
        standardized_kwargs: dict[str, Any] = dict(kwargs)
        standardized_kwargs.setdefault("depth", 16)
        standardized_kwargs.setdefault("height", 16)
        standardized_kwargs.setdefault("width", 16)

        params: dict[str, Any] = standardize_3d_params(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            output_activation=output_activation,
            **standardized_kwargs,
        )

        super().__init__(**params)
        if not trilinear:
            # Up3D's transposed conv assumes the input carries half the concatenated
            # channels, which a doubling feature ladder never does.
            raise ValueError("UNet3DGenerator supports trilinear=True only")
        self.trilinear = trilinear
        features = tuple(self.features)

        # Full-resolution stem, then one Down3D (pool + double conv) per deeper level.
        self.inc = DoubleConv3D(self.in_channels, features[0])
        self.downs = nn.ModuleList(Down3D(a, b) for a, b in pairwise(features))
        # Each Up3D upsamples level i+1 and concatenates the level-i skip.
        self.ups = nn.ModuleList(
            Up3D(b + a, a, trilinear)
            for a, b in reversed(list(pairwise(features)))
        )
        self.outc = OutConv3D(features[0], self.out_channels, activation=nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 3D UNet.

        Args:
            x: Input tensor [N, C, D, H, W]; each spatial extent must divide by
               ``2 ** (len(features) - 1)``.

        Returns:
            Output tensor [N, C_out, D, H, W] at the input resolution.
        """
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up in self.ups:
            x = up(x, skips.pop())
        return self.output_activation(self.outc(x))


def create_unet3d_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    features: tuple[int, ...] = (32, 64, 128, 256),
    trilinear: bool = True,
    output_activation: nn.Module | None = None,
    **kwargs,
) -> UNet3DGenerator:
    """Create a 3D UNet generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        features: Number of features at each level
        trilinear: Whether to use trilinear upsampling
        output_activation: Optional output activation function
        **kwargs: Additional arguments passed to UNet3DGenerator

    Returns:
        UNet3DGenerator instance

    """
    return UNet3DGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        trilinear=trilinear,
        output_activation=output_activation,
        **kwargs,
    )
