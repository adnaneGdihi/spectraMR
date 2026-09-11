"""Conditional PatchGAN Discriminator
==================================

A PatchGAN discriminator that accepts conditional inputs.
"""

import torch
from torch import nn

from spectramr.models.interfaces import IDiscriminator
from spectramr.models.registry import register_model
from spectramr.shared.utils.metaclass import ModuleABCMeta


# Concatenation-conditioned convolution stack, no internal transform: it scores whatever
# space it is handed. Both live arms (mrixfields2026/task3/field_cocycle_*) declare
# ``losses.policy.output_domain: image``, which is the space this critic is trained on.
@register_model(
    role="discriminator",
    name="conditional_patchgan_discriminator",
    training_mode="gan",
    input_domain="image",
)
class ConditionalPatchGANDiscriminator(IDiscriminator, nn.Module, metaclass=ModuleABCMeta):
    """A PatchGAN discriminator that is conditioned on additional inputs.
    The conditional inputs are concatenated to the input image along the
    channel dimension.
    """

    def __init__(
        self,
        in_channels: int = 1,
        cond_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        spectral_norm: bool = False,
    ):
        """Initialize ConditionalPatchGANDiscriminator.

        Args:
            in_channels: Number of channels in the input image.
            cond_channels: Number of channels in the conditional input.
            ndf: Number of discriminator filters in the first conv layer.
            n_layers: Number of conv layers in the discriminator.
            spectral_norm: If True, use spectral normalization.

        """
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.ndf = ndf
        self.n_layers = n_layers
        self.spectral_norm = spectral_norm

        # The total number of input channels is the sum of image and conditional
        # channels
        total_in_channels = in_channels + cond_channels

        self.layers = nn.ModuleList()
        self.feature_layers = []

        # First layer
        self.layers.append(
            self._create_conv_layer(
                total_in_channels,
                ndf,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            self.layers.append(
                self._create_conv_layer(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            self.layers.append(nn.BatchNorm2d(ndf * nf_mult))
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.feature_layers.append(len(self.layers) - 3)  # conv layer index

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        self.layers.append(
            self._create_conv_layer(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            )
        )
        self.layers.append(nn.BatchNorm2d(ndf * nf_mult))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.feature_layers.append(len(self.layers) - 3)  # conv layer index

        # Final output layer
        self.layers.append(
            self._create_conv_layer(
                ndf * nf_mult,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            )
        )

    def _create_conv_layer(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ) -> nn.Module:
        """_create_conv_layer.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        Returns:
            nn.Module: Description.
        """
        conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        if self.spectral_norm:
            return nn.utils.spectral_norm(conv)
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator.

        For conditional discriminators, x should be the concatenated input
        of image and conditional tensors.

        forward method for ConditionalPatchGANDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for layer in self.layers:
            x = layer(x)
        return x

    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Discriminate real vs fake samples."""
        cond = kwargs.get("cond")
        if cond is not None:
            # x and cond provided separately
            combined = torch.cat([x, cond], 1)
        else:
            # Assume x is already concatenated
            combined = x
        for layer in self.layers:
            combined = layer(combined)
        return combined

    def get_feature_maps(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Extract feature maps from intermediate layers."""
        cond = kwargs.get("cond")
        if cond is not None:
            # x and cond provided separately
            combined = torch.cat([x, cond], 1)
        else:
            # Assume x is already concatenated
            combined = x
        features = {}
        for i, layer in enumerate(self.layers):
            combined = layer(combined)
            if i in self.feature_layers:
                features[f"layer_{len(features) + 1}"] = combined
        return features

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "ConditionalPatchGANDiscriminator"

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
