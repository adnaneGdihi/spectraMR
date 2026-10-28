"""3D PatchGAN Discriminator
==========================

A 3D PatchGAN discriminator for volumetric data.
"""

import torch
from torch import nn

from spectramr.models.interfaces import IDiscriminator
from spectramr.models.registry import register_model


# 3-D convolution stack, no internal transform (see ``patchgan_discriminator.py`` for why
# the domain is declared on the registration rather than read off the class).
@register_model(
    role="discriminator",
    name="patchgan_3d_discriminator",
    training_mode="gan",
    input_domain="image",
)
class PatchGAN3DDiscriminator(IDiscriminator, nn.Module):
    """A 3D PatchGAN discriminator."""

    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3,
        spectral_norm: bool = False,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            ndf (int): Description.
            n_layers (int): Description.
            spectral_norm (bool): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.ndf = ndf
        self.n_layers = n_layers
        self.spectral_norm = spectral_norm

        layers = [
            self._create_conv_layer(
                in_channels,
                ndf,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                self._create_conv_layer(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm3d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            self._create_conv_layer(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            self._create_conv_layer(
                ndf * nf_mult,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        ]

        self.model = nn.Sequential(*layers)
        self.use_checkpoint = True

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
        conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        if self.spectral_norm:
            return nn.utils.spectral_norm(conv)
        return conv

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for PatchGAN3DDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if getattr(self, "use_checkpoint", False) and self.training:
            return torch.utils.checkpoint.checkpoint(self.model, x, use_reentrant=False)
        return self.model(x)

    def discriminate(self, x, **kwargs):
        """discriminate.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        return self.forward(x)

    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract feature maps for feature matching."""
        features = {}
        layer_idx = 0

        for _i, layer in enumerate(self.model):
            if isinstance(layer, nn.Conv3d):
                x = layer(x)
                features[f"conv_{layer_idx}"] = x
                layer_idx += 1
            elif isinstance(layer, (nn.BatchNorm3d, nn.LeakyReLU)):
                x = layer(x)
            else:
                # Final activation
                x = layer(x)

        return features

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "PatchGAN3DDiscriminator"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
