import math

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces import IDiscriminator, IGenerator
from spectramr.models.layers.kan.kan_convs.kans.layers import FastKANLayer
from spectramr.models.registry import register_model


class ModulatedConv2d(nn.Module):
    """ModulatedConv2d class."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        demodulate=True,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            kernel_size (Any): Description.
            style_dim (Any): Description.
            demodulate (Any): Description.
        """
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.padding = kernel_size // 2

        self.style = nn.Linear(style_dim, in_channels)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size),
        )

    def forward(self, x, style):
        """forward.

        Args:
            x (Any): Description.
            style (Any): Description.
        Returns:
            Any: Description.

        forward method for ModulatedConv2d.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch, in_channels, height, width = x.shape

        style = self.style(style).view(batch, 1, in_channels, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels,
            in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        x = x.view(1, batch * in_channels, height, width)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        out = out.view(batch, self.out_channels, height, width)

        return out


class ToRGB(nn.Module):
    """ToRGB class."""

    def __init__(self, in_channels, style_dim, out_channels=1):
        """__init__.

        Args:
            in_channels (Any): Description.
            style_dim (Any): Description.
            out_channels (Any): Description.
        """
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            1,
            style_dim,
            demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x, style):
        """forward.

        Args:
            x (Any): Description.
            style (Any): Description.
        Returns:
            Any: Description.

        forward method for ToRGB.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.conv(x, style)
        return x + self.bias


class GeneratorBlock(nn.Module):
    """GeneratorBlock class."""

    def __init__(self, in_channels, out_channels, style_dim):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            style_dim (Any): Description.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ModulatedConv2d(in_channels, out_channels, 3, style_dim)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, 3, style_dim)
        self.noise1 = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.noise2 = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.activation = nn.LeakyReLU(0.2)
        self.to_rgb = ToRGB(out_channels, style_dim)

    def forward(self, x, style, noise):
        """forward.

        Args:
            x (Any): Description.
            style (Any): Description.
            noise (Any): Description.
        Returns:
            Any: Description.

        forward method for GeneratorBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            style (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            noise (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.up(x)
        x = self.conv1(x, style)
        x = x + self.noise1 * noise
        x = self.activation(x)
        x = self.conv2(x, style)
        x = x + self.noise2 * noise
        x = self.activation(x)
        rgb = self.to_rgb(x, style)
        return x, rgb


@register_model(name="stylegan2", training_mode="gan")
class StyleGAN2Generator(IGenerator, nn.Module):
    """StyleGAN2 Generator implementation following SOLID principles.

    Single Responsibility: Only handles StyleGAN2 generation
    Open/Closed: Can be extended without modification
    Liskov Substitution: Fully substitutable for any IGenerator
    Interface Segregation: Only implements needed interfaces
    Dependency Inversion: Depends on abstractions, not concretions
    """

    def __init__(self, z_dim, style_dim, n_mlp, out_size, out_channels=1):
        """__init__.

        Args:
            z_dim (Any): Description.
            style_dim (Any): Description.
            n_mlp (Any): Description.
            out_size (Any): Description.
            out_channels (Any): Description.
        """
        super().__init__()
        self.z_dim = z_dim
        self.style_dim = style_dim
        self.out_size = out_size
        self.out_channels = out_channels

        layers = [nn.Linear(z_dim, style_dim), nn.LeakyReLU(0.2)]
        for _ in range(n_mlp - 1):
            layers.extend([nn.Linear(style_dim, style_dim), nn.LeakyReLU(0.2)])
        self.style_mlp = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256,
            128: 128,
            256: 64,
        }

        self.input = nn.Parameter(torch.randn(1, self.channels[4], 4, 4))
        self.log_size = int(math.log2(out_size))

        self.blocks = nn.ModuleList()
        in_ch = self.channels[4]
        for i in range(2, self.log_size):
            out_ch = self.channels[2**i]
            self.blocks.append(GeneratorBlock(in_ch, out_ch, style_dim))
            in_ch = out_ch

        self.tanh = nn.Tanh()

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "StyleGAN2Generator"

    def forward(self, z):
        """forward.

        Args:
            z (Any): Description.
        Returns:
            Any: Description.

        forward method for StyleGAN2Generator.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        styles = self.style_mlp(z)
        x = self.input.repeat(z.shape[0], 1, 1, 1)
        rgb = None

        for block in self.blocks:
            noise = torch.randn(
                z.shape[0],
                1,
                x.shape[2] * 2,
                x.shape[3] * 2,
                device=z.device,
            )
            x, new_rgb = block(x, styles, noise)
            if rgb is not None:
                rgb = F.interpolate(
                    rgb,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=False,
                )
            rgb = new_rgb if rgb is None else rgb + new_rgb

        return self.tanh(rgb)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent space."""
        return self.forward(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        batch_size = input_shape[0]
        return (batch_size, self.out_channels, self.out_size, self.out_size)

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# StyleGAN convolution stack over real-valued images; no internal transform.
@register_model(
    role="discriminator", name="stylegan_discriminator", training_mode="gan", input_domain="image"
)
class StyleGANDiscriminator(IDiscriminator, nn.Module):
    """StyleGAN Discriminator implementation following SOLID principles.

    Single Responsibility: Only handles StyleGAN discrimination
    Open/Closed: Can be extended without modification
    Liskov Substitution: Fully substitutable for any IDiscriminator
    Interface Segregation: Only implements needed interfaces
    Dependency Inversion: Depends on abstractions, not concretions
    """

    def __init__(self, in_size, in_channels=1):
        """__init__.

        Args:
            in_size (Any): Description.
            in_channels (Any): Description.
        """
        super().__init__()
        self.in_size = in_size
        self.in_channels = in_channels
        channels = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256, 128: 128, 256: 64}
        convs = [nn.Conv2d(in_channels, channels[in_size], 1), nn.LeakyReLU(0.2)]
        log_size = int(math.log2(in_size))
        in_ch = channels[in_size]

        for i in range(log_size, 2, -1):
            out_ch = channels[2 ** (i - 1)]
            convs.extend(
                [
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.AvgPool2d(2),
                ],
            )
            in_ch = out_ch

        convs.extend([nn.Conv2d(in_ch, channels[4], 3, padding=1), nn.LeakyReLU(0.2)])
        self.convs = nn.Sequential(*convs)
        self.final_conv = nn.Conv2d(channels[4], channels[4], 4)
        self.final_linear = nn.Sequential(
            nn.Linear(channels[4], channels[4]),
            nn.LeakyReLU(0.2),
            nn.Linear(channels[4], 1),
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "StyleGANDiscriminator"

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for StyleGANDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.convs(x)
        x = self.final_conv(x)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        return x

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake samples."""
        return self.forward(x)

    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract feature maps for feature matching."""
        features = {}
        layer_idx = 0

        for _i, layer in enumerate(self.convs):
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                features[f"conv_{layer_idx}"] = x
                layer_idx += 1
            elif isinstance(layer, (nn.LeakyReLU, nn.AvgPool2d)):
                x = layer(x)

        # Final conv layer
        x = self.final_conv(x)
        features["final_conv"] = x

        return features

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_stylegan2_pair(
    z_dim=512,
    style_dim=512,
    n_mlp=8,
    out_size=256,
    in_channels=1,
    **kwargs,
):
    """get_stylegan2_pair.

    Args:
        z_dim (Any): Description.
        style_dim (Any): Description.
        n_mlp (Any): Description.
        out_size (Any): Description.
        in_channels (Any): Description.
    Returns:
        Any: Description.
    """
    generator = StyleGAN2Generator(
        z_dim,
        style_dim,
        n_mlp,
        out_size,
        out_channels=in_channels,
    )
    discriminator = StyleGANDiscriminator(out_size, in_channels=in_channels)
    return generator, discriminator


class StyleKANMLP(nn.Module):
    """StyleKANMLP class."""

    def __init__(self, z_dim, style_dim, n_mlp):
        """__init__.

        Args:
            z_dim (Any): Description.
            style_dim (Any): Description.
            n_mlp (Any): Description.
        """
        super().__init__()
        layers = [FastKANLayer(z_dim, style_dim)]
        for _ in range(n_mlp - 1):
            layers.append(FastKANLayer(style_dim, style_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for StyleKANMLP.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


@register_model(name="stylegan_kan_gen", training_mode="gan")
class StyleGANKANGenerator(StyleGAN2Generator):
    """StyleGAN-KAN Generator implementation following SOLID principles.

    Single Responsibility: Only handles StyleGAN-KAN generation
    Open/Closed: Can be extended without modification
    Liskov Substitution: Fully substitutable for any IGenerator
    Interface Segregation: Only implements needed interfaces
    Dependency Inversion: Depends on abstractions, not concretions
    """

    def __init__(self, z_dim, style_dim, n_mlp, out_size, out_channels=1):
        """__init__.

        Args:
            z_dim (Any): Description.
            style_dim (Any): Description.
            n_mlp (Any): Description.
            out_size (Any): Description.
            out_channels (Any): Description.
        """
        super().__init__(z_dim, style_dim, n_mlp, out_size, out_channels)
        self.style_mlp = StyleKANMLP(z_dim, style_dim, n_mlp)


def get_stylegan_kan_pair(
    z_dim=512,
    style_dim=512,
    n_mlp=8,
    out_size=256,
    in_channels=1,
    **kwargs,
):
    """get_stylegan_kan_pair.

    Args:
        z_dim (Any): Description.
        style_dim (Any): Description.
        n_mlp (Any): Description.
        out_size (Any): Description.
        in_channels (Any): Description.
    Returns:
        Any: Description.
    """
    generator = StyleGANKANGenerator(
        z_dim,
        style_dim,
        n_mlp,
        out_size,
        out_channels=in_channels,
    )
    discriminator = StyleGANDiscriminator(out_size, in_channels=in_channels)
    return generator, discriminator
