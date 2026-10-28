"""Simple Enhanced Deep Diffusion Models
Simplified but robust implementations of the three diffusion variants.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.constants import DEFAULT_DIFFUSION_TIMESTEPS
from spectramr.models.registry import register_model


class SimpleTimeEmbedding(nn.Module):
    """Simple time embedding for diffusion models."""

    def __init__(self, dim: int):
        """__init__.

        Args:
            dim (int): Description.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """forward.

        Args:
            time (Any): Description.
        Returns:
            Any: Description.

        forward method for SimpleTimeEmbedding.

        Executes PyTorch tensor operations.

        Args:
            time (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class KANLayer(nn.Module):
    """Simplified Kolmogorov-Arnold Network Layer
    Uses learnable spline-based activation functions
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            grid_size (int): Description.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # Learnable spline coefficients
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size),
        )

        # Base linear transformation
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        """reset_parameters.

        Returns:
            Any: Description.
        """
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input to [-1, 1] range for splines
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KANLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x_norm = torch.tanh(x)

        # Create grid points
        grid = torch.linspace(-1, 1, self.grid_size, device=x.device)

        # Compute RBF-like basis functions (Gaussian kernels)
        # x_norm: [batch_size, in_features]
        # grid: [grid_size]
        x_expanded = x_norm.unsqueeze(-1)  # [batch_size, in_features, 1]
        grid_expanded = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, grid_size]

        # Gaussian basis with learnable width
        sigma = 2.0 / self.grid_size  # Adaptive width
        basis = torch.exp(-0.5 * ((x_expanded - grid_expanded) / sigma) ** 2)

        # Apply learnable spline transformation
        # basis: [batch_size, in_features, grid_size]
        # spline_weight: [out_features, in_features, grid_size]
        spline_output = torch.einsum("big,oig->bo", basis, self.spline_weight)

        # Base linear transformation
        base_output = F.linear(x, self.base_weight, self.bias)

        return base_output + spline_output


class SimpleKANTimeEmbedding(nn.Module):
    """KAN-based time embedding using learnable spline activations."""

    def __init__(self, dim: int, num_timesteps: int):
        """__init__.

        Args:
            dim (int): Description.
            num_timesteps (int): Diffusion horizon; ``t`` is divided by it.
        """
        super().__init__()
        self.dim = dim
        self.num_timesteps = int(num_timesteps)
        hidden_dim = dim // 2

        # Use KAN layers instead of regular linear layers
        self.kan_layers = nn.Sequential(
            KANLayer(1, hidden_dim, grid_size=5),
            KANLayer(hidden_dim, hidden_dim, grid_size=5),
            KANLayer(hidden_dim, dim, grid_size=5),
        )

    def forward(self, time):
        """forward.

        Args:
            time (Any): Description.
        Returns:
            Any: Description.

        forward method for SimpleKANTimeEmbedding.

        Executes PyTorch tensor operations.

        Args:
            time (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        time = time.float().unsqueeze(-1) / self.num_timesteps
        return self.kan_layers(time)


class SimpleResBlock(nn.Module):
    """Simple residual block with time conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_dim (int): Description.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)

        # Skip connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x, time_emb):
        """forward.

        Args:
            x (Any): Description.
            time_emb (Any): Description.
        Returns:
            Any: Description.

        forward method for SimpleResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            time_emb (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = self.skip_conv(x)

        # First conv
        out = self.conv1(x)
        out = self.norm1(out)

        # Add time conditioning
        time_out = self.time_mlp(time_emb)[:, :, None, None]
        out = out + time_out
        out = F.silu(out)

        # Second conv
        out = self.conv2(out)
        out = self.norm2(out)

        # Residual connection
        return F.silu(out + residual)


@register_model(name="simple_enhanced_diffusion", training_mode="diffusion")
class SimpleEnhancedDiffusionUNet(nn.Module):
    """Simple but robust enhanced diffusion U-Net."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        model_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: tuple = (1, 2, 4),
        time_embedding_type: str = "sinusoidal",
        noise_type: str = "gaussian",
        num_timesteps: int = DEFAULT_DIFFUSION_TIMESTEPS,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            model_channels (int): Description.
            num_res_blocks (int): Description.
            channel_mult (tuple): Description.
            time_embedding_type (str): Description.
            noise_type (str): Description.
            num_timesteps (int): Diffusion horizon the KAN time code divides by;
                the builder injects ``training.diffusion.timesteps``.
        """
        super().__init__()

        valid_time_embedding_types = ("sinusoidal", "kan")
        if time_embedding_type not in valid_time_embedding_types:
            raise ValueError(
                f"Unknown time_embedding_type '{time_embedding_type}'. "
                f"Valid options: {valid_time_embedding_types}"
            )
        valid_noise_types = ("gaussian", "chi_square", "gaussian_kan")
        if noise_type not in valid_noise_types:
            raise ValueError(
                f"Unknown noise_type '{noise_type}'. Valid options: {valid_noise_types}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.noise_type = noise_type

        # Time embeddings
        time_embed_dim = model_channels * 4
        if time_embedding_type == "kan":
            self.time_embed = SimpleKANTimeEmbedding(time_embed_dim, num_timesteps)
        else:
            self.time_embed = SimpleTimeEmbedding(time_embed_dim)

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        ch = model_channels
        channels_list = [ch]  # Track channels for skip connections

        for i, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            # Residual blocks
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = ch if j == 0 else out_ch
                blocks.append(SimpleResBlock(in_ch, out_ch, time_embed_dim))
                if j == 0:  # Only change channels on first block
                    ch = out_ch
            self.encoder_blocks.append(blocks)
            channels_list.append(ch)

            # Downsampling (except last level)
            if i < len(channel_mult) - 1:
                self.down_samples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())

        # Middle block
        self.middle_block = nn.ModuleList(
            [
                SimpleResBlock(ch, ch, time_embed_dim),
                SimpleResBlock(ch, ch, time_embed_dim),
            ],
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for i, mult in enumerate(reversed(channel_mult)):
            out_ch = model_channels * mult

            # Residual blocks
            blocks = nn.ModuleList()
            for j in range(num_res_blocks + 1):
                # First block takes concatenated channels (current + skip)
                if j == 0:
                    skip_ch = channels_list.pop() if channels_list else 0
                    in_ch = ch + skip_ch
                else:
                    in_ch = out_ch
                blocks.append(SimpleResBlock(in_ch, out_ch, time_embed_dim))
                ch = out_ch
            self.decoder_blocks.append(blocks)

            # Upsampling (except last level)
            if i < len(channel_mult) - 1:
                self.up_samples.append(
                    nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1),
                )
            else:
                self.up_samples.append(nn.Identity())

        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(32, model_channels), model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

        # Noise-specific components
        if noise_type == "chi_square":
            self.noise_scale = nn.Parameter(torch.ones(1) * 0.5)
        elif noise_type == "gaussian_kan":
            # Use KAN layers for enhanced feature processing
            self.kan_processor = nn.Sequential(
                KANLayer(model_channels, model_channels, grid_size=5),
                nn.Conv2d(model_channels, model_channels, 1),
                nn.GroupNorm(min(32, model_channels), model_channels),
            )

    def forward(self, x, timesteps):
        # Time embedding
        """forward.

        Args:
            x (Any): Description.
            timesteps (Any): Description.
        Returns:
            Any: Description.

        forward method for SimpleEnhancedDiffusionUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        time_emb = self.time_embed(timesteps)

        # Input projection
        h = self.input_conv(x)

        # Encoder with skip connections
        skip_connections = []

        for blocks, down_sample in zip(self.encoder_blocks, self.down_samples, strict=False):
            for block in blocks:
                h = block(h, time_emb)
            skip_connections.append(h)
            h = down_sample(h)

        # Middle block
        for block in self.middle_block:
            h = block(h, time_emb)

        # Decoder
        for blocks, up_sample in zip(self.decoder_blocks, self.up_samples, strict=False):
            # Skip connection
            if skip_connections:
                skip = skip_connections.pop()
                # Ensure spatial dimensions match
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(
                        h,
                        size=skip.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                h = torch.cat([h, skip], dim=1)

            # Process blocks
            for i, block in enumerate(blocks):
                if i == 0:
                    # First block handles concatenated channels
                    h = block(h, time_emb)
                else:
                    h = block(h, time_emb)

            h = up_sample(h)

        # Noise-specific processing
        if self.noise_type == "gaussian_kan" and hasattr(self, "kan_processor"):
            # Apply KAN processing to spatial features
            batch_size, channels, height, width = h.shape
            h_flat = h.view(batch_size, channels, -1).permute(0, 2, 1).contiguous()  # [B, H*W, C]
            h_flat = h_flat.view(-1, channels)  # [B*H*W, C] for KAN layer

            h_kan = self.kan_processor[0](h_flat)  # Apply KAN layer

            # Reshape back to spatial
            spatial_size = height * width
            h_kan = h_kan.view(batch_size, spatial_size, channels).permute(
                0,
                2,
                1,
            )  # [B, C, H*W]
            h_kan = h_kan.view(batch_size, channels, height, width)  # [B, C, H, W]

            h_kan = self.kan_processor[1](h_kan)  # Conv2d
            h = self.kan_processor[2](h_kan)  # GroupNorm
        elif self.noise_type == "chi_square" and hasattr(self, "noise_scale"):
            h = h * self.noise_scale

        # Output projection
        return self.output_conv(h)


@register_model(name="simple_gaussian_diffusion", training_mode="diffusion")
class SimpleGaussianDiffusionUNet(SimpleEnhancedDiffusionUNet):
    """Gaussian noise diffusion variant."""

    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(
            noise_type="gaussian",
            time_embedding_type="sinusoidal",
            **kwargs,
        )


@register_model(name="simple_chi_square_diffusion", training_mode="diffusion")
class SimpleChiSquareDiffusionUNet(SimpleEnhancedDiffusionUNet):
    """Chi-square noise diffusion variant."""

    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(
            noise_type="chi_square",
            time_embedding_type="sinusoidal",
            **kwargs,
        )


@register_model(name="simple_gaussian_kan_diffusion", training_mode="diffusion")
class SimpleGaussianKANDiffusionUNet(SimpleEnhancedDiffusionUNet):
    """Gaussian noise with KAN embeddings diffusion variant."""

    def __init__(self, num_timesteps: int = DEFAULT_DIFFUSION_TIMESTEPS, **kwargs: Any):
        """Name the horizon so the builder injects it into the KAN time code."""
        super().__init__(
            noise_type="gaussian_kan",
            time_embedding_type="kan",
            num_timesteps=num_timesteps,
            **kwargs,
        )


def create_simple_diffusion_model(variant: str = "gaussian", **kwargs):
    """Factory function to create simple diffusion model variants.

    Args:
        variant: One of ["gaussian", "chi_square", "gaussian_kan"]
        **kwargs: Additional arguments for model construction

    Returns:
        Diffusion model instance

    """
    model_classes = {
        "gaussian": SimpleGaussianDiffusionUNet,
        "chi_square": SimpleChiSquareDiffusionUNet,
        "gaussian_kan": SimpleGaussianKANDiffusionUNet,
    }

    if variant not in model_classes:
        raise ValueError(
            f"Unknown variant: {variant}. Choose from {list(model_classes.keys())}",
        )

    return model_classes[variant](**kwargs)


# Alias for compatibility with model factory
create_enhanced_deep_diffusion = create_simple_diffusion_model
