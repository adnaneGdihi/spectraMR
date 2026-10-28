"""Shared Embeddings
=================

Consolidated embedding layers used across models.
"""

import math

import torch
from torch import nn

# Import KAN classes from the kan_convs package
try:
    from spectramr.models.layers.kan.kan_convs.kans import (
        ChebyKANLayer,
        FastKAN,
        KANLayer,
        WavKANLayer,
    )

    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False


class PatchEmbedding(nn.Module):
    """Patch embedding layer for Vision Transformer"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """__init__.

        Args:
            img_size (int): Description.
            patch_size (int): Description.
            in_channels (int): Description.
            embed_dim (int): Description.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        patches_h = img_size // patch_size
        patches_w = img_size // patch_size
        self.patches_resolution = (patches_h, patches_w)
        self.num_patches = patches_h * patches_w

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for PatchEmbedding.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        # (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for time steps in diffusion models"""

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

        forward method for SinusoidalPositionEmbedding.

        Executes PyTorch tensor operations.

        Args:
            time (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        device = time.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FourierKANPositionalEmbedding(nn.Module):
    """Fourier-based KAN positional embedding for sequences"""

    def __init__(self, dim: int, max_positions: int = 10000, num_frequencies: int = 8):
        """__init__.

        Args:
            dim (int): Description.
            max_positions (int): Description.
            num_frequencies (int): Description.
        """
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions
        self.num_frequencies = num_frequencies

        # Fourier basis parameters
        self.frequencies = nn.Parameter(
            torch.randn(num_frequencies // 2) * 0.1,
            requires_grad=True,
        )

        # Learnable weights for Fourier components
        self.sin_weights = nn.Linear(num_frequencies // 2, dim // 2)
        self.cos_weights = nn.Linear(num_frequencies // 2, dim // 2)

        # Position normalization
        self.position_scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, positions):
        """Args:
            positions: Tensor of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, dim)

        forward method for FourierKANPositionalEmbedding.

        Executes PyTorch tensor operations.

        Args:
            positions (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # Add batch dimension

        # Normalize positions
        positions = positions.float() / self.max_positions * self.position_scale

        # Generate Fourier features
        sin_features = []
        cos_features = []

        for i in range(self.num_frequencies // 2):
            freq = self.frequencies[i]
            sin_features.append(torch.sin(2 * torch.pi * freq * positions))
            cos_features.append(torch.cos(2 * torch.pi * freq * positions))

        # Stack features: (batch, seq_len, num_freq/2)
        sin_features = torch.stack(sin_features, dim=-1)
        cos_features = torch.stack(cos_features, dim=-1)

        # Apply learnable weights
        sin_embed = self.sin_weights(sin_features)  # (batch, seq_len, dim//2)
        cos_embed = self.cos_weights(cos_features)  # (batch, seq_len, dim//2)

        # Concatenate sin and cos components
        embeddings = torch.cat([sin_embed, cos_embed], dim=-1)

        return embeddings


class AdvancedKANPositionalEmbedding(nn.Module):
    """Advanced KAN-based positional embedding using actual KAN layers"""

    def __init__(
        self,
        dim: int,
        max_positions: int = 10000,
        kan_type: str = "fast",
        grid_size: int = 8,
    ):
        """__init__.

        Args:
            dim (int): Description.
            max_positions (int): Description.
            kan_type (str): Description.
            grid_size (int): Description.
        """
        super().__init__()
        if not KAN_AVAILABLE:
            raise ImportError("KAN layers not available. Install kan_convs package.")

        self.dim = dim
        self.max_positions = max_positions
        self.kan_type = kan_type

        # Position encoding layers
        self.position_encoder = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
        )

        # Choose KAN layer type
        if kan_type == "fast":
            self.kan_layer = FastKAN(layers_hidden=[dim // 2, dim], grid_size=grid_size)
        elif kan_type == "chebyshev":
            self.kan_layer = ChebyKANLayer(
                input_dim=dim // 2,
                output_dim=dim,
                degree=grid_size,
            )
        elif kan_type == "wavelet":
            self.kan_layer = WavKANLayer(in_features=dim // 2, out_features=dim)
        else:
            self.kan_layer = KANLayer(
                input_features=dim // 2,
                output_features=dim,
                grid_size=grid_size,
            )

        # Position normalization
        self.position_scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, positions):
        """Args:
            positions: Tensor of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, dim)

        forward method for AdvancedKANPositionalEmbedding.

        Executes PyTorch tensor operations.

        Args:
            positions (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # Add batch dimension

        batch_size, seq_len = positions.shape

        # Normalize positions
        positions_norm = positions.float() / self.max_positions * self.position_scale

        # Reshape for processing: (batch * seq_len, 1)
        positions_flat = positions_norm.view(-1, 1)

        # Encode positions
        pos_encoded = self.position_encoder(positions_flat)
        # Shape: (batch*seq_len, dim//2)

        # Apply KAN transformation
        kan_output = self.kan_layer(pos_encoded)  # (batch*seq_len, dim)

        # Reshape back to sequence format
        embeddings = kan_output.view(batch_size, seq_len, self.dim)

        return embeddings


class HybridFourierKANEmbedding(nn.Module):
    """Hybrid embedding combining Fourier features with KAN processing"""

    def __init__(
        self,
        dim: int,
        max_positions: int = 10000,
        num_frequencies: int = 8,
        kan_type: str = "fast",
    ):
        """__init__.

        Args:
            dim (int): Description.
            max_positions (int): Description.
            num_frequencies (int): Description.
            kan_type (str): Description.
        """
        super().__init__()
        if not KAN_AVAILABLE:
            raise ImportError("KAN layers not available. Install kan_convs package.")

        self.dim = dim
        self.max_positions = max_positions
        self.num_frequencies = num_frequencies

        # Fourier basis parameters
        self.frequencies = nn.Parameter(
            torch.randn(num_frequencies // 2) * 0.1,
            requires_grad=True,
        )

        # KAN processing layers
        fourier_dim = num_frequencies
        if kan_type == "fast":
            self.kan_processor = FastKAN(layers_hidden=[fourier_dim, dim], grid_size=8)
        elif kan_type == "chebyshev":
            self.kan_processor = ChebyKANLayer(
                input_dim=fourier_dim,
                output_dim=dim,
                degree=8,
            )
        else:
            self.kan_processor = KANLayer(
                input_features=fourier_dim,
                output_features=dim,
                grid_size=8,
            )

        # Position normalization
        self.position_scale = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, positions):
        """Args:
            positions: Tensor of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, dim)

        forward method for HybridFourierKANEmbedding.

        Executes PyTorch tensor operations.

        Args:
            positions (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # Add batch dimension

        batch_size, seq_len = positions.shape

        # Normalize positions
        positions_norm = positions.float() / self.max_positions * self.position_scale

        # Generate Fourier features for each position
        fourier_features = []
        for i in range(self.num_frequencies // 2):
            freq = self.frequencies[i]
            sin_feat = torch.sin(2 * torch.pi * freq * positions_norm)
            cos_feat = torch.cos(2 * torch.pi * freq * positions_norm)
            fourier_features.extend([sin_feat, cos_feat])

        # Stack all features: (batch, seq_len, num_frequencies)
        fourier_tensor = torch.stack(fourier_features, dim=-1)

        # Reshape for KAN processing: (batch * seq_len, num_frequencies)
        fourier_flat = fourier_tensor.view(-1, self.num_frequencies)

        # Apply KAN transformation
        kan_output = self.kan_processor(fourier_flat)  # (batch*seq_len, dim)

        # Reshape back to sequence format
        embeddings = kan_output.view(batch_size, seq_len, self.dim)

        return embeddings


class KANTimeEmbedding(nn.Module):
    """KAN-based time embedding using advanced KAN layers"""

    def __init__(self, dim: int, max_timesteps: int, kan_type: str = "fast"):
        """__init__.

        Args:
            dim (int): Description.
            max_timesteps (int): Diffusion horizon the positional code spans.
            kan_type (str): Description.
        """
        super().__init__()
        if KAN_AVAILABLE:
            self.fourier_kan = AdvancedKANPositionalEmbedding(
                dim=dim,
                max_positions=max_timesteps,
                kan_type=kan_type,
            )
        else:
            # Fallback to original Fourier implementation
            self.fourier_kan = FourierKANPositionalEmbedding(
                dim=dim,
                max_positions=max_timesteps,
                num_frequencies=min(16, dim // 4),
            )

    def forward(self, time):
        """Args:
            time: Tensor of shape (batch_size,) containing timestep values
        Returns:
            embeddings: Tensor of shape (batch_size, dim)

        forward method for KANTimeEmbedding.

        Executes PyTorch tensor operations.

        Args:
            time (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Convert time to position indices
        positions = time.float().unsqueeze(-1)  # (batch,) -> (batch, 1)

        # Get KAN embeddings
        embeddings = self.fourier_kan(positions)  # (batch, 1, dim)

        # Remove sequence dimension since we have single timestep per batch
        return embeddings.squeeze(1)  # (batch, dim)


class CoordinatePositionalEncoding(nn.Module):
    """Fourier positional encoding for coordinates.

    Maps low-dimensional coordinates to high-dimensional features
    using sinusoidal functions at multiple frequencies.

    gamma(x) = [sin(2^0 * pi * x), cos(2^0 * pi * x),
                sin(2^1 * pi * x), cos(2^1 * pi * x), ...]
    """

    def __init__(
        self,
        in_features: int = 3,
        num_frequencies: int = 10,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        """
        Args:
            in_features: Input dimension
            num_frequencies: Number of frequency bands
            include_input: Whether to include raw input in output
            log_sampling: Whether to use log-spaced frequencies
        """
        super().__init__()
        self.in_features = in_features
        self.num_frequencies = num_frequencies
        self.include_input = include_input

        if log_sampling:
            # freq_bands: [1, 2, 4, 8, ..., 2^(L-1)]
            freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        else:
            freq_bands = torch.linspace(1, 2 ** (num_frequencies - 1), num_frequencies)

        self.register_buffer("freq_bands", freq_bands)

        # Calculate output dimension
        self.out_features = in_features * num_frequencies * 2
        if include_input:
            self.out_features += in_features

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [..., in_features] coordinates

        Returns:
            encoded: [..., out_features] positional encoding

        forward method for CoordinatePositionalEncoding.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # coords: [..., D]
        # freq_bands: [L]

        # Expand for broadcasting: [..., D, 1] * [L] -> [..., D, L]
        scaled = coords.unsqueeze(-1) * self.freq_bands * math.pi

        # Sin and cos: [..., D, L] -> [..., D, 2L]
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)

        # Flatten: [..., D, 2L] -> [..., D*2L]
        encoded = encoded.flatten(start_dim=-2)

        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)

        return encoded

    def output_dim(self, input_dim: int) -> int:
        """Calculate output dimension (helper for consistency)."""
        return self.out_features


__all__ = [
    "AdvancedKANPositionalEmbedding",
    "CoordinatePositionalEncoding",
    "FourierKANPositionalEmbedding",
    "HybridFourierKANEmbedding",
    "KANTimeEmbedding",
    "PatchEmbedding",
    "SinusoidalPositionEmbedding",
]
