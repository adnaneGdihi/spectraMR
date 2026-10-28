"""Fourier-KAN-Swin: Swin Transformer with KAN FFN.

Integration of Fourier-KAN into Swin Backbone for MRI Reconstruction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn

from spectramr.models.registry import register_model

from .fourier_kan import KANBlock


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA)."""

    def __init__(self, dim, window_size, num_heads):
        """__init__.

        Args:
            dim (Any): Description.
            window_size (Any): Description.
            num_heads (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for WindowAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Relative position bias logic omitted for brevity in minimal impl
        # Ideally would add relative_position_bias here
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinKANBlock(nn.Module):
    """Swin Transformer Block replacing MLP with KAN."""

    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=4.0):
        """__init__.

        Args:
            dim (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
            shift_size (Any): Description.
            mlp_ratio (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        # Replaced MLP with KANBlock
        self.kan_ffn = KANBlock(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, H: int, W: int):
        """forward.

        Args:
            x (torch.Tensor): Description.
            H (int): Description.
            W (int): Description.
        Returns:
            Any: Description.

        forward method for SwinKANBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            H (int): Expected input tensor.
            W (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # [nW*B, window_size*window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN (KAN)
        x = x + self.kan_ffn(self.norm2(x))

        return x


@register_model(name="fourier_kan_swin", training_mode="reconstruction")
class FourierKANSwin(nn.Module):
    """Fourier-KAN-Swin Model for MRI Reconstruction."""

    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            embed_dim (int): Description.
            num_heads (int): Description.
            num_layers (int): Description.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                SwinKANBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=8,
                    shift_size=0 if (i % 2 == 0) else 4,
                )
            )

        self.norm = nn.LayerNorm(embed_dim)
        self.upsample = nn.ConvTranspose2d(embed_dim, in_channels, kernel_size=1)  # Or PixelShuffle

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for FourierKANSwin.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Pad to multiple of window size (8) if needed
        # Assuming inputs are clean powers of 2 for now or padding handled externally

        x_embed = self.patch_embed(x)
        x_flat = x_embed.permute(0, 2, 3, 1).reshape(B, H * W, self.embed_dim)

        curr_x = x_flat
        for layer in self.layers:
            curr_x = layer(curr_x, H, W)

        curr_x = self.norm(curr_x)
        out_feat = curr_x.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2)

        out = self.upsample(out_feat)
        return out + x  # Residual learning


def create_fourier_kan_swin(
    in_channels: int = 2, embed_dim: int = 64, num_layers: int = 4
) -> FourierKANSwin:
    """create_fourier_kan_swin.

    Args:
        in_channels (int): Description.
        embed_dim (int): Description.
        num_layers (int): Description.
    Returns:
        FourierKANSwin: Description.
    """
    return FourierKANSwin(in_channels, embed_dim, num_layers=num_layers)
