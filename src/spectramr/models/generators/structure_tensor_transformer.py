"""Structure-Tensor Modulated Transformer Generator.

Implements Pillar 8 of the advanced paradigms for ultra-low field MRI synthesis.
This architecture uses a Swin-like hierarchical structure where the self-attention
matrix is explicitly modulated by the 3D Structure Tensor of the 64mT input volume.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spectramr.models.blocks.structure_tensor_attention import (
    StructureTensorTransformerBlock,
    compute_3d_structure_tensor,
)
from spectramr.models.registry import register_model


class WindowPartition3D(nn.Module):
    def __init__(self, window_size: tuple[int, int, int]):
        super().__init__()
        self.window_size = window_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            windows: (num_windows*B, N, C)
        """
        B, C, D, H, W = x.shape
        Wd, Wh, Ww = self.window_size
        x = x.view(B, C, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww)
        windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        windows = windows.view(-1, Wd * Wh * Ww, C)
        return windows


class WindowReverse3D(nn.Module):
    def __init__(self, window_size: tuple[int, int, int]):
        super().__init__()
        self.window_size = window_size

    def forward(self, windows: Tensor, B: int, D: int, H: int, W: int) -> Tensor:
        """
        Args:
            windows: (num_windows*B, N, C)
            B, D, H, W: Original shapes
        Returns:
            x: (B, C, D, H, W)
        """
        Wd, Wh, Ww = self.window_size
        N = Wd * Wh * Ww
        C = windows.shape[-1]
        x = windows.view(B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, D, H, W)
        return x


class BasicLayer(nn.Module):
    """A basic Structure-Tensor Transformer layer for one stage."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: tuple[int, int, int],
        mlp_ratio: float = 4.0,
        gamma: float = 1.0,
        gamma_trainable: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        self.blocks = nn.ModuleList(
            [
                StructureTensorTransformerBlock(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    gamma=gamma,
                    gamma_trainable=gamma_trainable,
                )
                for _ in range(depth)
            ]
        )

        self.partition = WindowPartition3D(window_size)
        self.reverse = WindowReverse3D(window_size)

    def forward(self, x: Tensor, S: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, D, H, W)
            S: (B, 6, D, H, W)
        """
        B, C, D, H, W = x.shape

        # Partition x and S
        x_windows = self.partition(x)  # (B*num_windows, N, C)
        S_windows = self.partition(S)  # (B*num_windows, N, 6)
        S_windows = S_windows.permute(0, 2, 1)  # (B*num_windows, 6, N)

        for blk in self.blocks:
            x_windows = blk(x_windows, S_windows)

        x = self.reverse(x_windows, B, D, H, W)
        return x


@register_model(name="structure_tensor_transformer", training_mode="reconstruction")
class StructureTensorTransformer(nn.Module):
    """Structure-Tensor Modulated Transformer.

    Extracts the 3D structure tensor from the input and modulates the
    attention map directly to enforce edge-aware deformations.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        embed_dim: int = 64,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (2, 4, 8, 16),
        window_size: tuple[int, int, int] = (4, 4, 4),
        mlp_ratio: float = 4.0,
        gamma: float = 1.0,
        gamma_trainable: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size

        # [FIX] Determine if operating in 2D mode (depth window = 1)
        # Use (1,2,2) kernel/stride for downsampling to preserve depth=1
        is_2d_mode = window_size[0] <= 1
        down_kernel = (1, 2, 2) if is_2d_mode else 2
        down_stride = (1, 2, 2) if is_2d_mode else 2

        # Embed input [B, in_channels, D, H, W] -> [B, embed_dim, D, H, W]
        # We do not patchify aggressively to preserve exact spatial resolution for S mapping
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1)

        self.layers = nn.ModuleList()
        # Scale levels
        for i in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                gamma=gamma,
                gamma_trainable=gamma_trainable,
            )
            self.layers.append(layer)

            # Downsample if not last
            if i < len(depths) - 1:
                self.layers.append(
                    nn.Conv3d(
                        int(embed_dim * 2**i),
                        int(embed_dim * 2 ** (i + 1)),
                        kernel_size=down_kernel,
                        stride=down_stride,
                    )
                )

        # Decoder path
        self.up_layers = nn.ModuleList()
        for i in range(len(depths) - 1, 0, -1):
            self.up_layers.append(
                nn.ConvTranspose3d(
                    int(embed_dim * 2**i),
                    int(embed_dim * 2 ** (i - 1)),
                    kernel_size=down_kernel,
                    stride=down_stride,
                )
            )
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (i - 1)),
                depth=depths[i - 1],
                num_heads=num_heads[i - 1],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                gamma=gamma,
                gamma_trainable=gamma_trainable,
            )
            self.up_layers.append(layer)

        self.final_proj = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """forward method for StructureTensorTransformer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Handle 4D input [B, C, H, W] by adding a depth dimension
        squeeze_output = False
        if x.dim() == 4:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
            squeeze_output = True

        B, C, D, H, W = x.shape

        # [FIX] Determine pool kernel for structure tensor downscaling
        # Must match Conv3d down_kernel to keep spatial dims consistent
        is_2d_mode = self.window_size[0] <= 1 or D == 1
        pool_kernel = (1, 2, 2) if is_2d_mode else 2
        pool_stride = (1, 2, 2) if is_2d_mode else 2

        # 1. Compute Structure Tensor strictly on the input image
        # S has shape [B, 6, D, H, W]
        if torch.is_complex(x):
            # If complex passed, we compute S on magnitude
            x_mag = torch.abs(x)
            S_full = compute_3d_structure_tensor(x_mag)
        else:
            S_full = compute_3d_structure_tensor(x)

        # Optional padding if D, H, W are not divisible by window_size or downsamplings
        # For this prototype we assume dimensions are nicely divisible (e.g., 32x32x32 patches)

        # 2. Embedding
        x_embed = self.patch_embed(x)

        # Encoder
        skips = []
        out = x_embed
        S_curr = S_full

        S_scales = [S_full]

        # Layers: [Layer0, Down0, Layer1, Down1, Layer2, Down2, Layer3]
        for i in range(len(self.layers)):
            if isinstance(self.layers[i], BasicLayer):
                out = self.layers[i](out, S_curr)
                if i < len(self.layers) - 1:  # Not the bottleneck
                    skips.append(out)
            else:
                out = self.layers[i](out)
                # Downsample structure tensor to match spatial dims
                S_curr = F.avg_pool3d(S_curr, kernel_size=pool_kernel, stride=pool_stride)
                S_scales.append(S_curr)

        # Decoder
        S_scales.pop()  # Discard bottleneck S_curr to align with skips
        for i in range(0, len(self.up_layers), 2):
            up = self.up_layers[i]
            layer = self.up_layers[i + 1]

            out = up(out)
            skip = skips.pop()
            out = out + skip  # Residual UNet connection

            S_curr = S_scales.pop()
            out = layer(out, S_curr)

        # Output projection
        out = self.final_proj(out)

        # If input was 4D, squeeze depth dimension back out
        if squeeze_output:
            out = out.squeeze(2)  # [B, C, H, W]

        return out
