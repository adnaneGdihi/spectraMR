"""Swin-V2 U-Net: a 2D encoder-decoder built from :class:`SwinV2Block` stages.

A 3x3 conv stem, ``L`` Swin-V2 stages joined by stride-2 conv downsampling, a
mirrored decoder with transposed-conv upsampling and concatenated skips, and a
3x3 conv head. Each stage alternates unshifted and half-window-shifted blocks.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spectramr.models.blocks.swin_v2 import SwinV2Block
from spectramr.models.registry import register_model

__all__ = ["SwinV2UNet"]


class _SwinV2Stage(nn.Module):
    """``depth`` Swin-V2 blocks on a ``(B, C, H, W)`` feature map."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinV2Block(
                dim,
                num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
            )
            for i in range(depth)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            tokens = block(tokens, (h, w))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


@register_model(
    name="swin_v2_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
)
class SwinV2UNet(nn.Module):
    """Swin-V2 U-Net for 2D reconstruction / restoration backbones.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        spatial_dims: Must be 2.
        base_channels: Width of the first stage; stage ``i`` has ``base_channels * 2**i``.
        depths: Blocks per stage, one entry per resolution level.
        num_heads: Attention heads per stage; same length as ``depths``.
        window_size: Side of the square attention window.
        mlp_ratio: Hidden width of each block's MLP relative to its dim.
        drop_path_rate: Stochastic-depth rate shared by every block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dims: int = 2,
        base_channels: int = 64,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (2, 4, 8, 16),
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if spatial_dims != 2:
            raise ValueError(f"swin_v2_unet is 2D only; got spatial_dims={spatial_dims}")
        if len(depths) != len(num_heads) or not depths:
            raise ValueError(
                f"depths {tuple(depths)} and num_heads {tuple(num_heads)} must be non-empty "
                "and of equal length"
            )
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1; got {window_size}")
        self.levels = len(depths)
        dims = [base_channels * 2**i for i in range(self.levels)]
        for dim, heads in zip(dims, num_heads, strict=True):
            if dim % heads != 0:
                raise ValueError(f"stage width {dim} does not divide by num_heads={heads}")

        def stage(i: int) -> _SwinV2Stage:
            return _SwinV2Stage(
                dims[i], depths[i], num_heads[i], window_size, mlp_ratio, drop_path_rate
            )

        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1)
        self.encoders = nn.ModuleList(stage(i) for i in range(self.levels))
        self.downs = nn.ModuleList(
            nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2) for i in range(self.levels - 1)
        )
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(dims[i + 1], dims[i], kernel_size=2, stride=2)
            for i in range(self.levels - 1)
        )
        self.fuse = nn.ModuleList(
            nn.Conv2d(2 * dims[i], dims[i], kernel_size=1) for i in range(self.levels - 1)
        )
        self.decoders = nn.ModuleList(stage(i) for i in range(self.levels - 1))
        self.head = nn.Conv2d(dims[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, in_channels, H, W) -> (B, out_channels, H, W)``; ``H, W`` divisible by ``2**(L-1)``."""
        factor = 2 ** (self.levels - 1)
        if x.ndim != 4 or x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(
                f"swin_v2_unet needs a (B, C, H, W) input with H and W divisible by {factor}; "
                f"got shape {tuple(x.shape)}"
            )
        h = self.stem(x)
        skips = []
        for i, encoder in enumerate(self.encoders):
            h = encoder(h)
            if i < self.levels - 1:
                skips.append(h)
                h = self.downs[i](h)
        for i in reversed(range(self.levels - 1)):
            h = self.fuse[i](torch.cat([self.ups[i](h), skips[i]], dim=1))
            h = self.decoders[i](h)
        return self.head(h)
