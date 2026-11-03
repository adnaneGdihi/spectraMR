"""Stationary-velocity-field head for diffeomorphic registration (Arsigny et al. 2006).

Maps a moving/fixed image pair, concatenated on the channel axis, to a 2-D stationary
velocity field ``v`` in pixel units. ``exp(v)`` is integrated by the consumer
(scaling-and-squaring, Ashburner 2007), so the head itself stays a plain conv stack.
The last convolution starts at zero: the first step integrates ``exp(0) = id`` and the
warp is learned away from the identity rather than from a random field.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.blocks.block_registry import register_block
from spectramr.models.blocks.interfaces import IBlockStrategy


@register_block("velocity_field_head")
class VelocityFieldHead(IBlockStrategy):
    """``(B, 2C, H, W)`` moving/fixed pair -> ``(B, 2, H, W)`` velocity (x, y) in pixels."""

    def __init__(self, in_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        if in_channels < 2 or in_channels % 2:
            raise ValueError(
                f"velocity_field_head takes a moving/fixed pair, so in_channels must be an "
                f"even number >= 2; got {in_channels}."
            )
        if hidden_channels < 1:
            raise ValueError(f"hidden_channels must be >= 1; got {hidden_channels}.")
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, 3, padding=1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"velocity_field_head is 2-D: expected (B, C, H, W), got shape {tuple(x.shape)}."
            )
        return self.net(x)


__all__ = ["VelocityFieldHead"]
