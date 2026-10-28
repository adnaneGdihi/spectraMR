"""Swin Transformer V2 blocks (2D), after Liu et al., CVPR 2022.

Three changes against the V1 block in :mod:`spectramr.models.blocks.swin`:

- **Scaled cosine attention.** ``q`` and ``k`` are L2-normalised and the logit is
  multiplied by a learned per-head temperature clamped at ``log(100)``
  (:func:`~spectramr.models.blocks.attention.qk_normalize`).
- **Log-spaced continuous position bias.** The relative offsets of a window are
  mapped to ``sign(d) * log2(1 + |d|) / log2(8)`` and passed through a two-layer
  MLP, then squashed to ``16 * sigmoid``, instead of indexing a learned table.
- **Res-post-norm.** ``x = x + norm(f(x))`` rather than ``x = x + f(norm(x))``.

Windowing, padding and the shifted-window mask come from
:mod:`spectramr.models.blocks.swin_windows`, the same owner the V1 block uses.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.attention import qk_normalize
from spectramr.models.blocks.swin import MLP, DropPath
from spectramr.models.blocks.swin_windows import (
    ShiftedWindowMaskCache,
    padded_resolution,
    window_count,
    window_partition,
    window_reverse,
)

__all__ = ["SwinV2Block", "SwinV2WindowAttention", "log_spaced_relative_coords"]


def log_spaced_relative_coords(window_size: int) -> torch.Tensor:
    """``(2W-1, 2W-1, 2)`` relative offsets, normalised to ``[-1, 1]`` then log-spaced."""
    span = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
    table = torch.stack(torch.meshgrid(span, span, indexing="ij"), dim=-1)
    if window_size > 1:
        table = table / (window_size - 1)
    table = table * 8.0
    return torch.sign(table) * torch.log2(table.abs() + 1.0) / math.log2(8.0)


def _relative_position_index(window_size: int) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
    ).flatten(1)
    rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (window_size - 1)
    return rel[..., 0] * (2 * window_size - 1) + rel[..., 1]


class SwinV2WindowAttention(nn.Module):
    """Window attention with cosine logits and a continuous position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        cpb_hidden: int = 512,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must divide by num_heads={num_heads}")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.full((num_heads, 1, 1), math.log(10.0)))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, cpb_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cpb_hidden, num_heads, bias=False),
        )
        self.register_buffer(
            "relative_coords_table", log_spaced_relative_coords(window_size), persistent=False
        )
        self.register_buffer(
            "relative_position_index", _relative_position_index(window_size), persistent=False
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def position_bias(self) -> torch.Tensor:
        """``(1, heads, N, N)`` bias, bounded in ``(0, 16)``."""
        n = self.window_size * self.window_size
        table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        bias = table[self.relative_position_index.view(-1)].view(n, n, self.num_heads)
        return (16.0 * torch.sigmoid(bias)).permute(2, 0, 1).unsqueeze(0)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, *, num_windows: int | None = None
    ) -> torch.Tensor:
        """``x``: ``(B*nW, N, C)``; ``mask``: ``(nW, N, N)`` or ``None``."""
        b_, n, c = x.shape
        head_dim = c // self.num_heads
        q, k, v = (
            self.qkv(x).reshape(b_, n, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        )
        q = qk_normalize(q, self.logit_scale)
        k = F.normalize(k, dim=-1)
        bias = self.position_bias()
        if mask is not None:
            n_win = mask.shape[0]
            if (num_windows is not None and n_win != num_windows) or b_ % n_win != 0:
                raise ValueError(
                    f"attention mask describes {n_win} windows but {b_} window-rows "
                    f"({num_windows} expected) were partitioned"
                )
            bias = (bias + mask.unsqueeze(1)).reshape(1, n_win * self.num_heads, n, n)
            q, k, v = (
                t.reshape(b_ // n_win, n_win * self.num_heads, n, head_dim) for t in (q, k, v)
            )
        dropout_p = self.attn_drop.p if self.training else 0.0
        # The temperature is already inside q; the default 1/sqrt(d) would shrink it.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.to(q.dtype), dropout_p=dropout_p, scale=1.0
        )
        out = out.reshape(b_, self.num_heads, n, head_dim).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(out))


class SwinV2Block(nn.Module):
    """Shifted-window block with res-post-norm, on ``(B, H*W, C)`` tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0 <= shift_size < window_size:
            raise ValueError(f"shift_size={shift_size} must lie in [0, window_size={window_size})")
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = SwinV2WindowAttention(
            dim,
            window_size,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_dropout,
            proj_drop=dropout,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self._mask_cache = ShiftedWindowMaskCache()

    def _attend(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, _, c = x.shape
        ws = self.window_size
        grid = x.view(b, height, width, c)
        padded_h, padded_w = padded_resolution(height, width, ws)
        if padded_h != height or padded_w != width:
            grid = F.pad(grid, (0, 0, 0, padded_w - width, 0, padded_h - height))
        mask = None
        if self.shift_size > 0:
            grid = torch.roll(grid, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = self._mask_cache.get(
                height, width, ws, self.shift_size, device=x.device, dtype=x.dtype
            )
        windows = window_partition(grid, ws).view(-1, ws * ws, c)
        windows = self.attn(windows, mask=mask, num_windows=window_count(height, width, ws))
        grid = window_reverse(windows.view(-1, ws, ws, c), ws, padded_h, padded_w)
        if self.shift_size > 0:
            grid = torch.roll(grid, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return grid[:, :height, :width, :].reshape(b, height * width, c)

    def forward(self, x: torch.Tensor, input_resolution: tuple[int, int]) -> torch.Tensor:
        """``x``: ``(B, H*W, C)`` with ``input_resolution == (H, W)``."""
        height, width = input_resolution
        if x.shape[1] != height * width:
            raise ValueError(f"{x.shape[1]} tokens do not tile a {height}x{width} grid")
        x = x + self.drop_path(self.norm1(self._attend(x, height, width)))
        return x + self.drop_path(self.norm2(self.mlp(x)))
