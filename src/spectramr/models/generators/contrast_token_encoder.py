"""Multi-contrast token encoder: B1 of the tissue-map cohort, used as B2's amortised initialiser.

Each contrast gets its own convolutional stem down to 1/4 resolution and a learned contrast
embedding. A contrast marked unavailable is replaced wholesale by a learned mask token, so
none of its values reach the network. Attention is factorised so it fits a 16 GB GPU at
256x256: per location across the (at most three) contrasts, then within non-overlapping
spatial windows per contrast. The three contrast streams are pooled, upsampled and mapped to
the initial tissue-parameter state that B2's unrolled steps refine.

Global attention over 3 x 64 x 64 tokens would need a 12k x 12k attention matrix per head; the
factorised form needs 3 x 3 per location and window^2 x window^2 per window.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ContrastTokenEncoder"]

_N = 3
_DOWN = 4


class _FactorisedBlock(nn.Module):
    """Cross-contrast attention, then windowed spatial attention, each with an MLP."""

    def __init__(self, dim: int, heads: int, window: int) -> None:
        super().__init__()
        self.window = window
        self.norm_c, self.norm_s, self.norm_m = (
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
        )
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.spatial = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        b, n, h, w, d = t.shape  # (B, contrasts, H, W, dim)
        c = t.permute(0, 2, 3, 1, 4).reshape(b * h * w, n, d)
        q = self.norm_c(c)
        c = c + self.cross(q, q, q, need_weights=False)[0]
        t = c.reshape(b, h, w, n, d).permute(0, 3, 1, 2, 4)
        k = self.window
        s = t.reshape(b * n, h // k, k, w // k, k, d).permute(0, 1, 3, 2, 4, 5)
        s = s.reshape(-1, k * k, d)
        q = self.norm_s(s)
        s = s + self.spatial(q, q, q, need_weights=False)[0]
        s = s.reshape(b * n, h // k, w // k, k, k, d).permute(0, 1, 3, 2, 4, 5)
        t = s.reshape(b, n, h, w, d)
        return t + self.mlp(self.norm_m(t))


class ContrastTokenEncoder(nn.Module):
    """``(x (B, 3, H, W), available (B, 3)) -> (B, 3, H, W)`` initial tissue-parameter state."""

    def __init__(self, dim: int = 64, depth: int = 4, heads: int = 4, window: int = 8) -> None:
        super().__init__()
        self.window = window
        self.stems = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, dim // 2, 3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1),
            )
            for _ in range(_N)
        )
        self.contrast_embedding = nn.Parameter(torch.zeros(_N, dim))
        self.mask_token = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.contrast_embedding, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(_FactorisedBlock(dim, heads, window) for _ in range(depth))
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, _N, 1)
        )

    def forward(self, x: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        grid = _DOWN * self.window
        if h % grid or w % grid:
            raise ValueError(f"H and W must be divisible by {grid} (stem x window), got {h}x{w}")
        tokens = torch.stack(
            [self.stems[k](x[:, k : k + 1]).permute(0, 2, 3, 1) for k in range(_N)], dim=1
        )  # (B, 3, H/4, W/4, dim)
        keep = available.view(b, _N, 1, 1, 1) > 0
        # A hidden contrast's tokens are replaced, not scaled: its values never pass, even
        # non-finite ones (0 * inf would be NaN).
        tokens = torch.where(keep, tokens, self.mask_token.to(tokens.dtype))
        tokens = tokens + self.contrast_embedding.view(1, _N, 1, 1, -1)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = tokens.mean(dim=1).permute(0, 3, 1, 2)  # (B, dim, H/4, W/4)
        up = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(up)
