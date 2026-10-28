r"""Cross-attention along the contrast axis on k-space patches.

Implements §3 of the multi-contrast brainstorm — a transformer block
that treats the contrast axis as a sequence and lets each contrast's
k-space tokens attend across all contrasts. The empirical motivation:

- At low :math:`k_r` (centre of k-space) tokens of one contrast attend
  mostly to themselves — the low frequencies are contrast-defining.
- At high :math:`k_r` (edges) tokens of one contrast attend to
  anatomically informative contrasts — the high frequencies share edge
  support across contrasts (Bilgic 2011 [7]).

The block accepts a tensor of shape ``(B, n_contrasts, C, H, W)`` —
either k-space coefficients (real/imag-as-channels) or image-domain
features. It tokenises by patches (default 8×8), applies multi-head
cross-attention along the contrast axis at each token position, and
returns the same shape.

Token grid is `(H/p) × (W/p)` per sample per contrast; for each token
position we run an attention block over the `n_contrasts` axis. This
is *cheaper* than spatial attention because the attention dimension is
typically 3-5 (number of contrasts in the cohort) rather than 256-1024
(spatial token count).

References
----------
[7] B. Bilgic, V. K. Goyal, E. Adalsteinsson, "Multi-contrast
    reconstruction with Bayesian compressed sensing," *Magn. Reson.
    Med.*, vol. 66, no. 6, 2011.

§3 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn

__all__ = ["MultiContrastKSpaceCrossAttention"]


class MultiContrastKSpaceCrossAttention(nn.Module):
    r"""Cross-attention across the contrast axis on patch tokens.

    Args:
        channels: Per-contrast channel count.
        n_heads: Number of attention heads.
        patch_size: Patch side for tokenisation. ``H`` and ``W`` must be
            divisible by ``patch_size``.
        head_dim: Per-head dimension. Defaults to ``channels * patch_size**2 / n_heads``.
        dropout: Attention dropout probability.

    Forward signature
    -----------------
    ``forward(x: Tensor) -> Tensor`` where ``x`` is
    ``(B, n_contrasts, C, H, W)`` and the return has the same shape.
    """

    def __init__(
        self,
        channels: int,
        n_heads: int = 4,
        patch_size: int = 8,
        head_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")
        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")

        token_dim = channels * patch_size * patch_size
        if head_dim is None:
            if token_dim % n_heads:
                # Round up so head_dim * n_heads >= token_dim.
                head_dim = (token_dim + n_heads - 1) // n_heads
            else:
                head_dim = token_dim // n_heads

        self.channels = channels
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.token_dim = token_dim
        self.attn_dim = head_dim * n_heads

        self.to_q = nn.Linear(token_dim, self.attn_dim)
        self.to_k = nn.Linear(token_dim, self.attn_dim)
        self.to_v = nn.Linear(token_dim, self.attn_dim)
        self.proj = nn.Linear(self.attn_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)
        # Initialise the output projection near zero so the block starts
        # as approximately the identity (residual passthrough).
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()

    def _tokenise(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """``(B, n_c, C, H, W)`` → ``(B*Hp*Wp, n_c, token_dim)`` + shape info."""
        B, n_c, C, H, W = x.shape
        p = self.patch_size
        if H % p or W % p:
            raise ValueError(f"Spatial dims ({H}, {W}) must be divisible by patch_size {p}.")
        Hp, Wp = H // p, W // p
        # Unfold into (B, n_c, C, Hp, p, Wp, p) → (B, Hp, Wp, n_c, C*p*p)
        x = x.view(B, n_c, C, Hp, p, Wp, p)
        x = x.permute(0, 3, 5, 1, 2, 4, 6).contiguous()
        x = x.view(B * Hp * Wp, n_c, C * p * p)
        return x, (B, n_c, Hp, Wp)

    def _untokenise(self, tokens: torch.Tensor, info: tuple[int, int, int, int]) -> torch.Tensor:
        B, n_c, Hp, Wp = info
        p = self.patch_size
        C = self.channels
        x = tokens.view(B, Hp, Wp, n_c, C, p, p)
        x = x.permute(0, 3, 4, 1, 5, 2, 6).contiguous()  # (B, n_c, C, Hp, p, Wp, p)
        return x.view(B, n_c, C, Hp * p, Wp * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected (B, n_contrasts, C, H, W); got {tuple(x.shape)}.")
        if x.shape[2] != self.channels:
            raise ValueError(f"channel mismatch: expected {self.channels}, got {x.shape[2]}.")

        tokens, info = self._tokenise(x)  # (Btok, n_c, token_dim)
        Btok, n_c, _ = tokens.shape
        residual = tokens

        normed = self.norm(tokens)
        q = self.to_q(normed).view(Btok, n_c, self.n_heads, -1).transpose(1, 2)
        k = self.to_k(normed).view(Btok, n_c, self.n_heads, -1).transpose(1, 2)
        v = self.to_v(normed).view(Btok, n_c, self.n_heads, -1).transpose(1, 2)

        # Scaled dot-product attention along the contrast axis.
        # q, k, v: (Btok, n_heads, n_c, head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=getattr(self.dropout, "p", 0.0) if self.training else 0.0
        )  # (Btok, n_heads, n_c, head_dim)
        out = out.transpose(1, 2).contiguous().view(Btok, n_c, self.attn_dim)
        out = self.proj(out)
        return self._untokenise(out + residual, info)
