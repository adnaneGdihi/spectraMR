r"""K-Space Radial Swin Attention Block.

Applies self-attention in the Fourier domain by grouping tokens
radially (by frequency band) rather than spherically or via
standard Cartesian sliding windows.

Standard Swin Transformers use rectangular grid windows, which in
k-space arbitrarily slice high and low frequencies into the same or
disconnected windows. Because k-space energy follows an inverse
power law radially, tokens must attend to frequencies of similar scale.

This block divides k-space into concentric rings (radial bands) and
restricts self-attention within each ring, followed by cross-band
communication in the MLP or shifted radial bands.

Reference:
    Liu et al., "Swin Transformer", *ICCV* (2021) — adapted for
    Fourier domains.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RadialWindowAttention(nn.Module):
    """Radial window based multi-head self-attention (W-MSA).

    Sorts spatial tokens by their distance from the k-space center (DC),
    chunks them into equal-sized "radial windows", and applies standard
    multi-head self-attention within those windows.

    This ensures that high frequencies attend to high frequencies, and
    low frequencies attend to low frequencies, maintaining physical
    consistency with MRI k-space properties.

    Args:
        dim: Number of input channels.
        window_size: Number of tokens per radial window.
        num_heads: Number of attention heads.
        qkv_bias: If True, add a learnable bias to Q, K, V.
        attn_drop: Dropout ratio of attention weight.
        proj_drop: Dropout ratio of output.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 64,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Buffer for radial sorting indices
        self.register_buffer("radial_indices", None, persistent=False)
        self.register_buffer("inverse_indices", None, persistent=False)
        self._cached_shape: tuple[int, int] = (0, 0)
        self._pad_masks: dict[tuple[int, torch.device], Tensor] = {}

    def _compute_radial_indices(self, H: int, W: int, device: torch.device) -> None:
        """Compute sorting indices based on distance from k-space center."""
        if self._cached_shape == (H, W) and self.radial_indices is not None:
            if self.radial_indices.device == device:
                return

        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing="ij",
        )

        # Center coordinates
        center_y, center_x = H / 2.0, W / 2.0

        # Radial distance from DC (assuming DC is centered via fftshift)
        r = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

        # Flatten and sort
        r_flat = r.flatten()
        sort_idx = torch.argsort(r_flat)
        inv_idx = torch.argsort(sort_idx)

        self.radial_indices = sort_idx
        self.inverse_indices = inv_idx
        self._cached_shape = (H, W)

    def _padding_mask(self, pad_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Additive ``(ws, ws)`` mask of the last window; padding attends to padding only."""
        key = (pad_len, device)
        mask = self._pad_masks.get(key)
        if mask is None:
            mask = torch.zeros(self.window_size, self.window_size, device=device)
            mask[-pad_len:, :] = float("-inf")
            mask[:, -pad_len:] = float("-inf")
            mask[-pad_len:, -pad_len:] = 0.0
            self._pad_masks[key] = mask
        return mask.to(dtype)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """Apply radial attention.

        Args:
            x: Input tensor ``(B, L, C)`` where L = H * W.
            H: Height of k-space grid.
            W: Width of k-space grid.

        Returns:
            Output tensor ``(B, L, C)``.
        """
        B, L, C = x.shape
        assert L == H * W, "Input sequence length must match H*W"

        self._compute_radial_indices(H, W, x.device)

        # Pad sequence if not divisible by window_size
        pad_len = (self.window_size - L % self.window_size) % self.window_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(B, pad_len, C, device=x.device)], dim=1)

        L_padded = x.shape[1]
        num_windows = L_padded // self.window_size

        # For padded elements, we just leave them at the end.
        # But we must sort the original L elements first.
        x_orig = x[:, :L, :]
        x_pad = x[:, L:, :]

        # Sort spatially by radius
        x_sorted = x_orig[:, self.radial_indices, :]

        # Recombine to padded
        x_sorted_padded = torch.cat([x_sorted, x_pad], dim=1)

        # Reshape to windows: (B, num_windows, window_size, C)
        x_windows = x_sorted_padded.view(B, num_windows, self.window_size, C)

        # Attention
        qkv = (
            self.qkv(x_windows)
            .reshape(B, num_windows, self.window_size, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nW, nH, ws, d)

        dropout_p = self.attn_drop.p if self.training else 0.0

        def _sdpa(q_, k_, v_, mask=None):
            # (B, nW, nH, ws, d) -> fold windows into batch for the fused kernel.
            n = q_.shape[1]
            q_, k_, v_ = (t.reshape(B * n, self.num_heads, self.window_size, -1) for t in (q_, k_, v_))
            out = F.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=mask, dropout_p=dropout_p, scale=self.scale
            )
            return out.reshape(B, n, self.num_heads, self.window_size, -1)

        if pad_len > 0:
            # Only the last window holds padding: it attends with a mask, the rest do not.
            out_last = _sdpa(
                q[:, -1:], k[:, -1:], v[:, -1:], self._padding_mask(pad_len, x.device, q.dtype)
            )
            if num_windows > 1:
                out_head = _sdpa(q[:, :-1], k[:, :-1], v[:, :-1])
                attn_out = torch.cat([out_head, out_last], dim=1)
            else:
                attn_out = out_last
        else:
            attn_out = _sdpa(q, k, v)

        x_out = attn_out.transpose(2, 3).reshape(B, num_windows, self.window_size, C)

        # Merge windows
        x_out = x_out.view(B, L_padded, C)

        # Split original vs padded
        x_out_orig = x_out[:, :L, :]

        # Unsort (inverse permutation)
        x_out_unsorted = x_out_orig[:, self.inverse_indices, :]

        # Project
        x_out = self.proj(x_out_unsorted)
        x_out = self.proj_drop(x_out)

        return x_out


class KSpaceRadialSwinBlock(nn.Module):
    """K-Space Radial Swin Transformer Block.

    Applies alternating radial Window Attention (R-WMSA) and
    Shifted Radial Window Attention (SR-WMSA) to frequency data.

    Shifted radial attention simply rolls the sorted 1D sequence
    by window_size // 2, allowing cross-band communication between
    adjacent frequency rings.

    Args:
        dim: Input channels.
        num_heads: Number of attention heads.
        window_size: Window size (number of tokens per radial band).
        shift_size: Shift size for SR-WMSA. Usually window_size // 2.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 64,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = RadialWindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """Apply radial Swin block.

        Args:
            x: Input ``(B, L, C)``. Complex tensors must be viewed as real
               or concatenated via features.
            H: Height of grid.
            W: Width of grid.

        Returns:
            Output ``(B, L, C)``.
        """
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            # Radial shift: roll across the sequence dimension
            # (which represents frequency magnitudes after sorting)
            # This allows cross-communication between frequency shells
            x = torch.roll(x, shifts=-self.shift_size, dims=1)

        x = self.attn(x, H, W)

        if self.shift_size > 0:
            # Reverse roll
            x = torch.roll(x, shifts=self.shift_size, dims=1)

        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


__all__ = ["KSpaceRadialSwinBlock", "RadialWindowAttention"]
