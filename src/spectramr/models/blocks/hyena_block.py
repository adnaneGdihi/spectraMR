"""Hyena Operator (PR-22 / Part-I D).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-22.

Reference: Poli et al. 2023, "Hyena Hierarchy: Towards Larger
Convolutional Language Models." Hyena replaces self-attention with a
recursive structure of long convolutions + element-wise gating. Each
"order" of the recursion expands the receptive field exponentially,
giving sub-quadratic compute with attention-comparable performance.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@functools.lru_cache(maxsize=16)
def _positional_encoding(L: int, d: int, device: torch.device) -> torch.Tensor:
    """Sinusoidal time-positional encoding of length L, dim d (cached per shape and device; read-only)."""
    pe = torch.zeros(L, d, device=device)
    pos = torch.arange(L, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class HyenaFilter(nn.Module):
    """Implicit long convolution kernel parameterised by an MLP over
    sinusoidal positional embeddings."""

    def __init__(self, d_model: int, d_filter: int = 64, num_layers: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        layers: list[nn.Module] = [nn.Linear(d_filter, d_model)]
        for _ in range(num_layers):
            layers += [nn.SiLU(), nn.Linear(d_model, d_model)]
        self.mlp = nn.Sequential(*layers)
        self.d_filter = d_filter

    def forward(self, L: int, device: torch.device) -> torch.Tensor:
        pe = _positional_encoding(L, self.d_filter, device)
        return self.mlp(pe).t()  # (d_model, L)


class HyenaOperator(nn.Module):
    """Hyena recursive operator.

    Args:
        d_model: per-channel dimension.
        order: recursion order (typically 2 or 3). order=N means
            (N+1) gating projections + N implicit long convolutions.
        d_filter: hidden dim of the implicit-filter MLP.
        max_seq_len: budget for the cached positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        order: int = 2,
        d_filter: int = 64,
        max_seq_len: int = 8192,
    ) -> None:
        super().__init__()
        if order < 1:
            raise ValueError(f"Hyena order must be ≥ 1, got {order}")
        self.d_model = d_model
        self.order = order
        self.in_proj = nn.Linear(d_model, d_model * (order + 1))
        self.filters = nn.ModuleList(
            [HyenaFilter(d_model, d_filter=d_filter) for _ in range(order)]
        )
        self.short_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.max_seq_len = max_seq_len

    @staticmethod
    def _fft_conv(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """FFT convolution: u (B, D, L) with K (D, L) → (B, D, L)."""
        L = u.shape[-1]
        K_pad = F.pad(K, (0, L))
        u_pad = F.pad(u, (0, L))
        K_f = torch.fft.rfft(K_pad)
        u_f = torch.fft.rfft(u_pad, dim=-1)
        return torch.fft.irfft(u_f * K_f.unsqueeze(0), n=2 * L)[..., :L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)."""
        B, L, D = x.shape
        if self.max_seq_len < L:
            raise ValueError(
                f"Hyena seq length {L} > max_seq_len {self.max_seq_len}. "
                f"Increase max_seq_len at construction."
            )
        # Project to (order + 1) parallel streams.
        u = self.in_proj(x)  # (B, L, D * (order + 1))
        u = u.transpose(-1, -2)  # (B, D*(order+1), L)
        u = u.view(B, self.order + 1, D, L)
        v = u[:, 0]  # value stream
        # Apply local depthwise conv to v (Hyena's "short conv").
        v = self.short_conv(v)
        # Recursively gate: v_n = (g_n * (h_n * v_{n-1}))
        for n in range(self.order):
            g = u[:, n + 1]
            K = self.filters[n](L, x.device)  # (D, L)
            v = g * self._fft_conv(v, K)
        v = v.transpose(-1, -2)  # (B, L, D)
        return self.out_proj(v)


__all__ = ["HyenaFilter", "HyenaOperator"]
