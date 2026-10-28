"""FE-Mamba: the ULF is pooled to its native grid, then Kronecker-expanded back to the HF grid.

The input is on the HF target grid. Pooling the Hilbert sequence in blocks of ``s**2``
consecutive tokens gives the ULF's coarse grid, because Hilbert order nests 2D blocks
(tested in ``test_topology_linearizer``). ``expand_proj`` gives each coarse token ``s**2``
children, and the children are the HF tokens. A Mamba smoother runs over the full
sequence, and the output is on the HF grid. ``expansion_factor: 1`` is the identity
grid and reproduces the earlier FE-Mamba control.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from spectramr.models.generators.hilbert_mamba_generators import (
    _ConvHead,
    _ConvStem,
    _HilbertLinearize,
    _MambaEncoder,
    _squeeze_volume_to_2d,
    _unsqueeze_to_volume,
)
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    name="fe_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class FEMamba(nn.Module):
    """Fractal-expansion Mamba: ULF-grid features expanded to HF children.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        num_layers: Mamba blocks in the smoother.
        d_state: SSM state dimension.
        expansion_factor: ``s``, the ratio of HF to ULF grid per side. A power of two.
            The input H and W must be divisible by ``s``.
        linearization_mode: Hilbert-family linearization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        expansion_factor: int = 2,
        linearization_mode: str = "hilbert_2d",
    ) -> None:
        super().__init__()
        s = expansion_factor
        if s < 1 or s & (s - 1):
            raise ValueError(f"FEMamba expansion_factor must be a power of two, got {s}.")
        self.expansion_factor = s
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.expand_proj = nn.Linear(d_model, d_model * s * s)
        self.smoother = _MambaEncoder(d_model, num_layers, d_state)
        self.head = _ConvHead(d_model, out_channels)

        logger.info(
            "[FEMamba] d_model=%d, expansion_factor=%d, mode=%s",
            d_model,
            s,
            linearization_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: HF-grid ULF ``[B, C_in, H, W]`` to ``[B, C_out, H, W]``."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        s = self.expansion_factor
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        if shape[-2] % s or shape[-1] % s:
            raise ValueError(f"FEMamba needs H and W divisible by {s}, got {tuple(shape)}.")
        seq = seq.permute(0, 2, 1)  # [B, L, D]
        b, n, d = seq.shape

        # Pool each block of s^2 Hilbert-consecutive tokens to one ULF token.
        coarse = seq.reshape(b, n // (s * s), s * s, d).mean(dim=2)  # [B, L/s^2, D]
        # Each ULF token expands to its s^2 HF children, which are consecutive in Hilbert order.
        children = self.expand_proj(coarse).reshape(b, n, d)  # [B, L, D]

        smoothed = self.smoother(children).permute(0, 2, 1)  # [B, D, L]
        out = self.linearize.reverse(smoothed, shape)
        out = self.head(out) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


__all__ = ["FEMamba"]
