"""INR-Mamba: a Mamba context read out at continuous coordinates, for any output grid.

The ULF grid is linearized in Hilbert order and encoded by Mamba. A query at a
continuous point ``p`` in ``[0, 1]^2`` gets its context from one of two readouts:

* ``hilbert_t`` (the claim): the point's sequence position is the bilinear mix of the
  Hilbert visit indices of its four surrounding cells. Features are linearly
  interpolated along the sequence at that position. At a cell centre the position is an
  integer, so the feature is the cell's own.
* ``spatial`` (control): bilinear ``grid_sample`` of the context map, reshaped from the
  sequence back onto the grid.

An MLP maps ``[context, Fourier(p)]`` to intensities. A bilinear sample of the ULF input
is added as the residual, so the untrained model starts near the input.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.embeddings import CoordinatePositionalEncoding
from spectramr.models.generators.hilbert_mamba_generators import (
    _ConvStem,
    _HilbertLinearize,
    _MambaEncoder,
    _squeeze_volume_to_2d,
    _unsqueeze_to_volume,
)
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

_INTERPS = ("hilbert_t", "spatial")
# Largest exponent of the Fourier bands; beyond it float32 sinusoids alias and their gradients blow up.
_MAX_FREQ_EXP = 10.0


def pixel_centre_coords(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Centres of an ``h x w`` grid as ``[h*w, 2]`` with ``(x, y)`` in ``[0, 1]``."""
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)


@register_model(
    name="inr_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class INRMamba(nn.Module):
    """Implicit-neural-representation Mamba with a Hilbert-parameter or spatial readout.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        num_layers: Mamba blocks for context extraction.
        d_state: SSM state dimension.
        num_freqs: Fourier bands for the coordinate encoding (at most 32 in practice).
        linearization_mode: Hilbert-family linearization.
        interp: ``hilbert_t`` or ``spatial``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        num_freqs: int = 32,
        linearization_mode: str = "hilbert_2d",
        interp: str = "hilbert_t",
    ) -> None:
        super().__init__()
        if interp not in _INTERPS:
            raise ValueError(f"INRMamba interp must be one of {_INTERPS}, got {interp!r}.")
        if num_freqs < 1:
            raise ValueError(f"INRMamba num_freqs must be >= 1, got {num_freqs}.")
        self.interp = interp
        self.out_channels = out_channels
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.context_encoder = _MambaEncoder(d_model, num_layers, d_state)

        # Octave bands capped at 2**_MAX_FREQ_EXP, as the former _FourierPosEnc did.
        self.pos_enc = CoordinatePositionalEncoding(
            in_features=2, num_frequencies=num_freqs, include_input=True, log_sampling=True
        )
        bands = 2.0 ** torch.linspace(0, min(num_freqs - 1, _MAX_FREQ_EXP), num_freqs)
        self.pos_enc.freq_bands.copy_(bands)
        pos_dim = self.pos_enc.out_features

        self.inr_mlp = nn.Sequential(
            nn.Linear(d_model + pos_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )

    def query(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """Predict intensities at continuous coordinates.

        Args:
            x: ULF image ``[B, C_in, H, W]`` (4-D; volumes are squeezed by :meth:`forward`).
            coords: ``[B, N, 2]`` query points, ``(x, y)`` in ``[0, 1]`` (x along W).

        Returns:
            ``[B, C_out, N]``.
        """
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        seq = seq.permute(0, 2, 1)  # [B, L, D]
        context = self.context_encoder(seq)  # [B, L, D]

        if self.interp == "hilbert_t":
            sampled = self._hilbert_readout(context, shape, coords)
        else:
            sampled = self._spatial_readout(context, shape, coords)

        combined = torch.cat([sampled, self.pos_enc(coords)], dim=-1)  # [B, N, D+P]
        intensities = self.inr_mlp(combined).permute(0, 2, 1)  # [B, C_out, N]
        residual = F.grid_sample(
            x[:, : self.out_channels], coords.unsqueeze(2) * 2 - 1, align_corners=False
        ).squeeze(-1)
        return intensities + residual

    def _hilbert_readout(
        self, context: torch.Tensor, shape: tuple[int, ...], coords: torch.Tensor
    ) -> torch.Tensor:
        """Interpolate ``context`` along the sequence at each query's Hilbert position."""
        h, w = shape[-2], shape[-1]
        length = context.shape[1]
        if length < 2:
            raise ValueError("INRMamba hilbert_t readout needs at least two ULF tokens.")
        visit = self.linearize._get_linearizer(shape, coords.device).inverse_idx.reshape(h, w)
        visit = visit.to(context.dtype)

        gx = (coords[..., 0] * w - 0.5).clamp(0.0, w - 1.0)
        gy = (coords[..., 1] * h - 0.5).clamp(0.0, h - 1.0)
        x0, y0 = gx.floor(), gy.floor()
        wx, wy = gx - x0, gy - y0
        x0, y0 = x0.long(), y0.long()
        x1, y1 = (x0 + 1).clamp(max=w - 1), (y0 + 1).clamp(max=h - 1)
        position = (1 - wy) * ((1 - wx) * visit[y0, x0] + wx * visit[y0, x1]) + wy * (
            (1 - wx) * visit[y1, x0] + wx * visit[y1, x1]
        )  # [B, N] fractional sequence position
        k0 = position.floor().clamp(0, length - 2)
        alpha = (position - k0).unsqueeze(-1)
        k0 = k0.long()
        d = context.shape[-1]
        lo = context.gather(1, k0.unsqueeze(-1).expand(-1, -1, d))
        hi = context.gather(1, (k0 + 1).unsqueeze(-1).expand(-1, -1, d))
        return (1 - alpha) * lo + alpha * hi

    def _spatial_readout(
        self, context: torch.Tensor, shape: tuple[int, ...], coords: torch.Tensor
    ) -> torch.Tensor:
        """Bilinear sample of the context map on the grid (LIIF-style control)."""
        ctx_map = self.linearize.reverse(context.permute(0, 2, 1), shape)  # [B, D, H, W]
        grid = coords.unsqueeze(2) * 2 - 1  # [B, N, 1, 2]
        sampled = F.grid_sample(ctx_map, grid, mode="bilinear", align_corners=False)
        return sampled.squeeze(-1).permute(0, 2, 1)  # [B, N, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct on the input grid: ``query`` at the pixel centres."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        b, _, h, w = x.shape
        coords = pixel_centre_coords(h, w, x.device).unsqueeze(0).expand(b, -1, -1)
        out = self.query(x, coords).reshape(b, self.out_channels, h, w)
        return _unsqueeze_to_volume(out, vol_shape)


__all__ = ["INRMamba", "pixel_centre_coords"]
