"""HW-Mamba: a fixed J-level Haar decomposition of the Hilbert sequence, with Mamba-predicted detail corrections.

The approximation band is the ULF's own ``cA_J``, which is not predicted. The Haar
details ``d_1..d_J`` of the ULF sequence are corrected by ``Δd_j``, predicted by Mamba
at each level's resolution, and the output is ``IDWT(cA_J, d_j + Δd_j)``. The
corrections start at zero, so the model begins as the identity on the ULF features.

``wavelet_domain: false`` is the control: the same backbone adds a pixel-domain
residual instead of a wavelet-domain correction.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.grad_checkpointing import NestedGradCheckpointingMixin
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

_SQRT2 = math.sqrt(2.0)


def _haar_level(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One orthonormal Haar analysis step along the last dim: ``[.., L] -> [.., L/2], [.., L/2]``."""
    even, odd = x[..., 0::2], x[..., 1::2]
    return (even + odd) / _SQRT2, (even - odd) / _SQRT2


def _haar_inverse_level(approx: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    """Exact inverse of :func:`_haar_level`, interleaving even and odd samples."""
    even, odd = (approx + detail) / _SQRT2, (approx - detail) / _SQRT2
    return torch.stack([even, odd], dim=-1).flatten(-2)


def haar_dwt(x: torch.Tensor, levels: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """``J``-level Haar DWT. Returns ``(cA_J, [d_1, ..., d_J])``, finest detail first."""
    details: list[torch.Tensor] = []
    approx = x
    for _ in range(levels):
        approx, detail = _haar_level(approx)
        details.append(detail)
    return approx, details


def haar_idwt(approx: torch.Tensor, details: list[torch.Tensor]) -> torch.Tensor:
    """Inverse of :func:`haar_dwt`."""
    for detail in reversed(details):
        approx = _haar_inverse_level(approx, detail)
    return approx


def _zero_linear(d: int) -> nn.Linear:
    layer = nn.Linear(d, d)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


@register_model(
    name="hw_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class HWMamba(NestedGradCheckpointingMixin, nn.Module):
    """Hilbert-wavelet Mamba with a fixed Haar analysis and Mamba-predicted detail corrections.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Feature dimension of the stem and the Mamba backbone.
        num_layers: Mamba blocks in the detail predictor.
        d_state: SSM state dimension.
        wavelet_levels: Haar levels ``J``; the sequence length must be divisible by ``2**J``.
            Default 3 when ``wavelet_domain`` is true. Must be unset when it is false.
        wavelet_domain: True for the wavelet-domain correction, False for the pixel-residual control.
        linearization_mode: Hilbert-family linearization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        wavelet_levels: int | None = None,
        wavelet_domain: bool = True,
        linearization_mode: str = "hilbert_2d",
    ) -> None:
        super().__init__()
        self.wavelet_domain = wavelet_domain
        self.wavelet_levels = 0
        if wavelet_domain:
            levels = 3 if wavelet_levels is None else wavelet_levels
            if not isinstance(levels, int) or levels < 1:
                raise ValueError(f"HWMamba wavelet_levels must be an int >= 1, got {levels!r}.")
            self.wavelet_levels = levels
        elif wavelet_levels is not None:
            raise ValueError("HWMamba wavelet_levels has no effect with wavelet_domain=false.")

        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.detail_predictor = _MambaEncoder(d_model, num_layers, d_state)
        if wavelet_domain:
            self.detail_corrections = nn.ModuleList(
                _zero_linear(d_model) for _ in range(self.wavelet_levels)
            )
        else:
            self.pixel_residual = _zero_linear(d_model)
        self.head = _ConvHead(d_model, out_channels)

    def _wavelet_reconstruct(self, seq: torch.Tensor) -> torch.Tensor:
        """``[B, D, L]`` ULF features to ``[B, D, L]``: ``IDWT(cA_J, d_j + Δd_j)``."""
        length = seq.shape[-1]
        if length % (2**self.wavelet_levels):
            raise ValueError(
                f"HWMamba needs the sequence length divisible by 2**wavelet_levels="
                f"{2**self.wavelet_levels}, got {length}."
            )
        approx, details = haar_dwt(seq, self.wavelet_levels)
        ctx = self.detail_predictor(seq.permute(0, 2, 1))  # [B, L, D]
        corrected = []
        for level, (detail, corr) in enumerate(
            zip(details, self.detail_corrections, strict=True), 1
        ):
            pooled = F.avg_pool1d(ctx.permute(0, 2, 1), kernel_size=2**level)  # [B, D, L/2^j]
            corrected.append(detail + corr(pooled.permute(0, 2, 1)).permute(0, 2, 1))
        return haar_idwt(approx, corrected)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ULF ``[B, C_in, H, W]`` to HF ``[B, C_out, H, W]``."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)  # [B, D, L]
        if self.wavelet_domain:
            recon = self._wavelet_reconstruct(seq)
        else:
            ctx = self.detail_predictor(seq.permute(0, 2, 1))
            recon = seq + self.pixel_residual(ctx).permute(0, 2, 1)

        out = self.linearize.reverse(recon, shape)
        out = self.head(out) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


__all__ = ["HWMamba", "haar_dwt", "haar_idwt"]
