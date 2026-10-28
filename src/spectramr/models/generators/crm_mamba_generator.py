"""CRM-Mamba: a ULF context, one state per coarse parent, that conditions the HF decoder.

The sequence is pooled to one token per 2x2 Hilbert block, encoded at that coarse
resolution, and each state is broadcast to its four children. Hilbert order puts the
children of coarse token ``k`` at fine positions ``4k..4k+3`` (tested in
``test_topology_linearizer``), so the broadcast is a ``repeat_interleave``.

``conditioning_site`` picks how the context enters the decoder:

* ``delta``: a zero-initialised per-layer projection of ``[ctx_j, x_{j-1}]`` shifts
  Δ_j before softplus. Using the decoder input one step back instead of the
  previous output keeps the scan parallel.
* ``input_gate``: the earlier multiplicative gate on the decoder input, kept as the
  control arm.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.delta_mamba import DeltaMambaStack
from spectramr.models.blocks.grad_checkpointing import NestedGradCheckpointingMixin
from spectramr.models.generators.hilbert_mamba_generators import (
    _ConvHead,
    _ConvStem,
    _HilbertLinearize,
    _squeeze_volume_to_2d,
    _unsqueeze_to_volume,
)
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

_SITES = ("delta", "input_gate")
_CHILDREN = 4  # 2D Hilbert: each coarse token covers a 2x2 block of fine tokens


def _pool_children(seq: torch.Tensor) -> torch.Tensor:
    """``[B, L, D]`` fine tokens to ``[B, L/4, D]`` coarse tokens (mean of each block of 4)."""
    b, n, d = seq.shape
    return seq.reshape(b, n // _CHILDREN, _CHILDREN, d).mean(dim=2)


def _broadcast_parents(coarse: torch.Tensor) -> torch.Tensor:
    """``[B, L/4, D]`` coarse states to ``[B, L, D]``, each shared by its four children."""
    return coarse.repeat_interleave(_CHILDREN, dim=1)


def _zero_linear(d_in: int, d_out: int) -> nn.Linear:
    layer = nn.Linear(d_in, d_out)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


@register_model(
    name="crm_mamba",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class CRMMamba(NestedGradCheckpointingMixin, nn.Module):
    """Cross-resolution modulated Mamba with a ULF-coarse context.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        d_model: Mamba model dimension.
        num_layers: Mamba blocks per stack.
        d_state: SSM state dimension.
        linearization_mode: Hilbert-family linearization. Shapes must be even in H and W.
        conditioning_site: ``delta`` or ``input_gate``.
        scan_backend: ``cuda`` (mamba_ssm kernel) or ``reference`` (pure torch, CPU tests only).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        num_layers: int = 4,
        d_state: int = 16,
        linearization_mode: str = "hilbert_2d",
        conditioning_site: str = "delta",
        scan_backend: str = "cuda",
    ) -> None:
        super().__init__()
        if conditioning_site not in _SITES:
            raise ValueError(
                f"CRMMamba conditioning_site must be one of {_SITES}, got {conditioning_site!r}."
            )
        self.conditioning_site = conditioning_site
        self.stem = _ConvStem(in_channels, d_model)
        self.linearize = _HilbertLinearize(mode=linearization_mode)
        self.encoder = DeltaMambaStack(d_model, num_layers, d_state, backend=scan_backend)
        self.decoder = DeltaMambaStack(d_model, num_layers, d_state, backend=scan_backend)
        d_inner = self.decoder.layers[0].d_inner
        if conditioning_site == "delta":
            self.delta_shift_proj = nn.ModuleList(
                _zero_linear(2 * d_model, d_inner) for _ in range(num_layers)
            )
        else:
            self.delta_modulator = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.softplus = nn.Softplus()
        self.head = _ConvHead(d_model, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: ULF ``[B, C_in, H, W]`` to HF ``[B, C_out, H, W]``."""
        x, vol_shape = _squeeze_volume_to_2d(x)
        feat = self.stem(x)
        seq, shape = self.linearize(feat)
        if shape[-2] % 2 or shape[-1] % 2:
            raise ValueError(f"CRMMamba needs even H and W for 2x2 parents, got {tuple(shape)}.")
        seq = seq.permute(0, 2, 1)  # [B, L, D]

        ctx = _broadcast_parents(self.encoder(_pool_children(seq)))  # [B, L, D]
        if self.conditioning_site == "delta":
            prev = F.pad(seq, (0, 0, 1, 0))[:, :-1]  # decoder input one step back; zero at j=0
            cond = torch.cat([ctx, prev], dim=-1)
            shifts = [proj(cond) for proj in self.delta_shift_proj]
            decoded = self.decoder(seq, delta_shift=shifts)
        else:
            gate = self.softplus(self.delta_modulator(torch.cat([ctx, seq], dim=-1)))
            decoded = self.decoder(seq * gate.clamp(max=10.0))

        decoded = decoded.permute(0, 2, 1)  # [B, D, L]
        out = self.linearize.reverse(decoded, shape)
        out = self.head(out) + x[:, : self.head.proj[-1].out_channels]
        return _unsqueeze_to_volume(out, vol_shape)


__all__ = ["CRMMamba"]
