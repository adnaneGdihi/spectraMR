"""Unrolled k-t reconstruction of velocity-encoded multicoil k-space (regime: ``mri_flow``).

One readout slice per sample: k-space ``[B, C, E, T, H, W]`` (complex), its mask
``[B, E, T, H, W]`` and coil maps ``[B, C, H, W]``. Each cascade takes a gradient
step on ``||A x - y||^2`` through the per-encoding SENSE operator ``A = M F S``
(``fft_ops.sense_forward`` / ``sense_adjoint``, which broadcast over encodings and
frames, so every coil, encoding and frame goes through one batched FFT) and adds
a learned (2+1)D correction that sees all encodings at once, so they share
anatomy. The output is the per-encoding complex image ``[B, E, T, H, W]``.
Velocity is decoded from it by the physics (``flow_encoding.referenced_decode_images``),
never predicted, so it stays in m/s without a units contract on this network.
"""

from __future__ import annotations

import math
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.fft_ops import sense_adjoint, sense_forward
from spectramr.models.registry import register_model

#: softplus^-1(1): every cascade starts with a unit gradient step, the largest
#: step that is stable for unit-RSS coil maps and a binary mask (||A|| <= 1).
_UNIT_STEP = math.log(math.e - 1.0)


class _Regulariser(nn.Module):
    """Conv3d stack over ``(T, H, W)`` on the encodings' real/imag channels."""

    def __init__(self, encodings: int, features: int, layers: int) -> None:
        super().__init__()
        channels = [2 * encodings] + [features] * (layers - 1) + [2 * encodings]
        blocks: list[nn.Module] = []
        for i, (c_in, c_out) in enumerate(pairwise(channels)):
            blocks.append(nn.Conv3d(c_in, c_out, kernel_size=3, padding=1))
            if i < layers - 1:
                blocks.append(nn.LeakyReLU(0.1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, e, t, h, w = x.shape
        real = torch.view_as_real(x).permute(0, 1, 5, 2, 3, 4).reshape(b, 2 * e, t, h, w)
        # float(): under autocast the convs return half; the complex view needs float32.
        out = self.net(real).float().reshape(b, e, 2, t, h, w).permute(0, 1, 3, 4, 5, 2)
        return torch.view_as_complex(out.contiguous())


@register_model(
    name="flow_kt_unrolled",
    training_mode="reconstruction",
    # spatial_dims is left undeclared on purpose: the input is [B, C, E, T, H, W],
    # two spatial axes behind two non-spatial ones, which the [B, C, *spatial]
    # dimension contract would read as rank 4. forward() checks the layout itself.
    input_domain="kspace",
    output_domain="image",
    accepts_complex=True,
    expects_real_imag_interleaved=False,
    requires_paired_data=False,
    workflows=frozenset({Regime.FLOW}),
    tasks=frozenset({Task.RECONSTRUCTION}),
    strict_kwargs=True,
)
class FlowKTUnrolled(nn.Module):
    """Gradient-step unrolling with a shared or per-cascade (2+1)D regulariser.

    Args:
        encodings: velocity encodings ``E`` (4 for four-point, 2 for two-point).
        cascades: unrolled iterations.
        features: regulariser width; a multiple of 8 so fp16 convs take the Tensor-Core path.
        layers: convolutions per regulariser (>= 2).
        share_weights: one regulariser for every cascade instead of one each.
        checkpoint_cascades: recompute each cascade in the backward pass, trading
            compute for activation memory.
    """

    #: The generic Tier-2 probe feeds a [B, C, *patch] image; this model acts on a
    #: k-space slice with its mask and coil maps, so the probe would only crash.
    synthetic_forward_probe_skip = frozenset({"forward"})

    def __init__(
        self,
        encodings: int = 4,
        cascades: int = 6,
        features: int = 32,
        layers: int = 3,
        share_weights: bool = False,
        checkpoint_cascades: bool = False,
    ) -> None:
        super().__init__()
        if encodings < 2 or cascades < 1 or layers < 2:
            raise ValueError(
                f"need encodings >= 2, cascades >= 1, layers >= 2; got {encodings}, {cascades}, {layers}"
            )
        if features % 8:
            raise ValueError(f"features must be a multiple of 8, got {features}")
        self.encodings = encodings
        self.cascades = cascades
        self.checkpoint_cascades = checkpoint_cascades
        n_regs = 1 if share_weights else cascades
        self.regularisers = nn.ModuleList(
            _Regulariser(encodings, features, layers) for _ in range(n_regs)
        )
        self.step = nn.Parameter(torch.full((cascades,), _UNIT_STEP))

    def _cascade(
        self, k: int, x: torch.Tensor, y: torch.Tensor, maps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        residual = sense_forward(x[:, None], maps, mask) - y
        grad = sense_adjoint(residual, maps, mask)[:, 0]
        regulariser = self.regularisers[k % len(self.regularisers)]
        return x - F.softplus(self.step[k]) * grad + regulariser(x)

    def forward(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor | None = None,
        coil_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``[B, C, E, T, H, W]`` k-space -> ``[B, E, T, H, W]`` complex images."""
        if mask is None or coil_maps is None:
            raise ValueError(
                "flow_kt_unrolled enforces data consistency, so it needs the sampling "
                "mask [B, E, T, H, W] and coil maps [B, C, H, W] with the k-space"
            )
        if not kspace.is_complex() or kspace.ndim != 6 or kspace.shape[2] != self.encodings:
            raise ValueError(
                f"expected complex k-space [B, C, E={self.encodings}, T, H, W]; "
                f"got {kspace.dtype} {tuple(kspace.shape)}"
            )
        m = mask[:, None].to(kspace.real.dtype)
        maps = coil_maps[:, :, None, None]
        y = kspace * m
        x = sense_adjoint(y, maps, m)[:, 0]
        # The operator is linear, so the network runs on a per-sample unit scale
        # and the result is scaled back; the regularisers never see the raw units.
        scale = x.abs().flatten(1).amax(dim=1).clamp_min(1e-12).view(-1, 1, 1, 1, 1)
        x, y = x / scale, y / scale[:, None]
        for k in range(self.cascades):
            if self.checkpoint_cascades and self.training:
                x = checkpoint(self._cascade, k, x, y, maps, m, use_reentrant=False)
            else:
                x = self._cascade(k, x, y, maps, m)
        return x * scale


__all__ = ["FlowKTUnrolled"]
