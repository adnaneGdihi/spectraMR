"""Depth-split counterparts of ``Conv3d``, ``ConvTranspose3d`` and ``Upsample``.

Each subclasses the original layer and *adopts* an existing instance in place
(``adopt`` switches its class), so parameters, buffers and state-dict keys are the
unsplit model's exactly and a checkpoint loads either way. Only the depth axis is
split; height and width keep the original padding. A geometry this cannot reproduce
exactly raises, because a slab that is silently off by a plane trains a different
network.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.infrastructure.distributed.spatial.halo import exchange_halo, overlap_add


def _depth(value: Any) -> int:
    return value[0] if isinstance(value, tuple) else value


class SpatialConv3d(nn.Conv3d):
    """``Conv3d`` on a depth slab: a halo exchange, then the convolution with no depth pad.

    For stride ``s``, kernel ``k``, dilation ``d`` and depth padding ``p``, output plane
    ``j`` reads input planes ``s*j - p + t*d``. A rank owning ``Dl`` input planes owns
    ``Dl / s`` output planes and needs ``p`` planes below and ``d*(k-1) - p - s + 1`` above
    (a negative count trims its own top planes instead).
    """

    spatial_group: Any

    @classmethod
    def adopt(cls, conv: nn.Conv3d, group: Any) -> SpatialConv3d:
        if conv.padding_mode != "zeros":
            raise ValueError(
                f"spatial Conv3d supports zero padding only, not {conv.padding_mode!r}"
            )
        if isinstance(conv.padding, str):
            raise ValueError("spatial Conv3d needs numeric padding, not padding='same'/'valid'")
        conv.__class__ = cls
        conv.spatial_group = group
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k, s = _depth(self.kernel_size), _depth(self.stride)
        d, p = _depth(self.dilation), _depth(self.padding)
        depth = x.shape[2] * torch.distributed.get_world_size(self.spatial_group)
        out_depth = (depth + 2 * p - d * (k - 1) - 1) // s + 1
        if x.shape[2] % s or out_depth != depth // s:
            raise ValueError(
                f"depth geometry not splittable: {x.shape[2]} planes per rank, stride {s}, "
                f"global output {out_depth} != {depth}/{s}"
            )
        hi = d * (k - 1) - p - s + 1
        if hi < 0:
            x, hi = x[:, :, : x.shape[2] + hi], 0
        x = exchange_halo(x, p, hi, "zeros", self.spatial_group)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            (0, *self.padding[1:]),
            self.dilation,
            self.groups,
        )


class SpatialConvTranspose3d(nn.ConvTranspose3d):
    """``ConvTranspose3d`` on a depth slab: local transpose without depth crop, then overlap-add.

    Supported when the global output depth is ``s`` times the input depth (the usual
    upsampling geometry) with no depth ``output_padding``. Bias is added after the
    overlap-add so spilled planes do not receive it twice.
    """

    spatial_group: Any

    @classmethod
    def adopt(cls, conv: nn.ConvTranspose3d, group: Any) -> SpatialConvTranspose3d:
        k, s, d = _depth(conv.kernel_size), _depth(conv.stride), _depth(conv.dilation)
        p, op = _depth(conv.padding), _depth(conv.output_padding)
        if op or d * (k - 1) - 2 * p + 1 != s:
            raise ValueError(
                f"ConvTranspose3d depth geometry (k={k}, s={s}, d={d}, p={p}, op={op}) does "
                "not upsample by exactly the stride; not splittable"
            )
        conv.__class__ = cls
        conv.spatial_group = group
        return conv

    def forward(self, x: torch.Tensor, output_size: Any = None) -> torch.Tensor:
        if output_size is not None:
            raise ValueError("spatial ConvTranspose3d does not take output_size")
        raw = F.conv_transpose3d(
            x,
            self.weight,
            None,
            self.stride,
            (0, *self.padding[1:]),
            (0, *self.output_padding[1:]),
            self.groups,
            self.dilation,
        )
        p = _depth(self.padding)
        out = overlap_add(raw, p, p, self.spatial_group)
        return out if self.bias is None else out + self.bias.view(1, -1, 1, 1, 1)


class SpatialUpsample(nn.Upsample):
    """``Upsample`` by an integer factor on a depth slab.

    ``nearest`` is local. ``trilinear`` with ``align_corners=False`` reads one plane on
    each side, so it upsamples a slab padded with a one-plane halo (replicated at the
    volume's ends, which is how the unsplit op clamps) and crops ``scale`` planes per side.
    """

    spatial_group: Any

    @classmethod
    def adopt(cls, up: nn.Upsample, group: Any) -> SpatialUpsample:
        scale = up.scale_factor
        scale = scale[0] if isinstance(scale, tuple) else scale
        if up.size is not None or scale is None or float(scale) != int(scale):
            raise ValueError("spatial Upsample needs an integer scale_factor, not a size")
        if up.mode not in ("nearest", "trilinear") or (up.mode == "trilinear" and up.align_corners):
            raise ValueError(
                f"spatial Upsample supports nearest and trilinear(align_corners=False), not {up.mode!r}"
            )
        up.__class__ = cls
        up.spatial_group = group
        return up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "nearest":
            return super().forward(x)
        scale = self.scale_factor
        scale = int(scale[0] if isinstance(scale, tuple) else scale)
        out = super().forward(exchange_halo(x, 1, 1, "replicate", self.spatial_group))
        return out[:, :, scale : out.shape[2] - scale]


__all__ = ["SpatialConv3d", "SpatialConvTranspose3d", "SpatialUpsample"]
