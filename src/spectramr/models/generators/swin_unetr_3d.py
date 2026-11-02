"""Volumetric Swin UNETR (Hatamizadeh et al., MICCAI BrainLes 2021).

A true-3D Swin UNETR for image-domain ULF->HF super-resolution: shifted-window
self-attention over 3-D windows and Conv3d/ConvTranspose3d encoder-decoder
blocks, so every output voxel sees neighbouring slices. It wraps MONAI's
reference ``SwinUNETR`` with ``spatial_dims=3``.

``monai_swin_unetr`` is a different model: a 2-D ViT encoder that folds depth
into batch, with no cross-slice receptive field. This class is the
depth-matched control for 3-D backbones such as ``geo_mamba_unet``.

MONAI requires every spatial extent to be divisible by ``patch_size ** 5`` (32
at ``patch_size=2``). A 16-slice training patch or an arbitrary validation
window does not meet that, so the wrapper pads each spatial axis up to the next
multiple of 32 with ``pad_mode`` and crops the output back to the input shape.
The padded voxels contribute nothing to the loss.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from torch import nn

from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

_PAD_MODES = frozenset({"replicate", "constant", "reflect"})
# MONAI's decoder is hard-wired to four Swin stages (plus the bottleneck).
_NUM_STAGES = 4
# Patch embedding at stride ``patch_size`` plus four 2x patch-merging steps.
_PATCH_SIZE = 2
_SPATIAL_MULTIPLE = _PATCH_SIZE**5


def _validate(
    feature_size: int,
    depths: Sequence[int],
    num_heads: Sequence[int],
    window_size: int,
    pad_mode: str,
) -> None:
    if feature_size <= 0 or feature_size % 12 != 0:
        raise ValueError(
            f"SwinUNETR3D feature_size must be a positive multiple of 12, got {feature_size}."
        )
    if len(depths) != _NUM_STAGES or len(num_heads) != _NUM_STAGES:
        raise ValueError(
            f"SwinUNETR3D needs exactly {_NUM_STAGES} stages: got depths={list(depths)}, "
            f"num_heads={list(num_heads)}."
        )
    if any(int(d) < 1 for d in depths) or any(int(h) < 1 for h in num_heads):
        raise ValueError(f"SwinUNETR3D depths and num_heads must be >= 1: {depths}, {num_heads}.")
    for stage, heads in enumerate(num_heads):
        width = feature_size * 2**stage
        if width % int(heads) != 0:
            raise ValueError(
                f"SwinUNETR3D stage {stage} width {width} is not divisible by "
                f"num_heads[{stage}]={heads}."
            )
    if window_size < 1:
        raise ValueError(f"SwinUNETR3D window_size must be >= 1, got {window_size}.")
    if pad_mode not in _PAD_MODES:
        raise ValueError(
            f"SwinUNETR3D pad_mode must be one of {sorted(_PAD_MODES)}, got {pad_mode!r}."
        )


@register_model(
    name="monai_swin_unetr_3d",
    training_mode="reconstruction",
    spatial_dims=(3,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class SwinUNETR3D(nn.Module):
    """True-3D Swin UNETR regressor on ``[B, C, X, Y, Z]`` volumes.

    Args:
        in_channels: Input channels.
        out_channels: Output channels. The head is linear (no activation).
        feature_size: Stage-0 embedding width; a multiple of 12 (paper: 48).
        depths: Swin blocks per stage, four stages.
        num_heads: Attention heads per stage, four stages.
        window_size: Cubic attention window edge (paper: 7).
        drop_rate: Dropout after the patch embedding.
        attn_drop_rate: Attention dropout.
        dropout_path_rate: Stochastic-depth rate.
        use_checkpoint: Activation checkpointing inside the Swin stages.
        use_v2: MONAI's SwinUNETR-v2 (a residual conv block before each stage).
        pad_mode: How each axis is padded up to a multiple of 32.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: int = 7,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        use_v2: bool = False,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()
        depths = tuple(int(d) for d in depths)
        num_heads = tuple(int(h) for h in num_heads)
        _validate(feature_size, depths, num_heads, window_size, pad_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_mode = pad_mode
        self.swin = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=_PATCH_SIZE,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            use_v2=use_v2,
        )
        logger.info(
            "[SwinUNETR3D] feature_size=%d depths=%s heads=%s window=%d pad_mode=%s v2=%s",
            feature_size,
            depths,
            num_heads,
            window_size,
            pad_mode,
            use_v2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, C_in, X, Y, Z]`` to ``[B, C_out, X, Y, Z]``."""
        if x.ndim != 5:
            raise ValueError(
                f"SwinUNETR3D expects a 5-D volume [B, C, X, Y, Z], got {tuple(x.shape)}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"SwinUNETR3D was built for {self.in_channels} input channels, got {x.shape[1]}."
            )
        spatial = x.shape[2:]
        pads = [(-s) % _SPATIAL_MULTIPLE for s in spatial]
        if not any(pads):
            return self.swin(x)
        # F.pad lists the last axis first, (left, right) per axis.
        pad_arg: list[int] = []
        for p in reversed(pads):
            pad_arg += [p // 2, p - p // 2]
        out = self.swin(F.pad(x, pad_arg, mode=self.pad_mode))
        starts = [p // 2 for p in pads]
        return out[
            :,
            :,
            starts[0] : starts[0] + spatial[0],
            starts[1] : starts[1] + spatial[1],
            starts[2] : starts[2] + spatial[2],
        ]


__all__ = ["SwinUNETR3D"]
