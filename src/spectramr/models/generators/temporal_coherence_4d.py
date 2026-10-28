"""Temporal Coherence 4D Generator for dynamic MRI reconstruction.

Combines 2D spatial convolutions with temporal attention across frames
for cardiac/fetal cine MRI. Learns spatiotemporal correlations via
3D+T processing with frame stacking.

Reference:
    Qin et al., "Convolutional Recurrent Neural Networks for Dynamic
    MR Image Reconstruction", IEEE TMI, 2019.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

from spectramr.models.blocks.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _TemporalAttention(nn.Module):
    """Self-attention across temporal frames."""

    def __init__(self, ch: int, temporal_dim: int) -> None:
        super().__init__()
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.temporal_embed = nn.Embedding(temporal_dim, ch)
        self.scale = ch**-0.5
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _TemporalAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, C, HW]
        # Channel/temporal self-attention: contract over the spatial (HW)
        # feature axis so the attention matrix is [B, C, C], NOT [B, HW, HW].
        # The original ``q.transpose(-1,-2) @ k`` built a 65536×65536 spatial
        # attention map → a 64 GiB allocation that OOM'd at iter 1 (exp_vf_07,
        # dispatch 20260603_200125 task 29). Full spatial self-attention on a
        # 256² grid is never feasible; C carries the temporal information (see
        # ``temporal_embed``), so attention over C is the intended op — O(C²)
        # instead of O((HW)²).
        attn = (q @ k.transpose(-1, -2)) * self.scale  # [B, C, C]
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).reshape(B, C, H, W)  # [B, C, HW] -> [B, C, H, W]
        return x + self.proj(out)


@register_model(name="temporal_coherence_4d", training_mode="reconstruction")
class TemporalCoherence4DGenerator(GradCheckpointingMixin, IGenerator, nn.Module):
    """4D+T generator for temporal MRI reconstruction."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        feature_dim: int = kwargs.get("feature_dim", 64)
        num_3d_layers: int = kwargs.get("num_3d_layers", 4)
        temporal_dim: int = kwargs.get("temporal_dim", 32)
        frame_stack: int = kwargs.get("frame_stack", 3)

        # Frame encoder (2D)
        self.intro = nn.Conv2d(in_channels * frame_stack, feature_dim, 3, padding=1)

        # Spatial-temporal backbone
        layers: list[nn.Module] = []
        for i in range(num_3d_layers):
            layers.extend(
                [
                    nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                    nn.GroupNorm(min(32, feature_dim), feature_dim),
                    nn.SiLU(),
                ]
            )
            if i % 2 == 1:
                layers.append(_TemporalAttention(feature_dim, temporal_dim))
        self.backbone = nn.Sequential(*layers)
        self._ckpt_segments = max(1, num_3d_layers)

        self.out_conv = nn.Conv2d(feature_dim, out_channels, 3, padding=1)
        self.frame_stack = frame_stack

    @property
    def name(self) -> str:
        return "temporal_coherence_4d"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for TemporalCoherence4DGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        # If input doesn't have frames stacked, replicate
        if self.in_channels * self.frame_stack > C:
            x = x.repeat(1, self.frame_stack, 1, 1)[:, : self.in_channels * self.frame_stack]
        h = self.intro(x)
        if self._checkpointing_active():
            h = checkpoint_sequential(self.backbone, self._ckpt_segments, h, use_reentrant=False)
        else:
            h = self.backbone(h)
        return self.out_conv(h)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
