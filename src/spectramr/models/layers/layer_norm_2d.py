"""Channels-first LayerNorm for NCHW feature maps."""

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel axis of ``[B, C, H, W]``.

    Matches ``timm.layers.LayerNorm2d`` (eps 1e-6, affine) without importing
    timm, which costs ~2 s at registry-walk time.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)
