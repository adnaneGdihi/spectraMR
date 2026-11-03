"""Contrast-agnostic encoder for multi-contrast contrastive pretraining.

A U-Net encoder path (the ``unet_double_conv`` / ``unet_down`` blocks the ``unet``
backbone is built from), global average pooling, and a SimCLR-style MLP projector
(Chen et al. 2020). The InfoNCE loss is computed on ``projection``; ``embedding``
(the pooled encoder feature) is what a downstream task reuses, and ``features`` is
the encoder's last feature map, the input a frozen-backbone decoder reads.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn

from spectramr.models.blocks.block_registry import create_block
from spectramr.models.registry import register_model

__all__ = ["ContrastAgnosticEncoder"]


def _projector(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    """``depth`` linear layers; BatchNorm + ReLU between them, none after the last."""
    layers: list[nn.Module] = []
    dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
    for i, (a, b) in enumerate(pairwise(dims)):
        layers.append(nn.Linear(a, b))
        if i < depth - 1:
            layers += [nn.BatchNorm1d(b), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


@register_model(
    name="contrast_agnostic_encoder",
    training_mode="ssl",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    requires_paired_data=False,
    strict_kwargs=True,
)
class ContrastAgnosticEncoder(nn.Module):
    """Encoder + projector; ``forward`` returns ``projection``, ``embedding`` and ``features``."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 128,
        spatial_dims: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        projector_depth: int = 2,
        projector_hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        if spatial_dims != 2:
            raise ValueError(f"contrast_agnostic_encoder is 2-D; got spatial_dims={spatial_dims}")
        if depth < 1 or projector_depth < 1 or base_channels < 1:
            raise ValueError(
                f"depth={depth}, projector_depth={projector_depth} and "
                f"base_channels={base_channels} must all be >= 1"
            )
        widths = [base_channels * 2**i for i in range(depth + 1)]
        stages: list[nn.Module] = [
            create_block("unet_double_conv", in_channels=in_channels, out_channels=widths[0])
        ]
        stages += [
            create_block("unet_down", in_channels=a, out_channels=b) for a, b in pairwise(widths)
        ]
        self.encoder = nn.Sequential(*stages)
        self.feature_channels = widths[-1]
        self.projector = _projector(widths[-1], projector_hidden_dim, out_channels, projector_depth)
        self.in_channels, self.out_channels = in_channels, out_channels

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 5 and x.shape[-1] == 1:  # TorchIO's trailing depth axis
            x = x.squeeze(-1)
        features = self.encoder(x)
        embedding = features.mean(dim=(-2, -1))
        return {
            "projection": self.projector(embedding),
            "embedding": embedding,
            "features": features,
        }
