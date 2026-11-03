"""Stage 2 of the multi-contrast SSL pipeline: a pretrained encoder, frozen, under a new decoder.

The backbone is the stage-1 ``contrast_agnostic_encoder`` with the same widths, so its
checkpoint loads strictly (``load_backbone_state``, called by ``ModelBuilder`` from
``model_kwargs.backbone_checkpoint``). The head is a U-Net decoder that takes the
encoder's stage outputs as skips; with ``contrast_conditioning`` a learned per-contrast
offset is added at the bottleneck. ``freeze_backbone`` stops the encoder's gradients,
and ``get_parameter_groups`` hands the optimizer only what trains.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.block_registry import create_block
from spectramr.models.generators.contrast_agnostic_encoder import ContrastAgnosticEncoder
from spectramr.models.registry import register_model

__all__ = ["FrozenBackboneWithHead"]

_HEAD_TYPES = ("unet_decoder",)


@register_model(
    name="frozen_backbone_with_head",
    training_mode="gan",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    supports_contrast_conditioning=True,
    requires_paired_data=False,
)
class FrozenBackboneWithHead(nn.Module):
    """Pretrained ``contrast_agnostic_encoder`` (frozen) + U-Net decoder head."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dims: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        projector_depth: int = 3,
        projector_hidden_dim: int = 1024,
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
        head: Mapping[str, Any] | None = None,
        n_contrasts: int = 4,
    ) -> None:
        super().__init__()
        head = dict(head or {"type": "unet_decoder"})
        if head.get("type") not in _HEAD_TYPES:
            raise ValueError(f"head.type {head.get('type')!r} not in {_HEAD_TYPES}")
        unknown = set(head) - {"type", "contrast_conditioning"}
        if unknown:
            raise ValueError(
                f"head keys {sorted(unknown)} are not read; the head is sized by the encoder"
            )
        self.backbone = ContrastAgnosticEncoder(
            in_channels,
            embedding_dim,
            spatial_dims,
            base_channels,
            depth,
            projector_depth,
            projector_hidden_dim,
        )
        widths = [base_channels * 2**i for i in range(depth + 1)]
        self.ups = nn.ModuleList(
            create_block("unet_up", in_channels=b, out_channels=a, bilinear=False)
            for a, b in reversed(list(pairwise(widths)))
        )
        self.out_conv = nn.Conv2d(widths[0], out_channels, kernel_size=1)
        self.contrast_table = (
            nn.Embedding(n_contrasts, widths[-1]) if head.get("contrast_conditioning") else None
        )
        if self.contrast_table is not None:
            nn.init.zeros_(self.contrast_table.weight)
        self.backbone_frozen = bool(freeze_backbone)
        if self.backbone_frozen:
            self.backbone.requires_grad_(False)

    def load_backbone_state(self, state: Mapping[str, torch.Tensor], *, source: str) -> None:
        """Load the stage-1 encoder's weights; a key mismatch raises (strict)."""
        try:
            self.backbone.load_state_dict(dict(state), strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{source} is not a matching contrast_agnostic_encoder: {exc}"
            ) from exc
        if self.backbone_frozen:
            self.backbone.requires_grad_(False)

    def get_parameter_groups(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def train(self, mode: bool = True) -> FrozenBackboneWithHead:
        super().train(mode)
        if self.backbone_frozen:  # frozen statistics too, not only frozen weights
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor, *, contrast_idx: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        skips = []
        h = x
        for stage in self.backbone.encoder:
            h = stage(h)
            skips.append(h)
        if self.contrast_table is not None:
            if contrast_idx is None:
                raise ValueError("head.contrast_conditioning needs contrast_idx per sample")
            h = h + self.contrast_table(contrast_idx.reshape(-1))[:, :, None, None]
        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            h = up(h, skip)
        return self.out_conv(h)
