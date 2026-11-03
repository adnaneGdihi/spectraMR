"""Vendor-conditioned image translation: a soft-prompt vendor offset on a contrast embedding.

The contrast id selects an acquisition embedding (``nn.Embedding(n_contrasts, D)``);
:class:`~spectramr.models.blocks.vendor_prompt_embedding.VendorPromptEmbedding` adds the
vendor's learned offset (zero at initialisation, so training starts vendor-agnostic); the
sum conditions every decoder stage of a configurable U-Net through its time-injection
blocks (``contrast_emb``, built with ``time_dim = D``).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from spectramr.models.blocks.vendor_prompt_embedding import VendorPromptEmbedding
from spectramr.models.reconstruction.unet import UNet, UNetConfig
from spectramr.models.registry import register_model

__all__ = ["VendorAwareTranslation"]


@register_model(
    name="vendor_aware_translation",
    training_mode="gan",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    supports_contrast_conditioning=True,
    supports_vendor_conditioning=True,
    strict_kwargs=True,
)
class VendorAwareTranslation(nn.Module):
    """U-Net generator conditioned on ``contrast_idx`` and ``vendor_id``."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dims: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        acquisition_embedding_dim: int = 128,
        n_contrasts: int = 1,
        n_vendors: int = 1,
        vendor_init: Literal["zero", "random"] = "zero",
    ) -> None:
        super().__init__()
        if spatial_dims != 2:
            raise ValueError(f"vendor_aware_translation is 2-D; got spatial_dims={spatial_dims}")
        if vendor_init not in ("zero", "random"):
            raise ValueError(f"vendor_init must be 'zero' or 'random', got {vendor_init!r}")
        if n_contrasts < 1 or depth < 1:
            raise ValueError(f"n_contrasts={n_contrasts} and depth={depth} must be >= 1")
        self.n_contrasts = n_contrasts
        self.contrast_table = nn.Embedding(n_contrasts, acquisition_embedding_dim)
        self.vendor_prompt = VendorPromptEmbedding(
            n_vendors, acquisition_embedding_dim, zero_init=vendor_init == "zero"
        )
        self.backbone = UNet(
            UNetConfig(
                in_channels=in_channels,
                out_channels=out_channels,
                features=tuple(base_channels * 2**i for i in range(depth + 1)),
                depth=depth,
                time_dim=acquisition_embedding_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        contrast_idx: torch.Tensor | None = None,
        vendor_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if vendor_id is None:
            raise ValueError("vendor_aware_translation needs a vendor_id per sample")
        if contrast_idx is None:
            if self.n_contrasts > 1:
                raise ValueError(f"{self.n_contrasts} contrasts declared but no contrast_idx")
            contrast_idx = torch.zeros_like(vendor_id)
        emb = self.vendor_prompt(
            self.contrast_table(contrast_idx.reshape(-1)), vendor_id.reshape(-1)
        )
        return self.backbone(x, contrast_emb=emb)
