"""Learnable per-vendor / per-protocol soft-prompt embeddings.

Each scanner / acquisition protocol gets a learnable embedding
initialised to **zero**, so the model starts from the physics-only
baseline (i.e. whatever ``AcquisitionEmbedding`` already produces) and
can only *add* vendor-specific deviations as it trains. This lets a
single model absorb vendor / protocol quirks (gradient non-linearities,
B1+ inhomogeneity patterns, scanner-specific reconstruction biases)
without baking them into the architecture.

Multi-contrast Item 4 (``TODO/backlog_multi_contrast_remaining.md``).

The schema side (``data.multi_contrast.vendor_map`` /
``data.multi_contrast.n_vendors``) is already declared on
:class:`spectramr.config.schemas.data.MultiContrastConfigSchema` and the
audit guard ``check_vendor_model_support`` (see
``infrastructure/validation/audit_checks``) rejects YAMLs that enable
the vendor map for a model that does not advertise vendor support.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VendorPromptEmbedding(nn.Module):
    """Compose an :class:`AcquisitionEmbedding` output with a vendor offset.

    Args:
        n_vendors: Cohort cardinality (size of the lookup table).
        embed_dim: Output embedding dimension, must match the
            ``AcquisitionEmbedding`` output dim so the sum is well-defined.
        zero_init: When True (the default) the vendor lookup is
            zero-initialised. The combined embedding therefore equals the
            physics-only baseline at t=0; vendor offsets are learned only
            as the model trains. Set to False to bootstrap with random
            init (useful when the cohort is fully balanced and a non-zero
            prior is desirable).

    The combined output is::

        out[b] = acquisition_embedding[b] + vendor_embedding(vendor_id[b])

    where ``vendor_id`` is a long tensor of shape ``(B,)``.
    """

    def __init__(
        self,
        n_vendors: int,
        embed_dim: int,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if n_vendors < 1:
            raise ValueError(
                f"VendorPromptEmbedding requires n_vendors >= 1; got {n_vendors}. "
                "Set ``data.multi_contrast.vendor_map`` in the YAML or remove the "
                "vendor block entirely."
            )
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")

        self.n_vendors = n_vendors
        self.embed_dim = embed_dim
        self.vendor_lookup = nn.Embedding(n_vendors, embed_dim)
        if zero_init:
            nn.init.zeros_(self.vendor_lookup.weight)

    def forward(
        self,
        acquisition_embedding: torch.Tensor,
        vendor_id: torch.Tensor,
    ) -> torch.Tensor:
        """Add the vendor offset to the physics-only acquisition embedding.

        Args:
            acquisition_embedding: ``[B, embed_dim]`` output of
                :class:`AcquisitionEmbedding` (or any compatible block).
            vendor_id: ``[B]`` long tensor of vendor / protocol indices.

        Returns:
            ``[B, embed_dim]`` combined embedding.
        """
        if vendor_id.dtype not in (torch.long, torch.int32, torch.int64):
            raise TypeError(f"vendor_id must be a long tensor; got dtype={vendor_id.dtype}")
        if vendor_id.shape[0] != acquisition_embedding.shape[0]:
            raise ValueError(
                f"vendor_id batch ({vendor_id.shape[0]}) does not match "
                f"acquisition_embedding batch ({acquisition_embedding.shape[0]})"
            )
        # On CUDA nn.Embedding's own device assert catches this; a host read here would
        # sync every training step.
        if not vendor_id.is_cuda and int(vendor_id.max()) >= self.n_vendors:
            raise IndexError(
                f"vendor_id contains index {int(vendor_id.max())} >= "
                f"n_vendors={self.n_vendors}. Check ``data.multi_contrast.vendor_map``."
            )
        offset = self.vendor_lookup(vendor_id)
        return acquisition_embedding + offset


__all__ = ["VendorPromptEmbedding"]
