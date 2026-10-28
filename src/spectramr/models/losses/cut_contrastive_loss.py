r"""Patch-wise InfoNCE for unpaired translation (CUT).

Implements the *Contrastive Unpaired Translation* (CUT) loss: positive
pairs come from spatially co-located patches in source :math:`\mathbf{x}_A`
and translated :math:`G(\mathbf{x}_A)` feature maps; negatives come from
*other* spatial locations in the same source feature map. This forces
each translated patch to retain the content of its source patch while
remaining free to alter style.

For each query patch :math:`q` and its positive :math:`k_+`,

.. math::
    \mathcal{L}_{\text{CUT}} = -\log\frac{\exp(\mathrm{sim}(q,k_+)/\tau)}
        {\sum_k \exp(\mathrm{sim}(q,k)/\tau)}.

The features are typically taken from intermediate layers of the
encoder portion of the translator. This loss therefore expects the
encoder features to be supplied via ``context`` (``feat_source`` and
``feat_translated``, both lists of ``(B, C, H, W)`` tensors at the same
scales). A small projection head (per-pixel MLP) is applied internally.

Reference:
    T. Park, A. A. Efros, R. Zhang, J.-Y. Zhu, "Contrastive learning for
    unpaired image-to-image translation," *Proc. ECCV*, 2020.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


class _PatchSampleMLP(nn.Module):
    """Per-feature-pixel projection head used by CUT.

    Applied independently to each spatial location of each feature map.
    A new linear-relu-linear head is lazily created the first time a
    feature map of a given channel count is observed.
    """

    def __init__(self, mlp_dim: int = 256) -> None:
        super().__init__()
        self.mlp_dim = mlp_dim
        self.heads: nn.ModuleDict = nn.ModuleDict()

    def _get_head(self, in_channels: int, device: torch.device) -> nn.Module:
        key = str(in_channels)
        if key not in self.heads:
            self.heads[key] = nn.Sequential(
                nn.Linear(in_channels, self.mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.mlp_dim, self.mlp_dim),
            ).to(device)
        return self.heads[key]

    def forward(self, feature: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        # feature: (B, C, H, W); indices: (B, S) spatial positions in [0, H*W)
        b, c, h, w = feature.shape
        flat = feature.reshape(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        gathered = torch.gather(flat, 1, indices.unsqueeze(-1).expand(-1, -1, c))  # (B, S, C)
        head = self._get_head(c, feature.device)
        proj: torch.Tensor = head(gathered)  # (B, S, mlp_dim)
        return F.normalize(proj, dim=-1)


@register_loss(
    name="cut_patch_nce",
    aliases=["CUTLoss", "PatchNCELoss", "patch_nce"],
    domain="image",
)
class CUTPatchNCELoss(nn.Module):
    r"""Patch-NCE contrastive loss (CUT).

    Forward signature deliberately ignores ``prediction`` / ``target`` —
    everything flows via the ``context`` dict. Strategies wiring CUT
    must populate:

    * ``context["feat_source"]``: list[Tensor] feature maps from the
      *source* image at each chosen layer.
    * ``context["feat_translated"]``: list[Tensor] feature maps from
      :math:`G(\text{source})` at the same layers.

    Both lists must have matching length and per-layer shapes.

    Args:
        n_patches: Number of patches sampled per layer.
        temperature: Softmax temperature :math:`\tau`. Default 0.07
            (matches the CUT paper).
        mlp_dim: Projection-head hidden / output width.
    """

    def __init__(
        self,
        n_patches: int = 256,
        temperature: float = 0.07,
        mlp_dim: int = 256,
    ) -> None:
        super().__init__()
        if n_patches <= 0:
            raise ValueError(f"n_patches must be > 0, got {n_patches}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.n_patches = n_patches
        self.temperature = temperature
        self.sampler = _PatchSampleMLP(mlp_dim=mlp_dim)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError(
                "CUTPatchNCELoss requires context with 'feat_source' and 'feat_translated' lists."
            )
        feats_src = context.get("feat_source")
        feats_tgt = context.get("feat_translated")
        if feats_src is None or feats_tgt is None:
            raise ValueError(
                "context['feat_source'] and context['feat_translated'] are both required."
            )
        if len(feats_src) != len(feats_tgt):
            raise ValueError(
                f"feat_source has {len(feats_src)} layers but feat_translated has {len(feats_tgt)}."
            )

        device = feats_src[0].device
        loss = torch.zeros((), device=device)
        for f_src, f_tgt in zip(feats_src, feats_tgt, strict=True):
            if f_src.shape != f_tgt.shape:
                raise ValueError(
                    f"feat shape mismatch {tuple(f_src.shape)} vs {tuple(f_tgt.shape)}"
                )
            b, _, h, w = f_src.shape
            n = min(self.n_patches, h * w)
            idx = torch.stack(
                [torch.randperm(h * w, device=device)[:n] for _ in range(b)],
                dim=0,
            )  # (B, n)
            q = self.sampler(f_tgt, idx)  # (B, n, D)
            k = self.sampler(f_src, idx)  # (B, n, D), positives in same row
            # logits: (B, n, n) — query i vs all keys
            logits = torch.einsum("bnd,bmd->bnm", q, k) / self.temperature
            labels = torch.arange(n, device=device).expand(b, n)
            loss = loss + F.cross_entropy(logits.reshape(b * n, n), labels.reshape(-1))
        return loss / max(len(feats_src), 1)
