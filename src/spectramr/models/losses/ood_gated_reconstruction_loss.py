r"""OOD-gated reconstruction loss for refinement on flagged regions.

Implements §7 of the multi-contrast brainstorm — a per-voxel
reconstruction loss whose weighting follows a *spatial out-of-
distribution score*. Voxels where the model output disagrees with the
target by more than a learned threshold (or are flagged by an external
scorer) get *up-weighted* so the next training pass refines those
regions specifically.

Two scoring modes are supported:

- ``"reconstruction_error"`` (default) — derives the OOD score from the
  current per-voxel reconstruction error itself, using a smoothed
  threshold transformation
  :math:`w = 1 + \beta\,\sigma\!\big((|\hat x - x| - \tau)/\tau\big)`
  where :math:`\sigma` is a sigmoid. Voxels with errors above the
  threshold get extra gradient mass; voxels well below get the standard
  weight 1.
- ``"context"`` — read the per-voxel OOD score from the loss
  ``context`` dict under ``"ood_score"``. Useful when an external module
  (a counterfactual healthy generator's saliency map, or a Mahalanobis
  feature-space scorer) computes the score independently.

The loss is a thin wrapper compatible with any per-voxel base
distance function. Default base is L1.

References
----------
[2]  J. E. Iglesias *et al.*, "Joint super-resolution and synthesis of
     1 mm isotropic MP-RAGE volumes from clinical MRI exams,"
     *NeuroImage*, vol. 237, p. 118206, 2021.
[11] H. Chung, J. C. Ye, "Score-based diffusion models for accelerated
     MRI," *Med. Image Anal.*, vol. 80, p. 102479, 2022.

§7 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss

__all__ = ["OODGatedReconstructionLoss"]


def _l1_per_voxel(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (p - t).abs()


def _l2_per_voxel(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (p - t).pow(2)


_BASE_FNS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "l1": _l1_per_voxel,
    "l2": _l2_per_voxel,
    "smooth_l1": lambda p, t: F.smooth_l1_loss(p, t, reduction="none"),
}


@register_loss(name="ood_gated_reconstruction", domain="image")
class OODGatedReconstructionLoss(nn.Module):
    r"""Reconstruction loss with per-voxel OOD-driven up-weighting.

    Args:
        base: Per-voxel distance — ``"l1"``, ``"l2"``, or ``"smooth_l1"``.
        score_source: ``"reconstruction_error"`` (self-derived) or
            ``"context"`` (read from ``context["ood_score"]``).
        boost: :math:`\beta` — multiplicative boost applied at fully-OOD
            voxels (sigmoid → 1 ⇒ weight = 1 + β).
        threshold: :math:`\tau` — error magnitude where the sigmoid
            crosses 0.5 (only used for ``score_source="reconstruction_error"``).
        require_context: If True (and ``score_source="context"``), raise
            when ``context["ood_score"]`` is missing; if False, fall back
            to uniform weighting.
    """

    _SOURCES: ClassVar[frozenset[str]] = frozenset({"reconstruction_error", "context"})

    def __init__(
        self,
        base: str = "l1",
        score_source: str = "reconstruction_error",
        boost: float = 4.0,
        threshold: float = 0.1,
        require_context: bool = True,
    ) -> None:
        super().__init__()
        if base not in _BASE_FNS:
            raise ValueError(f"unknown base {base!r}; expected one of {list(_BASE_FNS)}.")
        if score_source not in self._SOURCES:
            raise ValueError(
                f"score_source must be one of {sorted(self._SOURCES)}, got {score_source!r}."
            )
        if boost < 0:
            raise ValueError(f"boost must be >= 0, got {boost}")
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self.base = base
        self.score_source = score_source
        self.boost = boost
        self.threshold = threshold
        self.require_context = require_context

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        per_voxel = _BASE_FNS[self.base](prediction, target)

        if self.score_source == "reconstruction_error":
            # Self-derived score from current error magnitude.
            with torch.no_grad():
                err = (prediction - target).abs().mean(dim=1, keepdim=True)
                score = torch.sigmoid((err - self.threshold) / self.threshold)
        else:
            score_in = None if context is None else context.get("ood_score")
            if score_in is None:
                if self.require_context:
                    raise ValueError(
                        "OODGatedReconstructionLoss(score_source='context') "
                        "requires `context['ood_score']`. Set require_context=False "
                        "to fall back to uniform weighting."
                    )
                return per_voxel.mean()
            score = score_in.to(prediction.dtype).to(prediction.device)
            if score.dim() == 3:
                score = score.unsqueeze(1)
            if score.shape[-2:] != prediction.shape[-2:]:
                raise ValueError(
                    f"ood_score spatial shape {tuple(score.shape[-2:])} != "
                    f"prediction spatial shape {tuple(prediction.shape[-2:])}."
                )

        # Normalise by the weights broadcast over the channels they multiply, so
        # boost=0 (or a zero score) reduces to the base loss's plain mean for any C.
        weights = (1.0 + self.boost * score).expand_as(per_voxel)
        return (per_voxel * weights).sum() / weights.sum().clamp_min(1.0)
