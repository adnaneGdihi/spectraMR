r"""Anomaly-aware loss weighting for pathological MRI.

Implements §7 of the multi-contrast brainstorm — boosts the per-voxel loss
contribution inside known pathological regions so a model trained on a
majority-healthy cohort doesn't preferentially explain healthy tissue at
the expense of lesions.

The weight map is

.. math::

   w(\mathbf{r}) \;=\; 1 \;+\; \beta \cdot \mathbb{1}\!\big[\mathbf{r} \in \Omega_p\big],

where :math:`\Omega_p` is the lesion mask (binary or soft) and
:math:`\beta` is the boost factor. The weighted loss is a per-voxel
elementwise product of the underlying loss with :math:`w` followed by mean
reduction:

.. math::

   \mathcal{L}_w \;=\; \frac{\sum_\mathbf{r} w(\mathbf{r})\, \ell(\mathbf{r})}
                            {\sum_\mathbf{r} w(\mathbf{r})}.

The wrapper is composable: it accepts an *underlying* loss instance
(L1, L2, SSIM, perceptual, ...) and reduces *no further* than the
existing per-voxel pattern, so any loss whose ``forward`` returns the
unreduced ``(B, C, H, W)`` map is supported. A base that aggregates
internally (e.g. SSIM with reduction='mean') is rejected with a
``ValueError``: there is no per-voxel map to weight.

The lesion mask is supplied via the loss ``context`` dict under the key
``lesion_mask`` — same convention as
:class:`~spectramr.models.losses.bloch_consistency_loss.BlochConsistencyLoss`.
The strategy is responsible for populating it; the dataset attaches the
mask to the TorchIO Subject under the same key (per
``BrainExtractionService`` convention).

References
----------
[14] T. Park, M.-Y. Liu, T.-C. Wang, J.-Y. Zhu, "Semantic image synthesis
     with spatially-adaptive normalization (SPADE)," CVPR 2019.
[15] L. Zhang, A. Rao, M. Agrawala, "Adding conditional control to
     text-to-image diffusion models (ControlNet)," ICCV 2023.

§7 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss

__all__ = ["LesionWeightedLoss"]


def _l1_per_voxel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs()


def _l2_per_voxel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2)


_BASE_LOSS_FNS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "l1": _l1_per_voxel,
    "l2": _l2_per_voxel,
    "smooth_l1": lambda p, t: F.smooth_l1_loss(p, t, reduction="none"),
}


@register_loss(name="lesion_weighted", domain="image")
class LesionWeightedLoss(nn.Module):
    r"""Spatially boosts the loss inside lesion regions.

    Args:
        base: Either a string in ``{"l1", "l2", "smooth_l1"}`` selecting a
            built-in per-voxel base loss, or an ``nn.Module`` whose forward
            returns a per-voxel loss map of the same shape as ``prediction``.
        boost: Multiplicative weight applied to lesion voxels
            (:math:`\beta` in the formula). Default 4.0 (5× weighting).
        soft_mask: If True, treat the lesion mask as a soft probability in
            ``[0, 1]``. If False, threshold at 0.5 to obtain a binary mask.
        require_mask: If True (default), raise when ``context["lesion_mask"]``
            is missing. If False, fall back to uniform weighting (i.e. the
            base loss is reproduced). Useful when training on a mixed cohort
            where only some samples carry annotations.
    """

    def __init__(
        self,
        base: str | nn.Module = "l1",
        boost: float = 4.0,
        soft_mask: bool = True,
        require_mask: bool = True,
    ) -> None:
        super().__init__()
        if boost < 0:
            raise ValueError(f"boost must be >= 0, got {boost}")
        self.boost = boost
        self.soft_mask = soft_mask
        self.require_mask = require_mask
        self._base_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
        self.base_module: nn.Module | None
        if isinstance(base, str):
            if base not in _BASE_LOSS_FNS:
                raise ValueError(
                    f"unknown base {base!r}; expected one of {list(_BASE_LOSS_FNS)} "
                    f"or an nn.Module that returns a per-voxel map."
                )
            self.base_name = base
            self._base_fn = _BASE_LOSS_FNS[base]
            self.base_module = None
        else:
            self.base_name = type(base).__name__
            self._base_fn = None
            self.base_module = base

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        per_voxel: torch.Tensor
        if self._base_fn is not None:
            per_voxel = self._base_fn(prediction, target)
        else:
            assert self.base_module is not None
            per_voxel = self.base_module(prediction, target)
        if per_voxel.shape != prediction.shape:
            raise ValueError(
                f"base loss returned shape {tuple(per_voxel.shape)} but "
                f"prediction has shape {tuple(prediction.shape)} — "
                "LesionWeightedLoss requires a per-voxel base."
            )

        mask: torch.Tensor | None = None if context is None else context.get("lesion_mask")
        if mask is None:
            if self.require_mask:
                raise ValueError(
                    "LesionWeightedLoss requires `context['lesion_mask']`. "
                    "Set require_mask=False to fall back to uniform weighting."
                )
            return per_voxel.mean()

        mask = mask.to(prediction.dtype).to(prediction.device)
        if mask.dim() == 3:  # (B, H, W) → (B, 1, H, W)
            mask = mask.unsqueeze(1)
        if not self.soft_mask:
            mask = (mask > 0.5).to(prediction.dtype)
        if mask.shape[-2:] != prediction.shape[-2:]:
            raise ValueError(
                f"lesion_mask spatial shape {tuple(mask.shape[-2:])} != "
                f"prediction spatial shape {tuple(prediction.shape[-2:])}."
            )

        # Expand the (B, 1, H, W) map to the per-voxel shape so the denominator
        # counts every channel the numerator sums, keeping boost=0 equal to the mean.
        weights = (1.0 + self.boost * mask).expand_as(per_voxel)
        return (per_voxel * weights).sum() / weights.sum().clamp_min(1.0)
