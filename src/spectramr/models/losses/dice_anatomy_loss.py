r"""Segmentation-preserving Dice loss — anti-hallucination anchor.

Run a *frozen* segmentation network on both the prediction and the
ground-truth (or a structurally-equivalent reference) image and
penalise per-class Dice disagreement:

.. math::
    \mathcal{L}_{\text{seg}} = 1 - \tfrac{1}{C}\sum_c
        \frac{2\,|S_c(\hat{\mathbf{x}})\cap S_c(\mathbf{x})|}
             {|S_c(\hat{\mathbf{x}})|+|S_c(\mathbf{x})|}.

Pairs naturally with translation / SR / synthesis pipelines where the
network may be tempted to invent or erase anatomy that radiologically
matters: forcing the segmenter to agree on the prediction is a
content-preservation guarantee that pixel-level losses cannot give.

The frozen segmenter is supplied via the ``context`` dict
(``context["segmenter"]``) so that this loss does not require its own
service injection in the training schema. Strategies should resolve
``ISegmentationService`` once during ``init_session`` and stash the
network there.

Reference:
    B. Billot *et al.*, "SynthSeg: Segmentation of brain MRI scans of
    any contrast and resolution without retraining," *Med. Image
    Anal.*, 86, 2023.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


def _soft_dice_per_class(p: torch.Tensor, t: torch.Tensor, smooth: float) -> torch.Tensor:
    r"""Per-class soft Dice score over (B, C, *spatial). Returns (C,).

    Uses the Milletari (V-Net) squared-denominator formulation:

    .. math::

        \text{Dice}_c = \frac{2\,\sum p_c t_c + \epsilon}
                             {\sum p_c^2 + \sum t_c^2 + \epsilon}

    This variant is preferred over the linear-denominator form
    (``sum(p) + sum(t)``) because it returns **1.0 on identical
    continuous inputs** — e.g. matched softmax probabilities from a
    frozen segmenter — whereas the linear form collapses to
    ``mean(p^2)/mean(p) ≈ 1/C`` for uniform soft predictions. Both
    forms agree on hard-binary masks, but only the V-Net form is
    correct on the *soft* outputs the anti-hallucination loss
    actually consumes.
    """
    reduce_dims = [0, *range(2, p.dim())]
    inter = (p * t).sum(dim=reduce_dims)
    p_sq = (p * p).sum(dim=reduce_dims)
    t_sq = (t * t).sum(dim=reduce_dims)
    return (2.0 * inter + smooth) / (p_sq + t_sq + smooth)


@register_loss(
    name="segmentation_dice",
    aliases=["SegmentationDiceLoss", "anti_hallucination_dice", "synthseg_dice"],
    domain="image",
)
class SegmentationDiceLoss(nn.Module):
    r"""Anti-hallucination Dice on the soft segmentation of a frozen
    teacher segmenter.

    Forward expects:

    * ``prediction``: candidate image, shape ``(B, 1, H, W)`` (or 3D).
    * ``target``: reference image of the same shape; the segmenter is
      run on both.
    * ``context["segmenter"]`` (optional override): a callable
      ``segmenter(image) -> logits`` returning ``(B, C, *spatial)``.
      If absent, the loss raises unless ``require_segmenter=False``.

    Args:
        smooth: Numerical smoothing for the Dice denominator.
        ignore_background: If True (default) drop class index 0 before
            averaging — the background class is usually trivial and
            can dominate the score.
        require_segmenter: If True (default) raise when no segmenter is
            available. If False, the loss returns 0.0 (useful for
            warming up while the segmenter is still being loaded).
    """

    def __init__(
        self,
        smooth: float = 1e-4,
        ignore_background: bool = True,
        require_segmenter: bool = True,
    ) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError(f"smooth must be > 0, got {smooth}")
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.require_segmenter = require_segmenter

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        ctx = context or {}
        segmenter = ctx.get("segmenter")
        if segmenter is None:
            if self.require_segmenter:
                raise ValueError(
                    "SegmentationDiceLoss requires context['segmenter'] "
                    "(a frozen segmentation network). Set "
                    "require_segmenter=False to fall back to a no-op."
                )
            return prediction.new_zeros(())
        if not callable(segmenter):
            raise TypeError("context['segmenter'] must be callable.")
        logits_pred = segmenter(prediction)
        with torch.no_grad():
            logits_target = segmenter(target)
        if logits_pred.shape != logits_target.shape:
            raise ValueError(
                f"segmenter outputs differ: {tuple(logits_pred.shape)} vs "
                f"{tuple(logits_target.shape)}"
            )
        probs_pred = F.softmax(logits_pred, dim=1)
        probs_target = F.softmax(logits_target, dim=1).detach()
        per_class = _soft_dice_per_class(probs_pred, probs_target, self.smooth)
        if self.ignore_background and per_class.numel() > 1:
            per_class = per_class[1:]
        return 1.0 - per_class.mean()
