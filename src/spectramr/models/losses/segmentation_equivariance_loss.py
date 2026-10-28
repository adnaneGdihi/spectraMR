r"""Segmentation-equivariance loss for cross-contrast translation.

Pathological structure can be hallucinated by cycle-consistent translation
networks (CycleGAN, MUNIT) trained on healthy-dominant data. The standard
fix is to require that a *frozen* multi-contrast segmenter produces the
same segmentation map before and after translation:

.. math::

   \mathcal{L}_{\text{seg-eq}}
   \;=\;
   D\!\big(S(\mathbf{x}),\, S(G(\mathbf{x}))\big),

where :math:`S` is a frozen (or EMA-frozen) segmenter, :math:`G` is the
cross-contrast generator, and :math:`D` is a logit-space distance
(KL divergence by default, optionally L1 on softmax probabilities or
cross-entropy when targets are hard labels). Pretrained anatomy-aware
segmenters such as SynthSeg [3] are well-suited for :math:`S` because
they generalise across contrasts; alternatively any in-house segmenter
that handles the cohort works.

The loss is intentionally *generator-agnostic* — it only needs the
"before" image and the "after" image (both of which the strategy already
has). The frozen segmenter is supplied via the loss ``context`` dict
under the key ``segmenter`` (an ``nn.Module``).

The wrapper does not download or initialise a segmenter; production code
should resolve one through the DI container (per CLAUDE.md service-DI
rule) and inject it.

References
----------
[3]  B. Billot *et al.*, "SynthSeg: Segmentation of brain MRI scans of any
     contrast and resolution without retraining," *Med. Image Anal.*,
     vol. 86, p. 102789, 2023.
[6]  J.-Y. Zhu, T. Park, P. Isola, A. A. Efros, "Unpaired image-to-image
     translation using cycle-consistent adversarial networks (CycleGAN),"
     ICCV 2017.

§2 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss

__all__ = ["SegmentationEquivarianceLoss"]


@register_loss(name="segmentation_equivariance", domain="image")
class SegmentationEquivarianceLoss(nn.Module):
    r"""Equivariance penalty: :math:`S(\mathbf{x}) \approx S(G(\mathbf{x}))`.

    Args:
        distance: Distance functional in segmentation logit space.
            ``"kl"`` — symmetric KL divergence on softmax (default).
            ``"l1"`` — L1 on softmax probabilities.
            ``"l2"`` — MSE on softmax probabilities.
        temperature: Softmax temperature. Lower values sharpen the
            distribution before computing the distance.
        detach_anchor: If True, the segmentation of the *source* image is
            detached so gradients only flow through the generator's output.
            Default True — the canonical cycle-consistency convention.
    """

    _DISTANCES: ClassVar[frozenset[str]] = frozenset({"kl", "l1", "l2"})

    def __init__(
        self,
        distance: str = "kl",
        temperature: float = 1.0,
        detach_anchor: bool = True,
    ) -> None:
        super().__init__()
        if distance not in self._DISTANCES:
            raise ValueError(f"distance must be one of {sorted(self._DISTANCES)}, got {distance!r}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.distance = distance
        self.temperature = temperature
        self.detach_anchor = detach_anchor

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if context is None or "segmenter" not in context:
            raise ValueError(
                "SegmentationEquivarianceLoss requires `context['segmenter']` — "
                "a frozen nn.Module producing logits of shape (B, K, H, W)."
            )
        segmenter: nn.Module = context["segmenter"]

        anchor_input = target  # the source-domain image
        translated_input = prediction  # the generator's output
        # A detached anchor needs no graph: building one only to discard it costs
        # a full segmenter's worth of saved activations every step.
        anchor_logits: torch.Tensor
        if self.detach_anchor:
            with torch.no_grad():
                anchor_logits = segmenter(anchor_input)
        else:
            anchor_logits = segmenter(anchor_input)
        translated_logits: torch.Tensor = segmenter(translated_input)

        if anchor_logits.shape != translated_logits.shape:
            raise ValueError(
                f"segmenter returned mismatched shapes "
                f"{tuple(anchor_logits.shape)} vs {tuple(translated_logits.shape)}."
            )

        a = anchor_logits / self.temperature
        b = translated_logits / self.temperature
        if self.distance == "kl":
            log_p = F.log_softmax(a, dim=1)
            log_q = F.log_softmax(b, dim=1)
            p = log_p.exp()
            q = log_q.exp()
            return 0.5 * (
                F.kl_div(log_q, p, reduction="batchmean")
                + F.kl_div(log_p, q, reduction="batchmean")
            )
        # softmax-space comparisons
        p = F.softmax(a, dim=1)
        q = F.softmax(b, dim=1)
        if self.distance == "l1":
            return (p - q).abs().mean()
        return (p - q).pow(2).mean()
