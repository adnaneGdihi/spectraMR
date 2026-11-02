"""Which image the declared ``losses.image_losses`` of CycleBlochStrategy compare against.

Paired terms (``l1``, ``mse``, ...) compare the generated HF image with the HF target.
``segmentation_equivariance`` and ``gradient_domain_consistency`` read their second
argument as the *source* image (their anti-hallucination contract), so they compare
the generated image with the measured input. ``segmentation_equivariance`` also needs
a frozen segmenter in its context, which ``training.cycle_bloch.segmenter_backend``
names.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectramr.infrastructure.training.strategies.bloch_synth_strategy import (
    _IntensitySoftSegmenter,
)
from spectramr.models.losses.weights import canonical_loss_name

#: Losses whose second argument is the source image, not the target.
SOURCE_ANCHORED = frozenset({"segmentation_equivariance", "gradient_domain_consistency"})
_NEEDS_SEGMENTER = "segmentation_equivariance"


class _WithContext(nn.Module):
    """Binds a loss's ``context`` so the shared fold can call it as ``loss(pred, target)``."""

    def __init__(self, loss: nn.Module, context: dict[str, Any]) -> None:
        super().__init__()
        self.loss = loss
        self._context = context

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(prediction, target, context=self._context)


def resolve_segmenter(backend: str, device: torch.device) -> nn.Module | None:
    """The frozen segmenter ``segmenter_backend`` names, on ``device``; ``None`` for none."""
    if backend == "none":
        return None
    if backend == "label_dice":
        return _IntensitySoftSegmenter().to(device).eval().requires_grad_(False)
    raise ValueError(
        f"training.cycle_bloch.segmenter_backend={backend!r}; supported: none, label_dice."
    )


def split_image_losses(
    env_losses: dict[str, nn.Module] | None,
    weights: dict[str, float],
    segmenter: nn.Module | None,
) -> tuple[dict[str, nn.Module], dict[str, nn.Module]]:
    """(paired, source-anchored) losses; raises when a weighted seg term has no segmenter."""
    paired: dict[str, nn.Module] = {}
    anchored: dict[str, nn.Module] = {}
    for name, module in (env_losses or {}).items():
        canon = canonical_loss_name(name)
        if canon not in SOURCE_ANCHORED:
            paired[name] = module
            continue
        if canon == _NEEDS_SEGMENTER and weights.get(canon, 1.0):
            if segmenter is None:
                raise ValueError(
                    "losses.image_losses declares segmentation_equivariance, which needs a "
                    "frozen segmenter; set training.cycle_bloch.segmenter_backend."
                )
            module = _WithContext(module, {"segmenter": segmenter})
        anchored[name] = module
    return paired, anchored
