"""Masked MSE between a gain-scaled render and the acquired series, gain known or projected.

``calibrated`` takes the per-series gain the series-gain gate measured, which closes the
two gain directions the identifiability gate leaves open. ``projected`` fits a per-image
least-squares gain instead: the image-domain control, identifiable only up to those two.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

_MODES = ("calibrated", "projected")


@register_loss(name="calibrated_gain_bloch", domain="physics", compatible_with=["image"])
class CalibratedGainBlochLoss(nn.Module):
    """``mean_mask (g * render - target)^2`` with ``g`` supplied or projected per image."""

    def __init__(self, gain_mode: str) -> None:
        super().__init__()
        if gain_mode not in _MODES:
            raise ValueError(f"gain_mode={gain_mode!r}; expected one of {_MODES}")
        self.gain_mode = gain_mode

    def forward(
        self,
        prediction: Any,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not context or "render" not in context or "mask" not in context:
            raise ValueError("calibrated_gain_bloch needs context['render'] and context['mask']")
        render = context["render"]
        w = context["mask"].to(render.dtype)
        if self.gain_mode == "calibrated":
            gain = context.get("gain")
            if gain is None:
                raise ValueError("calibrated_gain_bloch in calibrated mode needs context['gain']")
        else:
            num = (render * target * w).flatten(1).sum(1)
            den = (render * render * w).flatten(1).sum(1).clamp_min(1e-12)
            gain = (num / den).view(-1, 1, 1, 1)
        resid = (gain * render - target) * w
        return resid.pow(2).sum() / w.sum().clamp_min(1)
