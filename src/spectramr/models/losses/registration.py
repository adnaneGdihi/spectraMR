from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


# `lncc` alias intentionally dropped — it was colliding with the canonical
# `lncc` registration in `cross_contrast_losses.py` (a different LNCC
# tailored to multi-contrast registration). The canonical disambiguated
# handle for *this* implementation is `local_cross_correlation_loss`. See
# TODO/audit/12_losses.md F2.
@register_loss("local_cross_correlation_loss", aliases=["lncc_loss"])
class LocalCrossCorrelationLoss(nn.Module):
    """Computes Local Normalized Cross Correlation (LNCC).
    Robust to intensity variations between scanners.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{LCC} = - \\mathbb{E} \\left[ \\frac{\\text{Cov}(\\hat{y}, y)^2}{\\text{Var}(\\hat{y}) \\text{Var}(y) + \\epsilon} \\right]"""

    def __init__(self, kernel_size: int = 9) -> None:
        """__init__.

        Args:
            kernel_size: Side of the square local window.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Negative mean local NCC^2 between ``prediction`` and ``target``.

        Args:
            prediction: ``(B, C, H, W)`` image (e.g. the corrected 64 mT scan).
            target: ``(B, C, H, W)`` reference (e.g. the 3 T scan).

        Each channel is windowed independently (depthwise conv) and the local
        statistics are averaged over channels, so ``C > 1`` is supported.
        """
        channels = prediction.shape[1]
        k = self.kernel_size
        win = torch.full(
            (channels, 1, k, k), 1.0 / (k * k), device=prediction.device, dtype=prediction.dtype
        )

        def _local_mean(x: torch.Tensor) -> torch.Tensor:
            return F.conv2d(x, win, padding=self.padding, groups=channels)

        mean_p = _local_mean(prediction)
        mean_t = _local_mean(target)
        cross = _local_mean(prediction * target) - mean_p * mean_t
        # Var = E[X^2] - E[X]^2 can dip a few ULP below zero; clamp so the
        # denominator cannot flip sign near flat regions.
        var_p = (_local_mean(prediction * prediction) - mean_p * mean_p).clamp_min(0.0)
        var_t = (_local_mean(target * target) - mean_t * mean_t).clamp_min(0.0)

        cc = cross * cross / (var_p * var_t + 1e-5)
        return -1.0 * torch.mean(cc)


@register_loss("smoothness_loss", aliases=["smooth_loss", "flow_smoothness"])
class SmoothnessLoss(nn.Module):
    """Penalizes jagged deformations to ensure physically realistic B0 maps.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{Smoothness} = \\frac{1}{2} (\\mathbb{E}[|\nabla_x v|] + \\mathbb{E}[|\nabla_y v|])"""

    def forward(
        self, flow: torch.Tensor, target: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """Mean absolute first difference of ``flow`` along H and W.

        Args:
            flow: ``(B, C, H, W)`` deformation / B0 field.
            target: Unused; accepted so the registry's ``loss(prediction, target)``
                call reaches the regulariser instead of raising ``TypeError``.
        """
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        return (torch.mean(dx) + torch.mean(dy)) / 2.0
