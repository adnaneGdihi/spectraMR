r"""SSDU's normalised L1-L2 loss (Yaman et al., MRM 2020, Eq. 6).

.. math::

    \mathcal L = \frac{\lVert M_\Theta(FS\hat x - y)\rVert_2}{\lVert M_\Theta y\rVert_2}
               + \frac{\lVert M_\Theta(FS\hat x - y)\rVert_1}{\lVert M_\Theta y\rVert_1},

per sample, averaged over the batch, with the real and imaginary parts as
separate entries in both norms. Normalising by the held-out measurement makes the
loss scale-free across slices whose k-space energy differs by orders of magnitude,
and the L1 half keeps the low-energy periphery from vanishing beside the centre,
which the plain mean-squared :class:`~spectramr.models.losses.ssdu_loss.SSDULoss`
lets dominate.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn

from spectramr.models.losses.registry import register_loss
from spectramr.models.losses.ssdu_loss import _coerce_mask, _predicted_kspace

__all__ = ["SSDUNormalizedLoss"]


@register_loss(name="ssdu_normalized", aliases=["SSDUNormalizedLoss"], domain="kspace")
class SSDUNormalizedLoss(nn.Module):
    r"""Normalised L1-L2 SSDU loss on the held-out set :math:`\Theta`.

    Reads the same context as :class:`~spectramr.models.losses.ssdu_loss.SSDULoss`:
    ``target_kspace`` (:math:`y`), ``theta_mask`` and optional ``coil_sensitivities``.
    Robust SSDU's ``loss_weights`` are refused: Eq. 35 is derived for the squared loss.
    """

    #: The prediction is an image and the loss applies ``F S`` itself.
    applies_forward_operator: ClassVar[bool] = True

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del target, kwargs  # the measurement in the context is the target
        if (
            context is None
            or context.get("target_kspace") is None
            or context.get("theta_mask") is None
        ):
            raise ValueError(
                "SSDUNormalizedLoss requires context['target_kspace'] and ['theta_mask']."
            )
        if context.get("loss_weights") is not None:
            raise ValueError(
                "SSDUNormalizedLoss is the vanilla-SSDU loss; Robust SSDU's Eq. 35 weights "
                "belong to the squared loss (`ssdu`)."
            )
        y = context["target_kspace"]
        pred_k = _predicted_kspace(prediction, context.get("coil_sensitivities"))
        if pred_k.shape != y.shape:
            raise ValueError(f"pred k-space {tuple(pred_k.shape)} != target {tuple(y.shape)}")
        theta = _coerce_mask(context["theta_mask"], y)
        resid = torch.view_as_real((pred_k - y) * theta).flatten(1)
        held = torch.view_as_real(y * theta).flatten(1)
        l2 = resid.norm(dim=1) / held.norm(dim=1).clamp_min(self.eps)
        l1 = resid.abs().sum(dim=1) / held.abs().sum(dim=1).clamp_min(self.eps)
        return (l2 + l1).mean()
