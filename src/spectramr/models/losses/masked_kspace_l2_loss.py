r"""L2 on the k-space lines a second repetition supplies, and nowhere else.

The CoNEX objective scores :math:`F f(M_1 y_i)` against an independent
repetition :math:`y_j` on the lines :math:`M_2`:

.. math::

    \mathcal L = \frac{\lVert M_2 \odot (\hat y - y_j) \rVert_2^2}
                      {\lVert M_2 \odot \mathbf 1 \rVert_0},

the mean squared error over the real/imaginary entries :math:`M_2` selects.
Dividing by the selected count rather than by every entry keeps the scale
per supervised entry, so an arm whose :math:`M_2` covers everything reduces
exactly to the unmasked ``l2`` and a sparser :math:`M_2` does not shrink the
loss merely by selecting less.

``loss_mask`` is mandatory. It is produced by the ``complementary_rep_masking``
transform and reaches this term through the reconstruction strategy's loss
context, which ``_call_safe_loss`` narrows to this signature. Scoring the whole
residual when the mask is missing would silently turn every CoNEX arm into the
``full_target`` bound, so its absence raises.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

__all__ = ["MaskedKSpaceL2Loss"]


@register_loss(name="masked_kspace_l2", domain="kspace")
class MaskedKSpaceL2Loss(nn.Module):
    """Masked k-space MSE, normalised by the number of supervised entries.

    Shape:
        ``pred``/``target``: ``[B, 2C, H, W]`` real/imag-interleaved or
        ``[B, C, H, W]`` complex. ``loss_mask``: real, broadcastable to
        ``pred`` -- ``[B, 1, H, W]`` as collation delivers it.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if loss_mask is None:
            raise ValueError(
                "masked_kspace_l2 requires loss_mask (the lines of the second "
                "repetition). Declare the `complementary_rep_masking` transform, "
                "which emits it, or use the unmasked `l2`."
            )
        if pred.shape != target.shape:
            raise ValueError(
                f"masked_kspace_l2: prediction {tuple(pred.shape)} and target "
                f"{tuple(target.shape)} disagree."
            )
        if torch.is_complex(loss_mask):
            raise ValueError("masked_kspace_l2: loss_mask must be real (a 0/1 line selector).")
        try:
            broadcast = torch.broadcast_shapes(loss_mask.shape, pred.shape)
        except RuntimeError:
            broadcast = None
        if broadcast != pred.shape:
            raise ValueError(
                f"masked_kspace_l2: loss_mask {tuple(loss_mask.shape)} does not broadcast "
                f"onto the prediction {tuple(pred.shape)}."
            )
        residual = pred - target
        if torch.is_complex(residual):
            residual = torch.view_as_real(residual)
            weights = loss_mask.to(residual.dtype).unsqueeze(-1).expand_as(residual)
        else:
            weights = loss_mask.to(residual.dtype).expand_as(residual)
        # Mask the RESIDUAL, not the prediction: the unselected lines must carry
        # no gradient at all, which scaling the prediction alone would not give.
        return (weights * residual.pow(2)).sum() / weights.sum().clamp(min=1.0)
