r"""Concomitant-phase residual loss (CAUR / idea 1 §1.2).

Penalises the magnitude of the *residual* concomitant-phase
correction emitted by an analytical-plus-residual head:

.. math::

    \mathcal{L}_{c} = \lambda \, \lVert \Delta\phi_\theta \rVert_2^2

where :math:`\Delta\phi_\theta` is the residual correction the network
adds on top of the closed-form Bernstein 1998 prediction, and
:math:`\lambda` is the weight. The residual head is *only* used when
``physics.concomitant.model = "analytical_plus_residual"`` — for the
default ``analytical`` mode this loss is omitted from the recipe.

Why penalise the residual?
--------------------------
The Bernstein closed-form already captures the leading concomitant
term. The residual head is a fall-back for un-modelled effects (gradient
imperfections, eddy currents, etc.). Allowing it to absorb arbitrary
phase content would let the network "explain away" data inconsistencies
that should be handled by the data-consistency layer instead. A small
ridge penalty keeps the head honest.

Reference
---------
Plan §1.2 of ``TODO/integration_plan_ulf_cheap_fast_mri.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss


@register_loss(
    name="concomitant_phase_residual",
    aliases=["ConcomitantPhaseResidualLoss"],
    domain="kspace",
    # Its input is a residual-phase head output in radians, not k-space or image
    # data; forward() refuses the (prediction, target) call a loss list makes.
    compatible_with=["image"],
)
class ConcomitantPhaseResidualLoss(nn.Module):
    r"""Squared-L2 ridge on the concomitant-phase residual head output.

    The residual phase must be passed by keyword. A loss list calls
    ``loss(prediction, target)``, and penalising the prediction instead of the
    residual phase trains ``λ · mean(prediction²)``, an image-energy penalty with
    no concomitant-field content (#2275), so a positional call raises.

    Args:
        weight: ``λ`` multiplier. Must be ≥ 0.
        reduction: ``"mean"`` (default) → ``λ · mean(|Δφ|²)``;
            ``"sum"`` → ``λ · sum(|Δφ|²)``.
    """

    def __init__(self, weight: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be ≥ 0; got {weight}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}")
        self.weight = float(weight)
        self.reduction = reduction

    def forward(
        self,
        *args: object,
        residual_phase: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        r"""Compute :math:`\lambda \, \lVert \Delta\phi \rVert_2^2`.

        Args:
            residual_phase: Real-valued residual-phase tensor (radians),
                keyword-only.

        Returns:
            Scalar tensor.

        Raises:
            TypeError: Called positionally, or without ``residual_phase``.
        """
        if args or residual_phase is None:
            raise TypeError(
                "concomitant_phase_residual penalises the residual-phase head output and "
                "must be called as loss(residual_phase=...). A (prediction, target) call "
                "from a loss list would penalise the prediction's energy instead; no "
                "strategy produces a residual phase, so disable this entry (#2275)."
            )
        sq = residual_phase.pow(2)
        penalty = sq.mean() if self.reduction == "mean" else sq.sum()
        return self.weight * penalty


__all__ = ["ConcomitantPhaseResidualLoss"]
