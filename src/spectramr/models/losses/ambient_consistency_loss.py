r"""Ambient Diffusion held-out k-space consistency loss (Phase B step 5).

Ambient Diffusion (Daras et al., NeurIPS 2023) learns a clean image prior from
*corrupted* (here: undersampled) data alone. For MRI (Aali et al., Ambient
Diffusion Posterior Sampling) the SSDU Λ/Θ split is lifted onto the diffusion
model: at each diffusion noise level the model's clean-image estimate
:math:`\hat{x}_0` must agree with the **held-out** Θ measurement,

.. math::
    \mathcal{L}_{\text{ambient}} = w \,\big\| M_\Theta (F\,\hat{x}_0 - \mathbf{y}) \big\|_2^2 .

This is the SSDU held-out residual evaluated on the diffusion model's
:math:`\hat{x}_0` (rather than a one-shot reconstruction). The scalar
``weight``/``ambient_weight`` is the ambient reweighting hook; at ``weight = 1``
it is the plain held-out residual, and combined with the standard diffusion
denoising term the whole objective reduces to vanilla SSDU as the diffusion
noise level → 0. ``M_Θ`` is the held-out k-space mask (disjoint from the
network-input partition Λ).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.registry import register_loss


def ambient_consistency_residual(
    x0_pred: torch.Tensor,
    target_kspace: torch.Tensor,
    theta_mask: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    r"""Held-out Θ k-space residual ``w·‖M_Θ(F·x0_pred − y)‖²`` (Hermitian).

    Args:
        x0_pred: the diffusion model's clean-image estimate (real or complex,
            ``[B, C, H, W]``).
        target_kspace: the measured k-space ``y`` (complex).
        theta_mask: the held-out Θ mask (1 where held out), broadcastable to
            ``target_kspace``.
        weight: ambient reweighting scalar (``1.0`` → plain held-out residual).

    Returns:
        Scalar consistency loss. Zero when ``theta_mask`` is empty or when
        ``F·x0_pred`` matches ``y`` on Θ.
    """
    x0_c = x0_pred if x0_pred.is_complex() else torch.complex(x0_pred, torch.zeros_like(x0_pred))
    k_pred = fft2c(x0_c)
    theta = theta_mask if theta_mask.is_complex() else theta_mask.to(k_pred.real.dtype)
    resid = theta * (k_pred - target_kspace)
    return weight * (resid.conj() * resid).real.mean()


@register_loss(
    name="ambient_consistency",
    aliases=["AmbientConsistency", "ambient_dps_consistency"],
    domain="kspace",
)
class AmbientConsistencyLoss(nn.Module):
    r"""Registered Ambient Diffusion held-out k-space consistency loss.

    Forward expects the model's clean-image estimate as ``prediction`` and reads
    the measured k-space + held-out mask from ``context``:

    * ``context["target_kspace"]``: the measured k-space ``y``.
    * ``context["theta_mask"]``: the held-out Θ mask.
    * ``context["ambient_weight"]`` (optional): overrides the reweighting scalar.

    ``target`` is ignored (Ambient is reference-free).
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        if weight < 0.0:
            raise ValueError(f"weight must be >= 0, got {weight}")
        self.weight = float(weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None or "target_kspace" not in context or "theta_mask" not in context:
            raise ValueError(
                "AmbientConsistencyLoss requires context['target_kspace'] "
                "(measured y) and context['theta_mask'] (held-out Θ mask)."
            )
        weight = float(context.get("ambient_weight", self.weight))
        return ambient_consistency_residual(
            prediction, context["target_kspace"], context["theta_mask"], weight=weight
        )


__all__ = ["AmbientConsistencyLoss", "ambient_consistency_residual"]
