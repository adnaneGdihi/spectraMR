r"""Equivariant SSL reconstruction — measurement-domain symmetry.

For a known group action :math:`R` (rotation, flip, translation, …)
that commutes with the forward operator :math:`\mathbf{A}`, the true
reconstruction satisfies :math:`f(R\mathbf{y}) = R\,f(\mathbf{y})`.
Penalising deviations from this identity provides supervision *without
ground truth*. Combined with SSDU (Yaman 2020) it injects a second
source of signal that mitigates the chief weakness of SSDU at high
acceleration where the held-out partition Θ leaks too much information.

Formally:

.. math::
    \mathcal{L}_{\text{eq}} = \big\|\,f_{\boldsymbol{\theta}}(R\mathbf{y}) - R\,f_{\boldsymbol{\theta}}(\mathbf{y})\,\big\|_p.

The strategy is responsible for actually evaluating the network on
:math:`R\mathbf{y}` and supplying both branches via the loss
``context``.

Reference:
    [67] D. Chen *et al.*, "Equivariant imaging: Learning beyond the
        range space," *Proc. IEEE/CVF ICCV*, 2021.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss(
    name="equivariant_ssl_recon",
    aliases=["EquivariantImagingLoss", "ei_loss", "equivariant_recon"],
    domain="image",
)
class EquivariantSSLReconLoss(nn.Module):
    r"""Measurement-domain equivariance penalty.

    Forward expects:

    * ``prediction``: :math:`f_{\boldsymbol{\theta}}(R\mathbf{y})` —
      the network's reconstruction on the *transformed* measurement.
    * ``target``: ignored (the comparison signal lives in
      ``context``).
    * ``context["transformed_recon"]``: equivalently
      :math:`R\,f_{\boldsymbol{\theta}}(\mathbf{y})` — the same network
      evaluated on the original measurement, then transformed via the
      same group element :math:`R`.

    The loss simply measures :math:`\|prediction - transformed\_recon\|`.

    Strategies typically draw :math:`R` from a small set of discrete
    rotations / flips that commute with the Cartesian forward operator
    (90° rotations and axis flips are exact; translations require care
    near the FOV edges).

    Args:
        norm: ``"l2"`` (default) or ``"l1"``.
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None or "transformed_recon" not in context:
            raise ValueError(
                "EquivariantSSLReconLoss requires "
                "context['transformed_recon'] (the reference branch "
                "R · f(y))."
            )
        ref = context["transformed_recon"]
        if ref.shape != prediction.shape:
            raise ValueError(
                f"transformed_recon shape {tuple(ref.shape)} != "
                f"prediction shape {tuple(prediction.shape)}"
            )
        if self.norm == "l1":
            return F.l1_loss(prediction, ref)
        return F.mse_loss(prediction, ref)
