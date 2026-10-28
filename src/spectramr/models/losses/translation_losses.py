r"""Cycle-consistency & identity losses for unpaired image translation.

Both losses follow the CycleGAN formulation. They are intentionally
**stateless** in the sense that the generator(s) are passed via the
``context`` dict at forward time — this lets the cycle/identity terms
participate in the registry-driven `LossBuilder` without dragging the
generator wiring into the schema.

* **Cycle consistency** (#23):
  :math:`\mathcal{L}_{\text{cyc}} = \|G_{B\to A}(G_{A\to B}(\mathbf{x}_A))-\mathbf{x}_A\|_1`
* **Identity** (#24):
  :math:`\mathcal{L}_{\text{id}}  = \|G_{A\to B}(\mathbf{x}_B)-\mathbf{x}_B\|_1`

Reference:
    J.-Y. Zhu *et al.*, "Unpaired image-to-image translation using
    cycle-consistent adversarial networks," *Proc. IEEE ICCV*, 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


def _resolve_norm(norm: str) -> int:
    if norm == "l1":
        return 1
    if norm == "l2":
        return 2
    raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")


@register_loss(name="cycle_consistency", aliases=["cycle", "CycleConsistencyLoss"], domain="image")
class CycleConsistencyLoss(nn.Module):
    r"""Reconstructed-after-roundtrip identity penalty.

    Forward expects ``prediction`` to already be the round-tripped image
    :math:`G_{B\to A}(G_{A\to B}(\mathbf{x}_A))` and ``target`` to be the
    original :math:`\mathbf{x}_A`. Strategies that compute the round trip
    can simply hand both tensors here; nothing else is required.

    Args:
        norm: ``"l1"`` (default) or ``"l2"``.
    """

    def __init__(self, norm: str = "l1") -> None:
        super().__init__()
        self.norm = norm
        self.p = _resolve_norm(norm)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if self.norm == "l1":
            return F.l1_loss(prediction, target)
        return F.mse_loss(prediction, target)


@register_loss(name="identity_preservation", aliases=["identity", "IdentityLoss"], domain="image")
class IdentityPreservationLoss(nn.Module):
    r"""Identity loss: when the input is already in the target domain,
    the translator should leave it unchanged.

    Forward expects ``prediction`` :math:`= G_{A\to B}(\mathbf{x}_B)` and
    ``target`` :math:`= \mathbf{x}_B`.

    Args:
        norm: ``"l1"`` (default) or ``"l2"``.
    """

    def __init__(self, norm: str = "l1") -> None:
        super().__init__()
        # Validated as in CycleConsistencyLoss; an unknown norm would otherwise run MSE.
        _resolve_norm(norm)
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if self.norm == "l1":
            return F.l1_loss(prediction, target)
        return F.mse_loss(prediction, target)
