r"""Physics-informed integration loss.

Moved here from ``src/infrastructure/physics/integration.py`` 2026-05-14
per ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 5 + CLAUDE.md
canonical-home rule (pitfall #12: ``@register_loss`` lives in
``src/models/losses/``).

The loss is a weighted sum of image-domain reconstruction error,
k-space data-consistency, and a placeholder regularization term — used
in physics-informed training where the model's output must both look
like the target image AND match the acquired k-space at the sampled
locations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.core.types import KSpace, Mask, Tensor
from spectramr.models.losses.registry import register_loss


# Alias ``PhysicsInformedLoss`` intentionally dropped — it was colliding
# with the canonical ``physics_informed`` registration in
# ``src/models/losses/physics_losses.py``. The canonical key here is
# ``physics_informed_integration``; use that name explicitly if you want
# the integration variant. Found by the strict duplicate-alias guard,
# see ``TODO/audit/00_implementation_tracker.md`` BB2.
@register_loss("physics_informed_integration")
class PhysicsInformedLoss(nn.Module):
    r"""Physics-informed loss: image fidelity + k-space DC + regularization.

    .. math::

        \mathcal{L} = \lambda_{L2} \| \hat{y} - y \|_2^2
                    + \lambda_{dc} \| (\hat{k} - k) \odot M \|_2^2
                    + \lambda_{reg} \mathcal{L}_{TV}

    Args:
        lambda_l2: Weight on image-domain MSE.
        lambda_dc: Weight on k-space data-consistency term.
        lambda_reg: Weight on the placeholder regularizer.
    """

    def __init__(
        self,
        lambda_l2: float = 1.0,
        lambda_dc: float = 0.0,
        lambda_reg: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_dc = lambda_dc
        self.lambda_reg = lambda_reg
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        pred_image: Tensor,
        target_image: Tensor,
        pred_kspace: KSpace | None = None,
        ref_kspace: KSpace | None = None,
        mask: Mask | None = None,
    ) -> dict[str, Tensor]:
        """Return dict with ``total``/``l2_image``/``dc_loss``/``regularization``."""
        l2_image = self.mse(pred_image, target_image)

        dc_loss = torch.tensor(0.0, device=pred_image.device)
        if self.lambda_dc > 0 and pred_kspace is not None and ref_kspace is not None:
            if mask is not None:
                if pred_kspace.ndim == 4 and mask.ndim == 3:
                    # k-space layout may carry real/imag in the last dim.
                    m = mask.unsqueeze(-1) if pred_kspace.shape[-1] == 2 else mask.unsqueeze(1)
                else:
                    m = mask
                diff = (pred_kspace - ref_kspace) * m
                dc_loss = torch.norm(diff) ** 2
            else:
                diff = pred_kspace - ref_kspace
                # ``mse_loss`` has no complex kernel; |z|² is taken on (Re, Im).
                dc_loss = (
                    torch.view_as_real(diff).pow(2).sum(-1).mean()
                    if diff.is_complex()
                    else self.mse(pred_kspace, ref_kspace)
                )

        # Placeholder for smoothness / TV / etc.
        reg_loss = torch.tensor(0.0, device=pred_image.device)

        total_loss = (
            self.lambda_l2 * l2_image + self.lambda_dc * dc_loss + self.lambda_reg * reg_loss
        )

        return {
            "total": total_loss,
            "l2_image": l2_image,
            "dc_loss": dc_loss,
            "regularization": reg_loss,
        }


__all__ = ["PhysicsInformedLoss"]
