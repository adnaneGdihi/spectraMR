"""Phase-contrast / 4D-flow losses (regime: ``mri_flow``).

These are the regime-specific losses that (together with the phase-contrast
forward operator and the flow metrics) move ``mri_flow`` off EVAL_ONLY. Each is
tagged ``workflows={Regime.FLOW}`` so the maturity ledger sees real loss
coverage for the regime.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.signal_models.flow_encoding import (
    velocity_to_phase,
)
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="phase_contrast_velocity",
    domain="image",
    workflows=frozenset({Regime.FLOW}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class PhaseContrastVelocityLoss(nn.Module):
    """Velocity-consistency loss in the phase domain.

    Compares predicted and target velocity by first mapping both to encoded
    phase at the arm's ``venc`` — so the penalty lives on the physically
    meaningful, wrap-aware quantity the scanner actually measures.
    """

    def __init__(self, venc: float = 1.0, p: int = 1):
        super().__init__()
        if venc <= 0:
            raise ValueError(f"venc must be positive, got {venc}")
        if p not in (1, 2):
            raise ValueError(f"p must be 1 (L1) or 2 (L2), got {p!r}")
        self.venc = float(venc)
        self.p = p

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        venc: float | None = None,
    ) -> torch.Tensor:
        vc = float(venc) if venc is not None else self.venc
        phi_pred = velocity_to_phase(v_pred, vc)
        phi_target = velocity_to_phase(v_target, vc)
        diff = phi_pred - phi_target
        return diff.abs().mean() if self.p == 1 else (diff**2).mean()


@register_loss(
    name="through_plane_flux_conservation",
    domain="image",
    workflows=frozenset({Regime.FLOW}),
)
class ThroughPlaneFluxConservationLoss(nn.Module):
    """Penalise net through-plane flux imbalance between two planes.

    For an incompressible flow through a closed segment, inlet flux equals
    outlet flux; the residual is a physically-grounded regulariser. Flux is the
    mask-weighted sum of the through-plane velocity component.
    """

    def forward(
        self,
        v_through_plane: torch.Tensor,
        inlet_mask: torch.Tensor,
        outlet_mask: torch.Tensor,
        voxel_area: float = 1.0,
    ) -> torch.Tensor:
        inlet = (v_through_plane * inlet_mask).sum(dim=(-2, -1)) * voxel_area
        outlet = (v_through_plane * outlet_mask).sum(dim=(-2, -1)) * voxel_area
        return (inlet - outlet).abs().mean()


@register_loss(
    name="velocity_unwrap_consistency",
    domain="image",
    workflows=frozenset({Regime.FLOW}),
)
class VelocityUnwrapConsistencyLoss(nn.Module):
    """Penalise velocity values that exceed ``venc`` (i.e. would phase-wrap).

    A soft hinge on ``|v| > venc``: an estimate outside the unaliased range is
    physically inconsistent with the chosen encoding and should be discouraged
    (or the arm re-run at a higher venc).
    """

    def __init__(self, venc: float = 1.0):
        super().__init__()
        if venc <= 0:
            raise ValueError(f"venc must be positive, got {venc}")
        self.venc = float(venc)

    def forward(self, v_pred: torch.Tensor, venc: float | None = None) -> torch.Tensor:
        vc = float(venc) if venc is not None else self.venc
        excess = torch.clamp(v_pred.abs() - vc, min=0.0)
        return excess.mean()


__all__ = [
    "PhaseContrastVelocityLoss",
    "ThroughPlaneFluxConservationLoss",
    "VelocityUnwrapConsistencyLoss",
]
