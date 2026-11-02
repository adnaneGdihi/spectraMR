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
    venc_on_axis,
    wrap_phase,
)
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="phase_contrast_velocity",
    domain="image",
    workflows=frozenset({Regime.FLOW}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class PhaseContrastVelocityLoss(nn.Module):
    """Velocity-consistency loss on the wrapped encoded-phase difference.

    Both fields are mapped to phase at the arm's ``venc`` and their difference is
    wrapped to ``(-pi, pi]``, so two velocities that alias to the same signal
    (``v`` and ``v - 2 venc``) score zero. ``weight`` (e.g. reference magnitude
    times a vessel mask) turns the mean into a weighted mean, so voxels whose
    phase is noise do not dominate.
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
        venc: float | torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Channel-first [B, n_dir, ...]: a per-direction venc goes on axis 1.
        vc = venc_on_axis(venc if venc is not None else self.venc, v_pred.ndim, 1)
        diff = wrap_phase(velocity_to_phase(v_pred, vc) - velocity_to_phase(v_target, vc))
        err = diff.abs() if self.p == 1 else diff**2
        if weight is None:
            return err.mean()
        w = torch.broadcast_to(weight.to(err.dtype), err.shape)
        return (w * err).sum() / w.sum().clamp_min(torch.finfo(err.dtype).eps)


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

    def forward(
        self, v_pred: torch.Tensor, venc: float | torch.Tensor | None = None
    ) -> torch.Tensor:
        vc = venc_on_axis(venc if venc is not None else self.venc, v_pred.ndim, 1)
        excess = torch.clamp(v_pred.abs() - vc, min=0.0)
        return excess.mean()


def _per_sample_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    num, den = num.flatten(1).sum(dim=1), den.flatten(1).sum(dim=1)
    return (num / den.clamp_min(torch.finfo(den.dtype).tiny)).mean()


@register_loss(
    name="phase_contrast_image_nmse",
    domain="complex",
    workflows=frozenset({Regime.FLOW}),
    tasks=frozenset({Task.RECONSTRUCTION}),
)
class PhaseContrastImageNMSELoss(nn.Module):
    """``sum |x_hat - x|^2 / sum |x|^2`` per sample over the encoded complex images.

    Scale-free, so its weight means the same on noise-whitened scanner data and
    on the phantom. Inputs are complex ``[B, E, ...]``.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return _per_sample_ratio((pred - target).abs().pow(2), target.abs().pow(2))


@register_loss(
    name="phase_contrast_phasor",
    domain="complex",
    workflows=frozenset({Regime.FLOW}),
    tasks=frozenset({Task.RECONSTRUCTION, Task.PARAMETER_MAPPING}),
)
class PhaseContrastPhasorLoss(nn.Module):
    """Agreement of the velocity-encoded phases, without ``angle()``.

    For each direction ``z = x_d conj(x_ref)`` (encoding axis 1, reference first),
    the loss is ``sum(|z_p||z_r| - Re(z_p conj z_r)) / sum |z_r|^2``, i.e. the
    reference-weighted ``1 - cos`` of the phase error: wrap-free, blind to any
    background phase (it multiplies both phasors), and smooth where ``angle()``'s
    gradient grows as ``1 / |z|``. It can also fall by shrinking ``|z_p|``, so a
    strategy pairs it with an image term.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        z_p = pred[:, 1:] * pred[:, :1].conj()
        z_r = target[:, 1:] * target[:, :1].conj()
        misfit = z_p.abs() * z_r.abs() - (z_p * z_r.conj()).real
        return _per_sample_ratio(misfit, z_r.abs().pow(2))


__all__ = [
    "PhaseContrastImageNMSELoss",
    "PhaseContrastPhasorLoss",
    "PhaseContrastVelocityLoss",
    "ThroughPlaneFluxConservationLoss",
    "VelocityUnwrapConsistencyLoss",
]
