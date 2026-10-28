"""Tests for the mri_flow regime losses."""

from __future__ import annotations

import torch

from spectramr.models.losses.flow_losses import (
    PhaseContrastVelocityLoss,
    ThroughPlaneFluxConservationLoss,
    VelocityUnwrapConsistencyLoss,
)
from spectramr.models.losses.registry import LossRegistry


def test_flow_losses_registered_and_tagged() -> None:
    from spectramr.config.schemas.enums import Regime

    meta = LossRegistry._loss_domains
    for name in (
        "phase_contrast_velocity",
        "through_plane_flux_conservation",
        "velocity_unwrap_consistency",
    ):
        assert name in meta
        assert Regime.FLOW in (meta[name].get("workflows") or frozenset())


def test_phase_contrast_velocity_zero_on_match() -> None:
    loss = PhaseContrastVelocityLoss(venc=100.0)
    v = torch.randn(2, 3, 8, 8)
    assert loss(v, v).item() == 0.0


def test_phase_contrast_velocity_positive_on_mismatch() -> None:
    loss = PhaseContrastVelocityLoss(venc=100.0)
    v = torch.randn(2, 3, 8, 8)
    assert loss(v, v + 5.0).item() > 0.0


def test_flux_conservation_zero_when_balanced() -> None:
    loss = ThroughPlaneFluxConservationLoss()
    v = torch.ones(1, 1, 4, 4)
    inlet = torch.zeros(1, 1, 4, 4)
    inlet[..., 0, :] = 1.0
    outlet = torch.zeros(1, 1, 4, 4)
    outlet[..., -1, :] = 1.0
    # Equal masks over a uniform field -> balanced flux.
    assert loss(v, inlet, inlet).item() == 0.0
    # Different rows of a uniform field still balance (same count).
    assert loss(v, inlet, outlet).item() == 0.0


def test_unwrap_penalty_only_beyond_venc() -> None:
    loss = VelocityUnwrapConsistencyLoss(venc=10.0)
    within = torch.full((4,), 5.0)
    assert loss(within).item() == 0.0
    beyond = torch.full((4,), 15.0)
    assert loss(beyond).item() > 0.0


def test_phase_contrast_velocity_rejects_an_unknown_norm() -> None:
    """``p`` other than 1 or 2 used to fall through to the L2 branch silently."""
    import pytest

    with pytest.raises(ValueError, match="p must be"):
        PhaseContrastVelocityLoss(venc=1.0, p=3)
