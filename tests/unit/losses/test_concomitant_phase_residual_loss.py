"""Unit tests for concomitant_phase_residual_loss.py."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.concomitant_phase_residual_loss import (
    ConcomitantPhaseResidualLoss,
)
from spectramr.models.losses.registry import create_loss

B, C, H, W = 2, 1, 16, 16


def _phase(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Residual phase in radians; plausible range [-0.5, 0.5]
    return (torch.rand(b, c, h, w, generator=gen) - 0.5)


class TestConcomitantPhaseResidualLoss:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_finite(self):
        loss_fn = ConcomitantPhaseResidualLoss()
        phi = _phase()
        out = loss_fn(residual_phase=phi)
        assert torch.isfinite(out) and out.shape == ()

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("weight", [0.1, 1.0, 5.0])
    def test_parametric_weight(self, weight: float):
        loss_fn = ConcomitantPhaseResidualLoss(weight=weight)
        phi = _phase()
        out = loss_fn(residual_phase=phi)
        assert torch.isfinite(out) and out.item() >= 0.0

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reduction(self, reduction: str):
        loss_fn = ConcomitantPhaseResidualLoss(reduction=reduction)
        phi = _phase()
        out = loss_fn(residual_phase=phi)
        assert torch.isfinite(out)

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_zero_residual_gives_zero_loss(self):
        """Residual = 0 → loss = λ * mean(0^2) = 0."""
        loss_fn = ConcomitantPhaseResidualLoss()
        phi = torch.zeros(B, C, H, W)
        out = loss_fn(residual_phase=phi)
        assert out.item() == pytest.approx(0.0, abs=1e-9)

    def test_property_nonneg(self):
        """Squared L2 is always ≥ 0."""
        loss_fn = ConcomitantPhaseResidualLoss()
        phi = _phase()
        assert loss_fn(residual_phase=phi).item() >= 0.0

    def test_property_scales_linearly_with_weight(self):
        phi = _phase()
        out_1 = ConcomitantPhaseResidualLoss(weight=1.0)(residual_phase=phi)
        out_2 = ConcomitantPhaseResidualLoss(weight=2.0)(residual_phase=phi)
        assert out_2.item() == pytest.approx(2.0 * out_1.item(), rel=1e-5)

    def test_property_gradient_reaches_input(self):
        phi = _phase().requires_grad_(True)
        loss_fn = ConcomitantPhaseResidualLoss()
        loss_fn(residual_phase=phi).backward()
        assert phi.grad is not None and torch.isfinite(phi.grad).all()

    def test_positional_call_raises(self):
        """The (prediction, target) call a loss list makes is refused (#2275)."""
        loss_fn = ConcomitantPhaseResidualLoss()
        phi = _phase()
        with pytest.raises(TypeError, match="residual_phase="):
            loss_fn(phi, torch.rand_like(phi))

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_large_phase_residual(self):
        loss_fn = ConcomitantPhaseResidualLoss()
        phi = torch.full((B, C, H, W), 1e3)
        out = loss_fn(residual_phase=phi)
        assert torch.isfinite(out)

    def test_edge_3d_input(self):
        loss_fn = ConcomitantPhaseResidualLoss()
        phi = torch.randn(B, C, 8, H, W)
        out = loss_fn(residual_phase=phi)
        assert torch.isfinite(out)

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_negative_weight(self):
        with pytest.raises(ValueError, match="weight"):
            ConcomitantPhaseResidualLoss(weight=-0.1)

    def test_raises_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            ConcomitantPhaseResidualLoss(reduction="max_xyz")

    # ── Registry ────────────────────────────────────────────────────────────

    def test_registry_lookup(self):
        loss_fn = create_loss("concomitant_phase_residual")
        assert isinstance(loss_fn, ConcomitantPhaseResidualLoss)
