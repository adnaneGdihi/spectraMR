"""Tests for ``ConcomitantPhaseResidualLoss``.

Targets ``spectramr.models.losses.concomitant_phase_residual_loss``. CAUR /
idea 1 §1.2 residual ridge penalty.

Categories:

- Construction: invalid weight / reduction rejected
- Identity (zero residual) → loss = 0
- Loss grows quadratically with residual scale
- Reduction modes: mean / sum
- Gradient flows back to the residual tensor
- A positional (prediction, target) call raises
- Registry: ``concomitant_phase_residual`` resolvable
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.concomitant_phase_residual_loss import (
    ConcomitantPhaseResidualLoss,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_negative_weight_rejected() -> None:
    """``weight < 0`` raises."""
    with pytest.raises(ValueError, match="weight"):
        ConcomitantPhaseResidualLoss(weight=-0.1)


def test_invalid_reduction_rejected() -> None:
    """Only 'mean' and 'sum' are accepted."""
    with pytest.raises(ValueError, match="reduction"):
        ConcomitantPhaseResidualLoss(reduction="median")


# ---------------------------------------------------------------------------
# Closed-form
# ---------------------------------------------------------------------------


def test_zero_residual_yields_zero_loss() -> None:
    """``Δφ = 0`` → ``L = 0``."""
    loss = ConcomitantPhaseResidualLoss(weight=2.0)
    assert pytest.approx(loss(residual_phase=torch.zeros(2, 1, 4, 4)).item(), abs=1e-7) == 0.0


def test_loss_quadratic_in_residual_scale() -> None:
    """Doubling the residual quadruples the mean-squared loss."""
    loss = ConcomitantPhaseResidualLoss(weight=1.0)
    base = torch.randn(1, 1, 4, 4)
    val_1 = loss(residual_phase=base).item()
    val_2 = loss(residual_phase=2 * base).item()
    assert pytest.approx(val_2) == 4.0 * val_1


def test_weight_multiplies_loss() -> None:
    """Doubling the weight doubles the loss."""
    base = torch.randn(1, 1, 4, 4)
    val_1 = ConcomitantPhaseResidualLoss(weight=1.0)(residual_phase=base).item()
    val_3 = ConcomitantPhaseResidualLoss(weight=3.0)(residual_phase=base).item()
    assert pytest.approx(val_3) == 3.0 * val_1


# ---------------------------------------------------------------------------
# Reduction modes
# ---------------------------------------------------------------------------


def test_sum_reduction_scales_with_count() -> None:
    """``sum`` reduction yields ``N · mean``."""
    base = torch.full((1, 1, 4, 4), 0.5)
    n = base.numel()
    mean_loss = ConcomitantPhaseResidualLoss(reduction="mean")(residual_phase=base).item()
    sum_loss = ConcomitantPhaseResidualLoss(reduction="sum")(residual_phase=base).item()
    assert pytest.approx(sum_loss) == mean_loss * n


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


def test_gradient_flows_to_residual() -> None:
    """Backward through the loss reaches the residual tensor."""
    loss = ConcomitantPhaseResidualLoss()
    residual = torch.randn(1, 1, 4, 4, requires_grad=True)
    loss(residual_phase=residual).backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


# ---------------------------------------------------------------------------
# Loss-list calls are refused (#2275)
# ---------------------------------------------------------------------------


def test_positional_prediction_target_call_raises() -> None:
    """A loss list's ``loss(pred, target)`` call raises instead of training mean(pred^2).

    Called that way the old forward treated the prediction as the residual phase:
    its value equalled mean(pred^2) and did not depend on the target.
    """
    pred = torch.randn(1, 1, 4, 4)
    loss = ConcomitantPhaseResidualLoss()
    with pytest.raises(TypeError, match="residual_phase="):
        loss(pred, torch.randn_like(pred))
    with pytest.raises(TypeError, match="residual_phase="):
        loss(pred)


def test_missing_residual_phase_raises() -> None:
    with pytest.raises(TypeError, match="residual_phase="):
        ConcomitantPhaseResidualLoss()(target=torch.zeros(1))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered() -> None:
    """``concomitant_phase_residual`` resolvable in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "concomitant_phase_residual" in list_available()
