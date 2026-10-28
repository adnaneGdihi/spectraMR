"""Tests for ``EvidentialLoss``.

Targets ``spectramr.models.losses.evidential_loss``. Normal-Inverse-Gamma
(NIG) distribution loss for evidential regression. The network outputs
4 channels (γ, ν, α, β); this loss minimises NLL + evidence regulariser.

Categories:

- Unit: 4-channel constraint enforcement, NLL helpers, regulariser
- Property: identity (γ=target) gives finite loss; mismatched γ
  gives larger loss
- Property: gradient flow into γ, ν, α, β
- Sanity-shape: 4D + 5D inputs
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.losses.evidential_loss import EvidentialLoss


def _make_evidential_pred(
    target: torch.Tensor,
    nu: float = 1.0,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Helper: build a 4-channel evidential prediction from a target."""
    gamma = target.clone()
    nu_t = torch.full_like(target, nu)
    alpha_t = torch.full_like(target, alpha)
    beta_t = torch.full_like(target, beta)
    return torch.cat([gamma, nu_t, alpha_t, beta_t], dim=1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_construction() -> None:
    """Default ``coeff=1.0``."""
    loss = EvidentialLoss()
    assert loss.coeff == 1.0


def test_explicit_coeff_persisted() -> None:
    """Custom ``coeff`` is stored."""
    loss = EvidentialLoss(coeff=0.1)
    assert loss.coeff == 0.1


# ---------------------------------------------------------------------------
# Channel constraint
# ---------------------------------------------------------------------------


def test_wrong_channel_count_raises() -> None:
    """Prediction with channels != 4 → ``ValueError``."""
    loss = EvidentialLoss()
    pred = torch.randn(1, 3, 4, 4)  # 3 channels instead of 4
    target = torch.randn(1, 1, 4, 4)
    with pytest.raises(ValueError, match="exactly 4 predicted channels"):
        loss(pred, target)


# ---------------------------------------------------------------------------
# NLL & regulariser helpers
# ---------------------------------------------------------------------------


def test_nig_regularizer_zero_at_perfect_prediction() -> None:
    """``regulariser(y, y, ν, α, β) == 0`` (no error)."""
    loss = EvidentialLoss()
    y = torch.tensor([[[[1.0]]]])
    nu = torch.tensor([[[[1.0]]]])
    alpha = torch.tensor([[[[2.0]]]])
    beta = torch.tensor([[[[1.0]]]])
    reg = loss.nig_regularizer(y, gamma=y, nu=nu, alpha=alpha, beta=beta)
    assert reg.item() == 0.0


def test_nig_regularizer_grows_with_error() -> None:
    """Regulariser grows linearly with |y − γ|."""
    loss = EvidentialLoss()
    y = torch.tensor([[[[1.0]]]])
    gamma_close = torch.tensor([[[[0.9]]]])
    gamma_far = torch.tensor([[[[5.0]]]])
    nu = torch.tensor([[[[1.0]]]])
    alpha = torch.tensor([[[[2.0]]]])
    beta = torch.tensor([[[[1.0]]]])
    reg_close = loss.nig_regularizer(y, gamma_close, nu, alpha, beta).item()
    reg_far = loss.nig_regularizer(y, gamma_far, nu, alpha, beta).item()
    assert reg_far > reg_close


def test_nll_finite_for_valid_inputs() -> None:
    """NLL is finite for valid (positive) NIG parameters."""
    loss = EvidentialLoss()
    y = torch.full((1, 1, 4, 4), 1.0)
    gamma = torch.zeros_like(y)
    nu = torch.full_like(y, 1.0)
    alpha = torch.full_like(y, 2.0)
    beta = torch.full_like(y, 1.0)
    nll = loss.nll_loss(y, gamma, nu, alpha, beta)
    assert torch.isfinite(nll).all()


# ---------------------------------------------------------------------------
# Forward — sanity
# ---------------------------------------------------------------------------


def test_forward_returns_scalar() -> None:
    """Loss returns a scalar."""
    loss = EvidentialLoss()
    target = torch.randn(1, 1, 8, 8)
    pred = _make_evidential_pred(target)
    out = loss(pred, target)
    assert out.dim() == 0
    assert torch.isfinite(out)


def test_forward_grows_with_gamma_mismatch() -> None:
    """Larger γ-vs-target gap → larger loss (driven by regulariser)."""
    loss = EvidentialLoss(coeff=1.0)
    target = torch.zeros(1, 1, 4, 4)
    # Two prediction sets — same ν/α/β but different γ.
    pred_close = _make_evidential_pred(torch.full_like(target, 0.1))
    pred_far = _make_evidential_pred(torch.full_like(target, 5.0))
    loss_close = loss(pred_close, target).item()
    loss_far = loss(pred_far, target).item()
    assert loss_far > loss_close


def test_forward_5d_volumetric_input() -> None:
    """5D ``[B, 4, D, H, W]`` input also works."""
    loss = EvidentialLoss()
    target = torch.randn(1, 1, 4, 8, 8)
    pred = _make_evidential_pred(target)
    out = loss(pred, target)
    assert out.dim() == 0
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_prediction() -> None:
    """Backward through the loss reaches the prediction tensor."""
    loss = EvidentialLoss()
    target = torch.randn(1, 1, 4, 4)
    pred = _make_evidential_pred(target).detach().requires_grad_(True)
    loss(pred, target).backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# Numerical safety
# ---------------------------------------------------------------------------


def test_zero_inputs_clamped_to_safe_values() -> None:
    """Zero ν, α, β are clamped → no NaN."""
    loss = EvidentialLoss()
    target = torch.zeros(1, 1, 4, 4)
    pred = torch.zeros(1, 4, 4, 4)  # all zeros, including ν, α, β
    out = loss(pred, target)
    # Should be finite thanks to internal clamping.
    assert torch.isfinite(out)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_evidential_registered() -> None:
    """``evidential`` is in the loss registry."""
    from spectramr.models.losses.registry import list_available

    assert "evidential" in list_available()


def test_nig_params_override_a_one_channel_mean() -> None:
    """The strategy passes gamma as ``prediction`` and the NIG tensor as ``nig_params``."""
    target = torch.rand(2, 1, 8, 8)
    full = _make_evidential_pred(target + 0.1)
    loss = EvidentialLoss(coeff=0.01)
    torch.testing.assert_close(loss(full[:, :1], target, nig_params=full), loss(full, target))


def test_a_one_channel_prediction_without_nig_params_raises() -> None:
    with pytest.raises(ValueError, match="no nig_params"):
        EvidentialLoss()(torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4))
