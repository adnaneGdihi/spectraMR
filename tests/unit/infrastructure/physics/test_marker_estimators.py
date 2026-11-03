r"""Tests for the synthetic-marker estimator utilities.

Targets ``spectramr.infrastructure.physics.marker_estimators``.

Categories:

- ``crlb_from_fisher`` recovers the analytic inverse for diagonal info
- ``fisher_information_motion`` produces a positive-semi-definite matrix
- ``greedy_marker_placement`` selects ``n_select`` indices and
  monotonically reduces the CRLB trace
- ``split_conformal_marker_uq`` quantile sits at the documented
  ``ceil((n+1)(1-α))`` rank
- ``split_conformal_marker_uq`` rejects a too-small calibration set
- Coverage of the produced band on a fresh test split is at least the
  documented ``1 - α - 1/(n+1)``
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.marker_estimators import (
    ConformalResult,
    GreedyPlacementResult,
    conformal_coverage_check,
    crlb_from_fisher,
    fisher_information_motion,
    greedy_marker_placement,
    split_conformal_marker_uq,
)


# ---------------------------------------------------------------------------
# Fisher / CRLB
# ---------------------------------------------------------------------------


def test_crlb_inverts_diagonal_fisher() -> None:
    """``CRLB = I⁻¹`` for a diagonal Fisher info."""
    fisher = torch.diag(torch.tensor([4.0, 9.0, 16.0]))
    crlb = crlb_from_fisher(fisher, jitter=0.0)
    expected = torch.diag(torch.tensor([0.25, 1.0 / 9, 1.0 / 16]))
    assert torch.allclose(crlb, expected, atol=1e-5)


def test_fisher_information_is_psd() -> None:
    """Fisher information for a linear forward model is PSD."""
    p = 4
    M = 8
    A = torch.randn(M, p)

    def forward(theta: torch.Tensor) -> torch.Tensor:
        return torch.complex(A @ theta, torch.zeros(M))

    theta = torch.randn(p, requires_grad=True)
    fi = fisher_information_motion(forward, theta, noise_cov=1.0)
    eigs = torch.linalg.eigvalsh(fi)
    # Allow a tiny negative slack for floating-point noise.
    assert (eigs >= -1e-5).all()


# ---------------------------------------------------------------------------
# Greedy marker placement
# ---------------------------------------------------------------------------


def test_greedy_returns_documented_count() -> None:
    """``selected_indices`` length equals ``min(n_select, K)``."""
    K, M, p = 6, 3, 2
    torch.manual_seed(0)
    cand = torch.randn(K, M, p)
    out = greedy_marker_placement(cand, n_select=4, noise_var=1.0)
    assert isinstance(out, GreedyPlacementResult)
    assert len(out.selected_indices) == 4
    assert len(out.fisher_history) == 4
    assert len(out.crlb_trace_history) == 4


def test_greedy_indices_unique() -> None:
    """No marker is selected twice."""
    torch.manual_seed(0)
    cand = torch.randn(8, 3, 3)
    out = greedy_marker_placement(cand, n_select=5)
    assert len(set(out.selected_indices)) == len(out.selected_indices)


def test_greedy_crlb_trace_non_increasing() -> None:
    """CRLB trace decreases monotonically (more markers = lower variance)."""
    torch.manual_seed(0)
    cand = torch.randn(8, 4, 3)
    out = greedy_marker_placement(cand, n_select=5)
    for prev, nxt in zip(out.crlb_trace_history, out.crlb_trace_history[1:]):
        assert nxt <= prev + 1e-6


def test_greedy_n_select_clipped_to_k() -> None:
    """If ``n_select > K``, only ``K`` markers are returned."""
    cand = torch.randn(3, 2, 2)
    out = greedy_marker_placement(cand, n_select=10)
    assert len(out.selected_indices) == 3


# ---------------------------------------------------------------------------
# Split-conformal marker UQ
# ---------------------------------------------------------------------------


def test_conformal_quantile_at_documented_rank() -> None:
    """``hat_q_α`` sits at the order-``ceil((n+1)(1-α))`` calibration value.

    With ``n = 100`` residuals and ``α = 0.1``, the rank is
    ``ceil(101 · 0.9) = 91``. Sorted ascending and 0-indexed that's
    index 90.
    """
    residuals = torch.linspace(0.0, 1.0, 100)
    pred = torch.zeros(4, 4)
    out = split_conformal_marker_uq(pred, residuals, alpha=0.1)
    expected = float(torch.sort(residuals)[0][90].item())
    assert pytest.approx(out.quantile, abs=1e-6) == expected


def test_conformal_too_few_residuals_rejected() -> None:
    """``n < 2`` calibration residuals → ValueError."""
    with pytest.raises(ValueError, match="≥ 2"):
        split_conformal_marker_uq(torch.zeros(4), torch.zeros(1))


def test_conformal_band_symmetric() -> None:
    """``upper - pred = pred - lower = q``."""
    pred = torch.linspace(0.0, 1.0, 8)
    residuals = torch.full((100,), 0.3)
    out = split_conformal_marker_uq(pred, residuals, alpha=0.1)
    assert torch.allclose(out.coverage_upper - pred, pred - out.coverage_lower)


def test_conformal_n_calibration_recorded() -> None:
    """``n_calibration`` reflects the input residual count."""
    residuals = torch.randn(123)
    out = split_conformal_marker_uq(torch.zeros(2, 2), residuals)
    assert out.n_calibration == 123


def test_conformal_marginal_coverage_holds_on_held_out() -> None:
    """Empirical coverage on a fresh test set ≥ ``1 - α - 1/(n+1)``."""
    torch.manual_seed(0)
    n_cal, n_test = 500, 5000
    alpha = 0.1
    cal_residuals = torch.randn(n_cal).abs()
    truth = torch.randn(n_test)
    pred = torch.zeros_like(truth)
    out = split_conformal_marker_uq(pred, cal_residuals, alpha=alpha)
    # Compare signed truth against the symmetric band around pred (= 0 here).
    # Earlier code used ``truth.abs()`` and ``coverage_lower.abs()`` which
    # collapsed the interval into a single point (|lower| == |upper| at pred=0)
    # and produced 0% empirical coverage.
    coverage = conformal_coverage_check(
        truth, out.coverage_lower, out.coverage_upper
    )
    # The plan's bound is 1 - α - 1/(n+1); allow a small empirical slack.
    bound = 1.0 - alpha - 1.0 / (n_cal + 1) - 0.05
    assert coverage >= bound, (
        f"empirical coverage {coverage:.3f} below bound {bound:.3f}"
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


def test_conformal_result_is_dataclass() -> None:
    """``ConformalResult`` exposes ``quantile``, ``n_calibration``, bounds."""
    out = split_conformal_marker_uq(
        torch.zeros(4), torch.linspace(0.0, 1.0, 50), alpha=0.1
    )
    assert isinstance(out, ConformalResult)
    assert hasattr(out, "quantile")
    assert hasattr(out, "n_calibration")
    assert hasattr(out, "coverage_lower")
    assert hasattr(out, "coverage_upper")


# ---------------------------------------------------------------------------
# privileged_learning_loss: per-sample weights
# ---------------------------------------------------------------------------


def test_privileged_loss_sample_weight_drops_zero_weight_samples() -> None:
    from spectramr.infrastructure.physics.marker_estimators import privileged_learning_loss

    student = torch.randn(4, 1, 3, 3)
    teacher = torch.randn(4, 1, 3, 3)
    target = torch.randn(4, 1, 3, 3)
    weight = torch.tensor([1.0, 1.0, 0.0, 0.0])
    weighted = privileged_learning_loss(student, teacher, target, sample_weight=weight)
    subset = privileged_learning_loss(student[:2], teacher[:2], target[:2])
    assert torch.allclose(weighted["g_priv_recon"], subset["g_priv_recon"])
    assert torch.allclose(weighted["g_priv_distill"], subset["g_priv_distill"])
    uniform = privileged_learning_loss(student, teacher, target, sample_weight=torch.ones(4))
    plain = privileged_learning_loss(student, teacher, target)
    assert torch.allclose(uniform["g_total_loss"], plain["g_total_loss"])
    with pytest.raises(ValueError, match="sample_weight"):
        privileged_learning_loss(student, teacher, target, sample_weight=torch.ones(3))
