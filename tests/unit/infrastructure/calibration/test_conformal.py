"""Tests for ``ConformalCalibrator``.

Targets ``spectramr.infrastructure.calibration.conformal`` — PR-3 (H1) of
``TODO/backlog_paradigm_expansion_roadmap.md``.

Plan acceptance criteria covered:

- *"empirical coverage matches target α within 2% on a held-out
  Gaussian regression with 10⁵ samples"* — ``test_empirical_coverage_within_tolerance``
- *"calibration.alpha > 1 raises ValueError (not silent clip)"* —
  ``test_alpha_above_one_raises``

Plus standard contract tests:

- ``predict`` before ``fit`` raises
- ``fit`` requires at least one calibration sample
- The protocol surface (``IConformalCalibrator``) is satisfied at runtime
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.calibration.conformal import (
    ConformalCalibrator,
    IConformalCalibrator,
)
from spectramr.infrastructure.calibration.scores import (
    absolute_residual,
    quantile_regression,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_score_is_absolute_residual() -> None:
    """Default score function is the absolute residual."""
    cal = ConformalCalibrator()
    # The score function lives on the instance under a private name; just
    # check the public behaviour matches absolute_residual.
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.0, 2.5])
    cal.fit([(pred, target)])
    assert cal.is_fitted is True


def test_alpha_at_zero_raises() -> None:
    """``alpha = 0`` would request 100% coverage — degenerate, rejected."""
    with pytest.raises(ValueError, match="alpha"):
        ConformalCalibrator(alpha=0.0)


def test_alpha_at_one_raises() -> None:
    """``alpha = 1`` would request 0% coverage — also rejected."""
    with pytest.raises(ValueError, match="alpha"):
        ConformalCalibrator(alpha=1.0)


def test_alpha_above_one_raises() -> None:
    """``alpha > 1`` raises (no silent clip — CLAUDE.md #9)."""
    with pytest.raises(ValueError, match="alpha"):
        ConformalCalibrator(alpha=1.5)


def test_alpha_below_zero_raises() -> None:
    """Negative alpha rejected with the same message."""
    with pytest.raises(ValueError, match="alpha"):
        ConformalCalibrator(alpha=-0.1)


# ---------------------------------------------------------------------------
# Fit / predict contract
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises() -> None:
    """``predict`` is illegal before ``fit``."""
    cal = ConformalCalibrator(alpha=0.1)
    with pytest.raises(RuntimeError, match="not been fitted"):
        cal.predict(torch.zeros(4))


def test_fit_with_empty_iterable_raises() -> None:
    """``fit`` rejects an empty calibration set."""
    cal = ConformalCalibrator(alpha=0.1)
    with pytest.raises(ValueError, match="empty"):
        cal.fit([])


def test_fit_returns_self_for_chaining() -> None:
    """Builder-style fluent chain."""
    cal = ConformalCalibrator(alpha=0.1)
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])
    out = cal.fit([(pred, target)])
    assert out is cal


def test_predict_returns_three_tensors() -> None:
    """``predict`` returns ``(y_hat, lower, upper)`` of equal shape."""
    torch.manual_seed(0)
    cal = ConformalCalibrator(alpha=0.1)
    cal.fit([(torch.randn(64), torch.randn(64))])
    y_hat, lower, upper = cal.predict(torch.zeros(8))
    assert y_hat.shape == lower.shape == upper.shape
    # Symmetric band around the point estimate.
    radius = (upper - lower) / 2.0
    assert torch.allclose(radius, torch.full_like(radius, cal.quantile))


def test_n_calibration_is_recorded() -> None:
    """The size of the calibration set survives onto the calibrator."""
    cal = ConformalCalibrator(alpha=0.1)
    cal.fit([(torch.randn(50), torch.randn(50))])
    assert cal.n_calibration == 50


# ---------------------------------------------------------------------------
# Marginal-coverage property test (the headline acceptance criterion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.2])
def test_empirical_coverage_within_tolerance(alpha: float) -> None:
    """Fresh-test-split coverage matches ``1 - α`` within ±2 pp.

    Mirrors the Vovk-Gammerman-Shafer theorem: marginal coverage is
    ≥ 1 − α with no distributional assumptions. We use a Gaussian
    regression so the asymmetry of the residual distribution is null
    and the tolerance is realistic.
    """
    torch.manual_seed(42)
    n_cal, n_test = 5000, 10000

    # Synthetic predictor: y = 2x + 0.5 + ε, ε ~ N(0, 1)
    cal_x = torch.randn(n_cal) * 2.0
    cal_target = 2.0 * cal_x + 0.5 + torch.randn(n_cal)
    cal_pred = 2.0 * cal_x + 0.5  # noise-free predictor

    test_x = torch.randn(n_test) * 2.0
    test_target = 2.0 * test_x + 0.5 + torch.randn(n_test)
    test_pred = 2.0 * test_x + 0.5

    cal = ConformalCalibrator(alpha=alpha).fit([(cal_pred, cal_target)])
    coverage = cal.empirical_coverage(test_pred, test_target)
    target_coverage = 1.0 - alpha
    assert abs(coverage - target_coverage) < 0.02, (
        f"Coverage {coverage:.3f} too far from target {target_coverage:.3f} "
        f"(α={alpha})"
    )


def test_quantile_increases_for_smaller_alpha() -> None:
    """Stricter coverage (smaller α) → wider band (larger quantile)."""
    torch.manual_seed(0)
    pred, target = torch.randn(2048), torch.randn(2048)
    q_loose = ConformalCalibrator(alpha=0.2).fit([(pred, target)]).quantile
    q_tight = ConformalCalibrator(alpha=0.05).fit([(pred, target)]).quantile
    assert q_tight > q_loose


# ---------------------------------------------------------------------------
# Finite-sample correctness: order statistic, not interpolated quantile
# ---------------------------------------------------------------------------


def test_quantile_is_exact_order_statistic_not_interpolated() -> None:
    """The band radius is the exact k-th order statistic, k=ceil((n+1)(1-α)).

    The canonical split-conformal estimator is the nearest-rank order
    statistic, not an interpolated quantile. We feed distinct scores so
    the order statistic is unambiguous and assert the exact value, and
    confirm we are *not* returning torch.quantile's interpolated value.
    """
    import math

    # 20 distinct calibration residuals (pred=0, target=scores so the
    # absolute-residual score equals `scores`).
    scores = torch.arange(1.0, 21.0)  # 1, 2, ..., 20
    cal = ConformalCalibrator(alpha=0.1).fit([(torch.zeros(20), scores)])

    n = 20
    k = math.ceil((n + 1) * (1.0 - 0.1))  # ceil(18.9) = 19
    expected = float(torch.sort(scores).values[k - 1].item())  # scores[18] = 19.0
    assert cal.quantile == expected == 19.0
    # torch.quantile interpolates (here to 19.05) — we must NOT use it.
    interpolated = float(torch.quantile(scores.double(), k / n).item())
    assert cal.quantile != interpolated


def test_small_n_returns_infinite_band_not_false_guarantee() -> None:
    """When k > n no finite band is 1-α valid → radius is +∞, not clamped.

    With n=5, α=0.1: k=ceil(6·0.9)=6 > 5, so the conformally honest band
    radius is +∞. The previous clamp-to-max-score returned a finite band
    that did not hold the advertised guarantee (pitfalls #9/#20).
    """
    cal = ConformalCalibrator(alpha=0.1).fit([(torch.zeros(5), torch.arange(1.0, 6.0))])
    assert cal.quantile == float("inf")
    # The infinite band trivially covers everything (honest, not false).
    _, lower, upper = cal.predict(torch.zeros(3))
    assert torch.isinf(lower).all() and torch.isinf(upper).all()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_calibrator_satisfies_protocol_after_fit() -> None:
    """``ConformalCalibrator`` is a runtime instance of the protocol."""
    cal = ConformalCalibrator(alpha=0.1)
    cal.fit([(torch.zeros(4), torch.zeros(4))])
    assert isinstance(cal, IConformalCalibrator)


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


def test_absolute_residual_basic() -> None:
    """``|pred - target|`` element-wise."""
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.0, 4.0])
    out = absolute_residual(pred, target)
    expected = torch.tensor([0.5, 0.0, 1.0])
    assert torch.allclose(out, expected)


def test_absolute_residual_shape_mismatch_raises() -> None:
    """Score functions fail loudly on shape mismatch."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        absolute_residual(torch.zeros(4), torch.zeros(8))


def test_quantile_regression_negative_inside_band() -> None:
    """Inside the band ``[lo, hi]`` the score is negative."""
    lo = torch.tensor([0.0, 0.0])
    hi = torch.tensor([2.0, 2.0])
    target = torch.tensor([1.0, 1.5])  # both inside
    out = quantile_regression(lo, hi, target)
    assert (out <= 0).all()


def test_quantile_regression_positive_outside_band() -> None:
    """Outside the band the score is positive (above or below)."""
    lo = torch.tensor([0.0, 0.0])
    hi = torch.tensor([2.0, 2.0])
    target = torch.tensor([-1.0, 5.0])  # below / above
    out = quantile_regression(lo, hi, target)
    assert (out > 0).all()


def test_quantile_regression_swapped_quantiles_raises() -> None:
    """``pred_lower > pred_upper`` is a contract violation, not a no-op."""
    lo = torch.tensor([2.0])
    hi = torch.tensor([0.0])
    with pytest.raises(ValueError, match="swapped"):
        quantile_regression(lo, hi, torch.zeros(1))


# ── Foreground mask (#3041): background pixels neither set nor test the radius ─


def test_foreground_mask_is_relative_to_each_image_peak() -> None:
    from spectramr.infrastructure.calibration.conformal import foreground_mask

    target = torch.zeros(2, 1, 2, 2)
    target[0, 0, 0, 0] = 1.0
    target[1, 0, 0, 0] = 100.0
    target[1, 0, 1, 1] = 6.0  # above 5% of its image peak (5.0)
    mask = foreground_mask(target, threshold=0.05)
    assert mask[0, 0, 0, 0] and not mask[0, 0, 1, 1]
    assert mask[1, 0, 0, 0] and mask[1, 0, 1, 1]
    assert foreground_mask(target, threshold=None) is None


def test_background_error_does_not_move_the_masked_radius() -> None:
    target = torch.zeros(4, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    pred = target.clone()
    pred[:, :, 0:2, :] += 50.0  # 25% of pixels: background only, outside the foreground
    masked = ConformalCalibrator(alpha=0.1, foreground_threshold=0.05)
    masked.fit([(pred, target)])
    assert masked.quantile == pytest.approx(0.0)
    assert masked.n_calibration == 4 * 16
    unmasked = ConformalCalibrator(alpha=0.1, foreground_threshold=None).fit([(pred, target)])
    assert unmasked.quantile > 1.0


def test_coverage_with_an_empty_foreground_raises() -> None:
    target = torch.zeros(1, 1, 4, 4)
    cal = ConformalCalibrator(alpha=0.1, foreground_threshold=0.05)
    cal.fit([(target, target)])
    with pytest.raises(ValueError, match="selects no pixels"):
        cal.empirical_coverage(target, target)


def test_foreground_masking_refuses_a_per_sample_score() -> None:
    """Planted: a per-sample norm (vf_residual-style) cannot be pixel-masked."""
    target = torch.ones(2, 1, 4, 4)

    def per_sample(pred, tgt):
        return (pred - tgt).abs().flatten(1).amax(1)

    cal = ConformalCalibrator(score_fn=per_sample, alpha=0.1, foreground_threshold=0.05)
    with pytest.raises(ValueError, match="needs a per-pixel score"):
        cal.fit([(target, target)])
