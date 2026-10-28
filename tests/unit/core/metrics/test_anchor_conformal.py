"""Tests for fiducial-calibrated conformal prediction intervals.

Two things must hold for this construction to be worth anything: the intervals
must actually cover at the nominal rate on exchangeable data, and the coverage
gate must FAIL when exchangeability breaks. A calibrator that always reports a
guarantee is not a guarantee, so the failing cases below matter more than the
passing ones.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.core.metrics.anchor_conformal import (  # noqa: E402
    AnchorConformalCalibrator,
    conformal_quantile,
    local_detail_score,
)


def _heteroscedastic(n: int = 4000, scale: float = 2.0, seed: int = 0):
    """Residuals whose spread grows with difficulty — the realistic case, and
    the one an unconditional interval gets wrong."""
    torch.manual_seed(seed)
    d = torch.rand(n)
    return torch.randn(n) * (0.1 + scale * d), d


# ── the quantile ─────────────────────────────────────────────────────────────


def test_finite_sample_correction_delivers_marginal_coverage() -> None:
    """The plain (1-alpha) empirical quantile under-covers at small n. This is
    the correction that makes the guarantee hold at the sizes this cohort has."""
    torch.manual_seed(0)
    hits = sum(
        int(torch.randn(1).abs() <= conformal_quantile(torch.randn(40).abs(), 0.1))
        for _ in range(2000)
    )
    assert 0.86 <= hits / 2000 <= 0.94


def test_too_few_points_returns_infinity_rather_than_a_false_promise() -> None:
    """With n < 1/alpha - 1 no finite interval can carry the level, and saying
    so is more useful than returning the largest observed residual."""
    assert torch.isinf(conformal_quantile(torch.randn(5).abs(), 0.01))


def test_quantile_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="alpha must be"):
        conformal_quantile(torch.randn(10).abs(), 1.0)
    with pytest.raises(ValueError, match="empty"):
        conformal_quantile(torch.empty(0), 0.1)


# ── the difficulty score ─────────────────────────────────────────────────────


def test_detail_score_is_larger_at_edges_than_in_flat_regions() -> None:
    x = torch.zeros(1, 1, 16, 16)
    x[..., 8:, :] = 1.0
    s = local_detail_score(x)
    assert float(s[..., 8, :].mean()) > float(s[..., 2, :].mean())
    assert float(s.min()) >= 0.0
    assert float(s.max()) == pytest.approx(1.0)


def test_detail_score_rejects_the_wrong_rank() -> None:
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        local_detail_score(torch.randn(4, 4))


# ── calibration ──────────────────────────────────────────────────────────────


def test_stratified_calibration_covers_within_tolerance() -> None:
    r_cal, d_cal = _heteroscedastic(seed=0)
    r_te, d_te = _heteroscedastic(seed=1)
    cal = AnchorConformalCalibrator(alpha=0.1, n_strata=4).fit(r_cal, d_cal)
    report = cal.coverage(r_te, d_te)
    assert report.guaranteed
    assert float(report.coverage.min()) > 0.85


@pytest.mark.parametrize("scale", [0.01, 5.0])
def test_gate_fails_over_and_under_coverage(scale: float) -> None:
    """Planted violations: intervals far too wide and far too narrow both fail."""
    r_cal, d_cal = _heteroscedastic(seed=0)
    r_te, d_te = _heteroscedastic(seed=1)
    cal = AnchorConformalCalibrator(alpha=0.1, n_strata=4).fit(r_cal, d_cal)
    report = cal.coverage(r_te * scale, d_te)
    assert not report.guaranteed


def test_half_widths_increase_with_difficulty() -> None:
    """The point of conditioning: a flat region gets a tight interval and an
    edge gets a wide one, rather than both getting the average."""
    cal = AnchorConformalCalibrator(alpha=0.1, n_strata=4).fit(*_heteroscedastic())
    q = cal.quantiles
    assert torch.all(q[1:] > q[:-1])


def test_marginal_calibration_undercovers_the_hard_stratum() -> None:
    """n_strata=1 is not a harmless simplification. A single quantile
    over-covers the easy regime and under-covers the hard one, which is exactly
    the regime a reader cares about."""
    r_cal, d_cal = _heteroscedastic(seed=0)
    r_te, d_te = _heteroscedastic(seed=1)
    marginal = AnchorConformalCalibrator(alpha=0.1, n_strata=1).fit(r_cal, d_cal)
    stratified = AnchorConformalCalibrator(alpha=0.1, n_strata=4).fit(r_cal, d_cal)
    # evaluate the marginal quantile against the stratified partition
    probe = AnchorConformalCalibrator(alpha=0.1, n_strata=4)
    probe._edges = stratified._edges
    probe._quantiles = marginal.quantiles.repeat(4)
    probe._counts = stratified.counts
    cov = probe.coverage(r_te, d_te).coverage
    assert float(cov[0]) > 0.97  # easiest stratum: far over-covered
    assert float(cov[-1]) < 0.85  # hardest: under-covered
    assert not probe.coverage(r_te, d_te).guaranteed


# ── the gate must be able to fail ────────────────────────────────────────────


def test_gate_fails_when_exchangeability_breaks() -> None:
    """The decisive test. If the calibration population stops resembling the
    test population, the guarantee is void and must be REPORTED void."""
    cal = AnchorConformalCalibrator(alpha=0.1, n_strata=4).fit(*_heteroscedastic(seed=0))
    r_shift, d_shift = _heteroscedastic(scale=6.0, seed=1)
    report = cal.coverage(r_shift, d_shift)
    assert not report.guaranteed
    assert float(report.coverage.max()) < 0.9


def test_guarantee_requires_every_stratum_not_the_average() -> None:
    """Marginal coverage can look fine while one difficulty regime collapses."""
    cal = AnchorConformalCalibrator(alpha=0.1, n_strata=2).fit(*_heteroscedastic())
    from spectramr.core.metrics.anchor_conformal import CoverageReport

    mixed = CoverageReport(
        nominal=0.9,
        tolerance=0.05,
        counts=torch.tensor([100, 100]),
        coverage=torch.tensor([1.00, 0.60]),  # mean 0.80, one stratum collapsed
        passed=torch.tensor([True, False]),
    )
    assert not mixed.guaranteed
    assert cal is not None


def test_unmeasured_stratum_is_nan_not_perfect() -> None:
    """ "No data" and "perfect coverage" must not look the same in a report."""
    cal = AnchorConformalCalibrator(alpha=0.1, n_strata=4).fit(*_heteroscedastic())
    r, d = _heteroscedastic(n=200, seed=2)
    report = cal.coverage(r[d < 0.2], d[d < 0.2])
    assert bool(torch.isnan(report.coverage).any())


# ── contract ─────────────────────────────────────────────────────────────────


def test_support_restricts_calibration_to_where_truth_is_known() -> None:
    """Off the marker the "truth" is a zero background, and including it would
    shrink every interval toward nothing."""
    torch.manual_seed(0)
    r = torch.cat([torch.randn(500) * 3.0, torch.zeros(4500)])
    d = torch.rand(5000)
    support = torch.cat([torch.ones(500), torch.zeros(4500)])
    wide = AnchorConformalCalibrator(alpha=0.1, n_strata=1).fit(r, d, support)
    narrow = AnchorConformalCalibrator(alpha=0.1, n_strata=1).fit(r, d)
    assert float(wide.quantiles[0]) > 10 * float(narrow.quantiles[0])


def test_empty_support_raises() -> None:
    with pytest.raises(ValueError, match="no calibration points"):
        AnchorConformalCalibrator().fit(torch.randn(100), torch.rand(100), torch.zeros(100))


def test_empty_stratum_raises_rather_than_borrowing() -> None:
    """Borrowing another stratum's quantile would report a conditional
    guarantee that was never conditioned."""
    d = torch.zeros(50)
    with pytest.raises(ValueError, match="no calibration points"):
        AnchorConformalCalibrator(n_strata=4).fit(torch.randn(50), d)


def test_using_the_calibrator_before_fitting_raises() -> None:
    cal = AnchorConformalCalibrator()
    with pytest.raises(RuntimeError, match="not been fitted"):
        _ = cal.quantiles


def test_interval_brackets_the_prediction_symmetrically() -> None:
    cal = AnchorConformalCalibrator(alpha=0.1, n_strata=2).fit(*_heteroscedastic())
    pred = torch.randn(1, 1, 8, 8)
    diff = torch.rand(1, 1, 8, 8)
    lo, hi = cal.interval(pred, diff)
    assert torch.allclose(hi - pred, pred - lo)
    assert torch.all(hi >= pred) and torch.all(lo <= pred)


def test_report_dict_exposes_every_stratum_not_just_the_verdict() -> None:
    cal = AnchorConformalCalibrator(alpha=0.1, n_strata=3).fit(*_heteroscedastic())
    d = cal.coverage(*_heteroscedastic(seed=1)).as_dict()
    assert sum(k.startswith("conformal_coverage_s") for k in d) == 3
    assert "conformal_guaranteed" in d and "conformal_worst_coverage" in d
