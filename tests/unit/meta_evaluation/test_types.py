"""Tests for the shared meta-evaluation dataclasses (``types.py``).

Focus: ``RankingResult.standardize`` is the z-scoring chokepoint every ranker's
scores flow through before aggregation. A metric that crashed (recorded as NaN
per the crash→NaN contract) must not poison the z-scores of the *other* metrics.
"""

from __future__ import annotations

import math

import pytest

from spectramr.core.metrics.meta_evaluation.types import (
    AggregateRankingResult,
    RankingResult,
)


def _rr(scores: dict[str, float]) -> RankingResult:
    return RankingResult(method="m", scores=scores, ranks=[], diagnostics={})


def test_aggregate_result_defaults_to_breadth_semantics() -> None:
    # The defensibility-flag semantics field (critique A2) defaults to "breadth"
    # (the z-aggregator's rule); the Kemeny aggregator overrides it to "condorcet".
    agg = AggregateRankingResult(
        final_scores={}, final_ranks=[], per_method_z={},
        defensibility_flags={}, weights={},
    )
    assert agg.defensibility_semantics == "breadth"


def test_standardize_basic_zscore() -> None:
    # Two finite scores -> standard z-scores (mean 0, unit population std).
    z = _rr({"b": 1.0, "c": 0.5}).standardize()
    assert z["b"] == pytest.approx(1.0)
    assert z["c"] == pytest.approx(-1.0)


def test_standardize_nan_score_does_not_poison_others() -> None:
    # crash→NaN contract: one NaN-scored metric must not turn every z-score NaN.
    z = _rr({"a": float("nan"), "b": 1.0, "c": 0.5}).standardize()
    # The crashed metric is placed one unit below the worst finite z.
    assert z["a"] == pytest.approx(-2.0)
    # The finite metrics keep their correct, finite z-scores.
    assert math.isfinite(z["b"]) and math.isfinite(z["c"])
    assert z["b"] == pytest.approx(1.0)
    assert z["c"] == pytest.approx(-1.0)


def test_standardize_inf_score_is_kept_as_best() -> None:
    z = _rr({"a": float("inf"), "b": 2.0, "c": 0.0}).standardize()
    assert z["a"] == pytest.approx(2.0)
    assert all(math.isfinite(v) for v in z.values())


def test_standardize_places_universe_metrics_absent_from_scores_as_worst() -> None:
    z = _rr({"b": 1.0, "c": 0.5}).standardize(["b", "c", "gone"])
    assert z["gone"] == pytest.approx(-2.0)


def test_standardize_all_nonfinite_keeps_inf_above_nan() -> None:
    # No finite score to standardize: +inf still outranks a crash.
    assert _rr({"a": float("nan"), "b": float("inf")}).standardize() == {"a": -1.0, "b": 1.0}


def test_standardize_degenerate_constant_unchanged() -> None:
    # Pre-existing contract: constant finite scores -> all-zero z (std < 1e-12).
    z = _rr({"a": 3.0, "b": 3.0}).standardize()
    assert z == {"a": 0.0, "b": 0.0}
