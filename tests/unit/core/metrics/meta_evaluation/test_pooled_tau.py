"""Pooled Kendall τ-b: pair counts summed over strata, non-finite samples dropped per metric."""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.core.metrics.meta_evaluation import pooled_tau
from spectramr.core.metrics.meta_evaluation.iqa_recalibration import kendall_tau
from spectramr.core.metrics.meta_evaluation.pooled_tau import (
    PooledTau,
    measured_circular,
    tau_b_rows,
    within_content_tau,
)


def test_chunked_rows_match_one_block(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    d = np.round(rng.uniform(0, 1, 12), 1)
    x = np.round(rng.normal(size=(7, 12)), 0)
    whole = tau_b_rows(x, d)
    monkeypatch.setattr(pooled_tau, "_CHUNK_ELEMS", 1)
    np.testing.assert_allclose(tau_b_rows(x, d), whole, atol=0)
    np.testing.assert_allclose(whole, [kendall_tau(r, d) for r in x], atol=1e-12)


def test_nan_sample_drops_only_for_that_metric() -> None:
    d = np.array([0.1, 0.2, 0.3, 0.4])
    x = np.stack([d, d.copy()])
    x[1, 0] = np.nan
    acc = PooledTau(2)
    acc.add(x, d, min_levels=3)
    np.testing.assert_allclose(acc.tau(), [1.0, 1.0])
    assert acc.nd.tolist() == [6.0, 3.0]  # per-metric pair count, not one per stratum


def test_stratum_with_too_few_finite_levels_does_not_vote() -> None:
    d = np.array([0.1, 0.2, 0.3])
    x = np.stack([d, np.array([np.nan, 0.2, 0.3])])
    acc = PooledTau(2)
    acc.add(x, d, min_levels=3)
    assert acc.voted.tolist() == [True, False]


def test_within_content_tau_pools_axes_but_not_contents() -> None:
    # Within each content the metric orders every sample; across contents it is offset.
    d = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    content = np.array([0, 0, 0, 1, 1, 1])
    x = d + np.array([0, 0, 0, 5.0, 5.0, 5.0])
    assert within_content_tau({"m": x}, d, content)["m"] == pytest.approx(1.0)


def test_measured_circular_rejects_a_bad_threshold() -> None:
    d = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="threshold"):
        measured_circular({"m": d}, d, np.array(["a", "b", "b"]), np.zeros(3), threshold=1.5)


def test_sort_path_matches_the_sign_matrix_with_ties(monkeypatch) -> None:
    rng = np.random.default_rng(2)
    d = np.round(rng.uniform(0, 1, 80), 1)
    x = np.round(rng.normal(size=(5, 80)), 0)
    x[0] = 3.0  # constant row
    monkeypatch.setattr(pooled_tau, "_SORT_PATH_PAIRS", 10**9)
    matrix = pooled_tau.tau_parts(x, d)
    monkeypatch.setattr(pooled_tau, "_SORT_PATH_PAIRS", 0)
    sorted_ = pooled_tau.tau_parts(x, d)
    np.testing.assert_array_equal(sorted_[0], matrix[0])
    np.testing.assert_array_equal(sorted_[1], matrix[1])
    assert sorted_[2] == matrix[2]


def test_sort_path_with_constant_damage_skips_scipy(monkeypatch) -> None:
    import warnings

    monkeypatch.setattr(pooled_tau, "_SORT_PATH_PAIRS", 0)
    x = np.arange(12.0).reshape(2, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s, _nx, nd = pooled_tau.tau_parts(x, np.full(6, 0.5))
    assert nd == 0.0 and s.tolist() == [0.0, 0.0]


def test_measured_circular_returns_the_flagged_taus() -> None:
    d = np.array([0.1, 0.2, 0.3, 0.15, 0.25, 0.35])
    axis = np.array(["a"] * 3 + ["b"] * 3)
    got = measured_circular({"same": d.copy(), "flip": -d}, d, axis, np.zeros(6))
    assert got == {"same": pytest.approx(1.0), "flip": pytest.approx(-1.0)}
