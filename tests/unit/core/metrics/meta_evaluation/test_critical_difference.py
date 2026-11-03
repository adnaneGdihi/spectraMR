"""Tests for :mod:`spectramr.core.metrics.meta_evaluation.critical_difference`."""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.core.metrics.meta_evaluation.critical_difference import (
    cliques,
    friedman_p,
    nemenyi_cd,
    rank_rows,
)


def test_ranks_follow_direction_share_ties_and_sink_nan():
    np.testing.assert_allclose(rank_rows(np.array([[3.0, 1.0, 1.0]])), [[1, 2.5, 2.5]])
    np.testing.assert_allclose(
        rank_rows(np.array([[3.0, 1.0, 2.0]]), higher_is_better=False), [[3, 1, 2]]
    )
    np.testing.assert_allclose(rank_rows(np.array([[np.nan, 1.0, 2.0]])), [[3, 2, 1]])


def test_nemenyi_cd_matches_demsar_table():
    # Demšar (2006) Table 5a: q_0.05 = 2.343 for k = 3.
    assert nemenyi_cd(3, 10) == pytest.approx(2.343 * np.sqrt(3 * 4 / 60), rel=1e-3)


def test_friedman_detects_a_consistent_order_and_skips_two_methods():
    ranks = rank_rows(np.tile([3.0, 2.0, 1.0], (12, 1)))
    assert friedman_p(ranks) < 1e-4
    assert np.isnan(friedman_p(ranks[:, :2]))


def test_cliques_join_close_neighbours_only():
    assert cliques(np.array([1.0, 1.4, 2.9, 3.1]), cd=0.5) == [(0, 1), (2, 3)]
