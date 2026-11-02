"""The composite rank over τ, discriminability and sensitivity (banded_composite)."""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.core.metrics.meta_evaluation.banded_composite import (
    competition_ranks,
    composite_score,
    direction_ambiguous,
    parse_composite_weights,
    pool_strata,
    sensitivity_table,
)


def test_composite_ties_share_a_rank_and_missing_is_worst() -> None:
    tau = np.array([1.0, 1.0, 0.5, 0.2])
    disc = np.array([0.4, 0.4, 0.9, np.nan])
    sens = np.array([2.0, 2.0, 1.0, 0.5])
    w = parse_composite_weights(None)
    comp = composite_score(tau, disc, sens, w, ranked=np.ones(4, bool))
    assert comp[0] == comp[1]
    assert competition_ranks(-comp).tolist() == [1, 1, 3, 4]
    assert competition_ranks(np.array([0.5, np.nan, 0.5])).tolist()[::2] == [1, 1]


def test_composite_weights_parse_validate_and_normalise() -> None:
    assert parse_composite_weights(None) == pytest.approx(
        {"tau": 1 / 3, "discriminability": 1 / 3, "sensitivity": 1 / 3}
    )
    w = parse_composite_weights("tau=2,discriminability=1,sensitivity=1")
    assert w == pytest.approx({"tau": 0.5, "discriminability": 0.25, "sensitivity": 0.25})
    with pytest.raises(ValueError, match="unknown"):
        parse_composite_weights("tau=1,speed=1")
    with pytest.raises(ValueError, match="negative"):
        parse_composite_weights("tau=-1,sensitivity=1")
    with pytest.raises(ValueError):
        parse_composite_weights("tau=0")


def test_direction_ambiguous_needs_opposite_families() -> None:
    per_cell = {
        ("sharp", "noise", "light"): 0.8,
        ("sharp", "blur", "light"): -0.7,
        ("good", "noise", "light"): 0.8,
        ("good", "blur", "light"): 0.6,
    }
    fam = {"noise": "noise", "blur": "resolution"}
    out = direction_ambiguous(per_cell, fam, ("light",))
    assert out == {"sharp": True, "good": False}


def test_sensitivity_table_uses_competition_ranks() -> None:
    table = sensitivity_table({"A": np.array([2.0, 2.0, 1.0, np.nan])}, ("a", "b", "c", "d"))
    assert table["A"] == [("a", 2.0, 1), ("b", 2.0, 1), ("c", 1.0, 3)]


def test_minus_inf_ranks_last_and_nan_is_unranked() -> None:
    assert competition_ranks(np.array([3.0, 3.0, 1.0, np.nan, -np.inf])).tolist()[:3] == [1, 1, 3]
    r = competition_ranks(np.array([3.0, np.nan, -np.inf]))
    assert r[2] == 2 and np.isnan(r[1])


def test_pool_strata_means_over_strata_and_flags_partial_metrics() -> None:
    tau = np.array([[1.0, 0.2, np.nan], [0.0, 0.4, 0.5]])  # (C=2, M=3)
    inc = np.array([[False, True, False], [False, False, False]])
    t, _d, _s, i = pool_strata(tau, tau, tau, inc)
    assert t.tolist() == pytest.approx([0.5, 0.3, 0.5])
    assert i.tolist() == [False, True, True]
    one = pool_strata(tau[:1], tau[:1], tau[:1], inc[:1])
    assert np.array_equal(one[0], tau[0], equal_nan=True) and one[3].tolist() == inc[0].tolist()
