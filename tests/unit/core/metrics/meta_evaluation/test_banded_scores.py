"""Discriminability, per-degradation sensitivity and the scale rule (banded_scores)."""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.core.metrics.meta_evaluation.banded_composite import (
    composite_score,
    parse_composite_weights,
)
from spectramr.core.metrics.meta_evaluation.banded_leaderboard import banded_leaderboard
from spectramr.core.metrics.meta_evaluation.banded_scores import cell_stats, reduce_scores
from spectramr.core.metrics.meta_evaluation.damage import DAMAGE_BANDS

tiered_ranks = pytest.importorskip("scripts.sim2rank.strata_report").tiered_ranks

BANDS = DAMAGE_BANDS["rel_l2"]
LEVELS = np.array([0.03, 0.05, 0.07, 0.09, 0.12, 0.18, 0.24, 0.35, 0.45, 0.6])


def _two_metrics(n_c: int = 2):
    """Metric 0 moves 1e-7 per level, metric 1 moves 10 per level; both strictly monotone."""
    step = np.arange(LEVELS.size, dtype=float)
    v = np.stack([1.0 + 1e-7 * step, 10.0 * step])[:, None, :].repeat(n_c, axis=1)
    return v, np.tile(LEVELS, (n_c, 1))


def _stats(values, damage, floors=None, hib=(False, False), real=None, names=None):
    n_c = next(iter(damage.values())).shape[0]
    return cell_stats(
        values, damage, floors or {}, hib, BANDS,
        metrics=names or tuple(f"m{i}" for i in range(len(hib))),
        real=np.ones(n_c, bool) if real is None else real,
    )  # fmt: skip


def test_discriminability_breaks_the_tau_ceiling() -> None:
    v, d = _two_metrics()
    floors = {a: np.full((2, 2), 0.01) for a in ("A", "B")}
    st = _stats({"A": v, "B": v}, {"A": d, "B": d}, floors)
    sc = reduce_scores(st, np.ones((1, 2)), min_axes=2)
    assert sc.tau[0].tolist() == [1.0, 1.0]
    assert sc.disc[0].tolist() == [0.0, 1.0]
    assert st.axis_mode == {"A": "measured", "B": "measured"}
    assert st.scale_source["A"] == {"m0": "measured", "m1": "measured"}


def test_reviewer_probe_unmeasured_metrics_are_incomplete_not_best() -> None:
    # A pooled metric moving 1e-3/step with no floor used to rank best on the rtol scale,
    # over a per-image metric moving 5 sigma/step with a measured floor.
    rng = np.random.default_rng(0)
    n_real, t = 4, LEVELS.size
    v = np.full((3, n_real + 1, t), np.nan)
    v[0, :n_real] = 5 * np.arange(t) + rng.normal(0, 1, (n_real, t))
    v[1, n_real] = 1e-3 * np.arange(t)
    v[2, :n_real] = 5 * np.arange(t) + rng.normal(0, 1, (n_real, t))
    d = np.tile(LEVELS, (n_real + 1, 1))
    fl = np.full((3, n_real + 1), np.nan)
    fl[0, :n_real] = 1.0
    st = _stats({"A": v, "B": v}, {"A": d, "B": d}, {"A": fl, "B": fl}, hib=(False,) * 3,
                real=np.r_[np.ones(n_real, bool), False], names=("img", "pooled", "img_nofloor"))  # fmt: skip
    sc = reduce_scores(st, np.ones((1, n_real + 1)), min_axes=2)
    assert sc.incomplete[0].tolist() == [False, True, True]
    assert np.isnan(sc.disc[0, 1:]).all() and np.isnan(sc.sens[0, 1:]).all()
    comp = composite_score(sc.tau, sc.disc, sc.sens, parse_composite_weights(None),
                           np.isfinite(sc.tau))  # fmt: skip
    assert tiered_ranks(comp, sc.incomplete)[0, 0] == 1
    assert st.scale_source["A"] == {"img": "measured", "pooled": "none", "img_nofloor": "none"}
    assert st.rtol_fraction["A"] == 0.0  # unmeasured entries are "none", not clamped


def test_deterministic_axis_scores_sensitivity_on_the_range_and_skips_discriminability() -> None:
    v, d = _two_metrics(1)
    zero = {a: np.zeros((2, 1)) for a in ("A", "B")}
    st = _stats({"A": v, "B": v}, {"A": d, "B": d}, zero)
    assert st.axis_mode == {"A": "deterministic", "B": "deterministic"}
    assert set(st.scale_source["A"].values()) == {"fraction_of_range"}
    assert st.rtol_fraction == {"A": 0.0, "B": 0.0}
    sc = reduce_scores(st, np.ones((1, 1)), min_axes=2)
    assert np.isnan(sc.disc[0]).all() and np.isfinite(sc.sens[0]).all()
    assert not sc.incomplete[0].any()  # an inapplicable component is not a missing one


def test_an_axis_without_floors_scores_neither() -> None:
    v, d = _two_metrics(1)
    st = _stats({"A": v, "B": v}, {"A": d, "B": d})
    assert st.axis_mode == {"A": "none", "B": "none"}
    sc = reduce_scores(st, np.ones((1, 1)), min_axes=2)
    assert np.isnan(sc.sens[0]).all() and sc.incomplete[0].all()


def test_discriminability_threshold_is_z_root2_floors() -> None:
    step = np.arange(LEVELS.size, dtype=float)
    v = np.stack([2.5 * step, 3.0 * step])[:, None, :]  # 2*sqrt(2) = 2.83 sits between
    fl = {a: np.ones((2, 1)) for a in ("A", "B")}
    sc = reduce_scores(_stats({"A": v, "B": v}, {"A": LEVELS[None], "B": LEVELS[None]}, fl),
                       np.ones((1, 1)), 2)  # fmt: skip
    assert sc.disc[0].tolist() == [0.0, 1.0]


def test_null_hit_rate_of_a_flat_metric_at_k3_is_controlled() -> None:
    # One-sided: a step counts only when the oriented metric rises past z·√2 floors, so the
    # baseline for a flat metric with a known sigma is Phi(-2) = 0.023 at z = 2.
    from scripts.sim2rank.response_floor import axis_response_floor

    rng = np.random.default_rng(1)
    n_c, t = 2000, LEVELS.size
    runs = rng.normal(0.0, 1.0, size=(4, 1, n_c, t))
    floor = np.stack([axis_response_floor(runs[0][:, c], [r[:, c] for r in runs[1:]])
                      for c in range(n_c)], axis=1)  # fmt: skip
    d = np.tile(LEVELS, (n_c, 1))
    st = _stats(
        {"A": runs[0], "B": runs[0]}, {"A": d, "B": d}, {"A": floor, "B": floor}, hib=(False,)
    )
    rate = reduce_scores(st, np.ones((1, n_c)), 2).disc[0, 0]
    assert 0.015 <= rate <= 0.035


def test_sensitivity_slope_is_invariant_to_n_steps() -> None:
    out = []
    for n in (5, 20):
        d = np.linspace(0.03, 0.9, n)[None]
        st = _stats({"A": (3.0 * d)[None]}, {"A": d}, {"A": np.full((1, 1), 0.5)}, hib=(False,))
        out.append(st.axis_slope["A"][0, 0])
    assert out[0] == pytest.approx(out[1]) == pytest.approx(6.0)


def test_unit_weights_reproduce_the_banded_tau() -> None:
    rng = np.random.default_rng(3)
    n_c, t = 3, LEVELS.size
    values = {a: rng.normal(size=(3, n_c, t)) + LEVELS for a in ("A", "B")}
    values["A"][1, 0, 2] = np.nan  # a partial row is scored on its finite levels
    values["B"][2, 1, :] = np.nan
    damage = {a: np.tile(LEVELS, (n_c, 1)) * (1 + 0.1 * np.arange(n_c))[:, None] for a in values}
    hib = (False, True, False)
    floors = {a: np.full((3, n_c), 0.1) for a in values}
    st = _stats(values, damage, floors, hib=hib)
    sc = reduce_scores(st, np.ones((1, n_c)), min_axes=2)
    flat = {f"m{m}": np.concatenate([values[a][m].ravel() for a in values]) for m in range(3)}
    axis = np.concatenate([np.full(n_c * t, a) for a in values])
    content = np.concatenate([np.repeat(np.arange(n_c), t) for _ in values])
    dmg = np.concatenate([damage[a].ravel() for a in values])
    ref = banded_leaderboard(flat, axis, content, dmg, dict(zip(flat, hib, strict=True)), BANDS)
    for m in range(3):
        assert sc.tau[0, m] == pytest.approx(ref.overall[f"m{m}"], abs=1e-12)
        # With every component measured, incomplete is exactly the board's tau gaps.
        assert bool(sc.incomplete[0, m]) == (f"m{m}" in ref.incomplete)
    # Component-aware: m0 without a floor on axis A keeps its tau but lacks a sensitivity.
    floors["A"][0] = np.nan
    gap = reduce_scores(_stats(values, damage, floors, hib=hib), np.ones((1, n_c)), min_axes=2)
    assert gap.incomplete[0, 0] and "m0" not in ref.incomplete
    assert gap.tau[0, 0] == pytest.approx(sc.tau[0, 0])
    for k, (a, band) in enumerate(st.cells):
        for m in range(3):
            assert bool(sc.tau_scored[0, m, k]) == ((f"m{m}", a, band) in ref.per_cell)


def test_a_pooled_stratum_takes_the_weight_given_and_never_covers() -> None:
    v = np.full((2, 2, LEVELS.size), np.nan)
    v[0, 0] = LEVELS  # a per-image metric on the real content
    v[1, 1] = LEVELS  # a distributional metric on the pooled stratum only
    d = np.tile(LEVELS, (2, 1))
    st = _stats({"A": v, "B": v}, {"A": d, "B": d}, real=np.array([True, False]))
    on, off, alone = reduce_scores(st, np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]), 2).tau
    assert on.tolist() == [1.0, 1.0]
    assert off[0] == 1.0 and np.isnan(off[1])
    assert np.isnan(alone).all()  # no real content: nothing is covered


def test_sensitivity_is_a_per_cell_percentile_and_excludes_circular_metrics() -> None:
    step = np.arange(LEVELS.size, dtype=float)
    v = np.stack([1.0 * step, 2.0 * step, 9.0 * step])[:, None, :]
    fl = {a: np.ones((3, 1)) for a in ("A", "B")}
    st = _stats({"A": v, "B": v}, {"A": LEVELS[None], "B": LEVELS[None]}, fl, hib=(False,) * 3)
    assert reduce_scores(st, np.ones((1, 1)), 2).sens[0].tolist() == [0.0, 0.5, 1.0]
    out = reduce_scores(st, np.ones((1, 1)), 2, exclude=np.array([False, False, True]))
    assert out.sens[0, :2].tolist() == [0.0, 1.0] and np.isnan(out.sens[0, 2])


def test_a_zero_floor_on_a_stochastic_axis_is_stamped_rtol_clamped_per_entry() -> None:
    # The reviewer's probe: B does not move across seeds (floor 0) on a measured axis.
    bands = (("light", 0.0, 10.0),)
    n, t = 3, 8
    d = np.tile(np.linspace(0, 1, t), (n, 1))
    rng = np.random.default_rng(0)
    a = d[None] + 0.01 * rng.normal(size=(1, n, t))
    c = 10 * d[None] + 0.5 * rng.normal(size=(1, n, t))
    v = np.concatenate([a, d[None], c])
    floors = {"noise": np.array([[0.01] * n, [0.0] * n, [0.5] * n])}
    st = cell_stats({"noise": v}, {"noise": d}, floors, [False] * 3, bands,
                    metrics=["A", "B", "C"], real=np.ones(n, bool))  # fmt: skip
    assert st.scale_source["noise"] == {"A": "measured", "B": "rtol_clamped", "C": "measured"}
    clamped = sum(v == "rtol_clamped" for v in st.scale_source["noise"].values())
    assert st.rtol_fraction["noise"] == pytest.approx(clamped / 3)
    sc = reduce_scores(st, np.ones((1, n)), min_axes=1)
    # Zero seed noise is real: B stays complete and keeps its (top) sensitivity percentile.
    assert not sc.incomplete[0].any() and sc.sens[0, 1] == 1.0


def test_a_clamped_floor_on_an_unscored_content_leaves_the_stamp_measured() -> None:
    # The stamp and rtol_fraction count the same entries: scored ones.
    v, d = _two_metrics()
    v[:, 1] = np.nan  # content 1 scored nothing
    fl = np.full((2, 2), 0.01)
    fl[:, 1] = 1e-30  # below rtol * span, so clamped, but on no scored entry
    st = _stats({"A": v, "B": v}, {"A": d, "B": d}, {"A": fl, "B": fl})
    assert st.rtol_fraction["A"] == 0.0
    assert st.scale_source["A"] == {"m0": "measured", "m1": "measured"}
