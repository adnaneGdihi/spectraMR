"""Tests for :mod:`spectramr.infrastructure.reporting.dashboard.charts`."""

from __future__ import annotations

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.dashboard import charts


def _long() -> pd.DataFrame:
    rows = [
        ("a", "psnr", 30.0, "higher"),
        ("b", "psnr", 31.0, "higher"),
        ("c", "psnr", 29.0, "higher"),
        ("a", "nmse", 0.01, "lower"),
        ("b", "nmse", 0.03, "lower"),
        ("c", "nmse", 0.02, "lower"),
        ("a", "loss", 1.0, None),
        ("b", "loss", 2.0, None),
        ("c", "loss", 3.0, None),
    ]
    return pd.DataFrame(
        [
            {"arm": a, "phase": "test", "metric": m, "mean": v, "std": 1.0, "n": 10, "direction": d}
            for a, m, v, d in rows
        ]
    )


def test_directions_leave_out_undeclared_metrics():
    assert charts.directions(_long()) == {"psnr": "higher", "nmse": "lower"}


def test_rank_table_respects_direction_and_orders_best_first():
    ranks = charts.rank_table(_long(), "test")
    assert ranks.loc["b", "psnr"] == 1 and ranks.loc["a", "nmse"] == 1
    assert list(ranks.index) == ["a", "b", "c"]  # mean ranks 1.5, 2.0, 2.5


def test_rank_table_restricted_to_chosen_metrics():
    assert list(charts.rank_table(_long(), "test", ["psnr"]).index) == ["b", "a", "c"]


def test_best_arm_follows_direction():
    assert charts.best_arm(_long(), "test", "psnr") == "b"
    assert charts.best_arm(_long(), "test", "nmse") == "a"
    assert charts.best_arm(_long(), "test", "loss") is None


def _stats(metric: str, mean_diff: float, scale: str = "metric") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "variant": "v",
                "mean_diff": mean_diff,
                "ci_low": mean_diff - 0.1,
                "ci_high": mean_diff + 0.1,
                "cohens_d": mean_diff * 2,
                "p_holm": 0.01,
                "scale": scale,
            }
        ]
    )


def test_effects_are_signed_so_positive_means_better():
    dirs = {"psnr": "higher", "nmse": "lower"}
    assert charts.oriented_effects(_stats("psnr", 0.5), dirs)["d"].iloc[0] == pytest.approx(1.0)
    assert charts.oriented_effects(_stats("nmse", 0.5), dirs)["d"].iloc[0] == pytest.approx(-1.0)


def test_a_declared_test_scale_overrides_the_metric_direction():
    # nex_equivalent is higher-is-better, but its paired test runs on 1/N_eq, where lower is
    # better: a positive difference there is a worse arm and must plot left of zero.
    eff = charts.oriented_effects(
        _stats("nex_equivalent", 0.5, "1/N_eq (lower is better)"), {"nex_equivalent": "higher"}
    )
    assert eff["d"].iloc[0] == pytest.approx(-1.0)
    assert eff["lo"].iloc[0] <= eff["d"].iloc[0] <= eff["hi"].iloc[0]


def test_undirected_metrics_are_dropped_from_effects():
    assert charts.oriented_effects(_stats("loss", 0.5), {}).empty


def test_leaderboard_div_marks_baseline_and_best_by_role():
    pytest.importorskip("plotly")
    div = charts.leaderboard_div(_long(), phase="test", metrics=["psnr"], baseline="a")
    assert div is not None and "plotly-graph-div" in div
    # rows ordered b, a, c on psnr: b is best (accent), a is the baseline
    assert '"marker_roles":["accent","baseline","series"]' in div.replace(" ", "")
