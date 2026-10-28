"""``comparison/metric_group_bars.py``: direction-aligned percent improvement over the baseline."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.catalog.examples_payloads import (
    make_paired_rows,
    make_paired_subjects,
)
from spectramr.infrastructure.reporting.comparison.metric_group_bars import (
    improvement_rows,
    make,
)
from spectramr.infrastructure.reporting.per_subject_stats import PairedSubjects


def _pair(a, b, scale="metric"):
    table = pd.DataFrame({"a": a, "b": b}, index=pd.Index([f"s{i}" for i in range(len(a))]))
    return PairedSubjects(
        table=table,
        n_slices=len(a),
        nonfinite_dropped={"a": 0, "b": 0},
        unpaired_slices={"a": 0, "b": 0},
        scale=scale,
    )


def _row(metric, diff, lo, hi, phase="val", variant="b"):
    return {
        "phase": phase,
        "metric": metric,
        "variant": variant,
        "mean_diff": diff,
        "ci_low": lo,
        "ci_high": hi,
    }


def _rows(entries):
    paired_df = pd.DataFrame([e[0] for e in entries])
    paired = {(e[0]["phase"], e[0]["metric"], e[0]["variant"]): e[1] for e in entries}
    return improvement_rows(paired_df, paired)


def test_a_gain_is_positive_for_higher_better_and_for_lower_better_metrics():
    rows, omitted = _rows(
        [
            (_row("psnr", 3.0, 2.0, 4.0), _pair([33.0] * 4, [30.0] * 4)),
            (_row("nmse", -0.01, -0.02, 0.0), _pair([0.04] * 4, [0.05] * 4)),
        ]
    )
    pct = rows.set_index("metric")["pct"]
    assert pct["psnr"] == pytest.approx(10.0)  # 100 * 3 / 30
    assert pct["nmse"] == pytest.approx(20.0)  # sign flipped, 100 * 0.01 / 0.05
    assert omitted == {}


def test_a_negative_sign_swaps_the_interval_so_that_lo_is_below_hi():
    rows, _ = _rows([(_row("nmse", -0.01, -0.02, 0.0), _pair([0.04] * 4, [0.05] * 4))])
    r = rows.iloc[0]
    assert r["lo"] == pytest.approx(0.0) and r["hi"] == pytest.approx(40.0)


def test_a_reciprocal_metric_is_read_on_the_reciprocal_scale_with_the_opposite_direction():
    # nex_equivalent is higher-better in N_eq, so on 1/N_eq a decrease is the improvement
    rows, _ = _rows(
        [(_row("nex_equivalent", -0.1, -0.2, 0.0), _pair([0.3] * 4, [0.4] * 4, "1/N_eq"))]
    )
    assert rows.iloc[0]["pct"] == pytest.approx(25.0)


def test_metrics_without_a_direction_or_with_a_zero_baseline_are_named_and_not_drawn():
    rows, omitted = _rows(
        [
            (_row("not_a_metric_xyz", 1.0, 0.0, 2.0), _pair([1.0] * 4, [0.5] * 4)),
            (_row("psnr", 1.0, 0.0, 2.0), _pair([1.0] * 4, [0.0] * 4)),
        ]
    )
    assert rows.empty
    assert set(omitted) == {"not_a_metric_xyz", "psnr"}
    assert "no declared direction" in omitted["not_a_metric_xyz"]
    assert "zero" in omitted["psnr"]


def test_make_writes_one_figure_and_gives_none_when_nothing_has_a_percentage(tmp_path):
    out = make(
        None,
        tmp_path / "f",
        paired_df=make_paired_rows(),
        paired=make_paired_subjects(),
        arms=["a", "b", "c"],
        baseline="a",
        formats=("png",),
        dpi=50,
    )
    assert out is not None and out.exists()
    none = make(
        None,
        tmp_path / "g",
        paired_df=pd.DataFrame([_row("not_a_metric_xyz", 1.0, 0.0, 2.0)]),
        paired={("val", "not_a_metric_xyz", "b"): _pair([1.0] * 4, [0.5] * 4)},
        arms=["a", "b"],
        baseline="a",
    )
    assert none is None


def test_more_metrics_than_the_limit_continue_in_numbered_files(tmp_path):
    metrics = ["psnr", "ssim", "nmse", "mse", "nrmse", "lpips", "fsim", "vif", "mae", "hfen"]
    entries = [(_row(m, 1.0, 0.5, 1.5), _pair([2.0] * 4, [1.0] * 4)) for m in metrics]
    out = make(
        None,
        tmp_path / "f",
        paired_df=pd.DataFrame([e[0] for e in entries]),
        paired={("val", e[0]["metric"], "b"): e[1] for e in entries},
        arms=["a", "b"],
        baseline="a",
        max_metrics=4,
        formats=("png",),
        dpi=50,
    )
    assert out is not None
    assert (tmp_path / "f_2.png").exists()
