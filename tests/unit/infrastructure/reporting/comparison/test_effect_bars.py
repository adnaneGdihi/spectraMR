"""``comparison/effect_bars.py``: bars are the table's mean difference, whiskers its interval."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from spectramr.infrastructure.reporting.catalog.examples_payloads import (
    make_paired_rows,
    make_paired_subjects,
)
from spectramr.infrastructure.reporting.comparison.effect_bars import _draw, make

ARMS = ["a", "b", "c"]
GROUPS = {"a": "baseline", "b": "grp1", "c": "grp2"}


def _fig(rows=None, groups=GROUPS):
    rows = make_paired_rows() if rows is None else rows
    fig = _draw(["psnr"], rows, make_paired_subjects(), ARMS, groups, "a")
    return fig, next(a for a in fig.axes if a.get_visible()), rows


def test_each_bar_height_is_the_csv_mean_difference():
    fig, ax, rows = _fig()
    one = rows[rows["metric"] == "psnr"]
    expected = sorted(one["mean_diff"])
    assert sorted(p.get_height() for p in ax.patches) == pytest.approx(expected)
    plt.close(fig)


def test_the_whisker_is_the_csv_interval_not_one_recomputed_from_the_data():
    fig, ax, rows = _fig()
    one = rows[rows["metric"] == "psnr"]
    spans = [
        (float(seg[:, 1].min()), float(seg[:, 1].max()))
        for lc in ax.collections
        if hasattr(lc, "get_segments")
        for seg in lc.get_segments()
    ]
    assert spans
    for low, high in zip(sorted(one["ci_low"]), sorted(one["ci_high"]), strict=True):
        assert any(abs(lo - low) < 1e-9 and abs(hi - high) < 1e-9 for lo, hi in spans)
    plt.close(fig)


def test_inference_bars_are_hatched_and_validation_bars_are_solid():
    fig, ax, rows = _fig()
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert len(hatched) == len(rows[(rows["metric"] == "psnr") & (rows["phase"] == "test")])
    assert len(ax.patches) - len(hatched) == len(
        rows[(rows["metric"] == "psnr") & (rows["phase"] == "val")]
    )
    plt.close(fig)


def test_a_bar_whose_adjusted_p_is_not_below_one_in_twenty_is_faded():
    rows = make_paired_rows().copy()
    rows["p_holm"] = 0.01
    fig, ax, _ = _fig(rows)
    assert {p.get_alpha() for p in ax.patches} == {1.0}
    plt.close(fig)
    rows["p_holm"] = 0.5
    fig, ax, _ = _fig(rows)
    assert {p.get_alpha() for p in ax.patches} == {0.4}
    plt.close(fig)


def test_the_baseline_has_no_bar_and_an_ungrouped_arm_joins_arms():
    fig, ax, rows = _fig(groups=None)
    assert len(ax.patches) == len(rows[rows["metric"] == "psnr"])
    assert "a" not in [t.get_text() for t in ax.get_xticklabels()]
    plt.close(fig)


def test_make_writes_the_figure_and_gives_none_without_statistics(tmp_path):
    out = make(
        None,
        tmp_path / "f",
        paired_df=make_paired_rows(),
        paired=make_paired_subjects(),
        arms=ARMS,
        groups=GROUPS,
        baseline="a",
        formats=("png",),
        dpi=50,
    )
    assert out is not None and out.exists()
    assert make(None, tmp_path / "g", paired_df=None, arms=ARMS, baseline="a") is None
