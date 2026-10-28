"""``comparison/effect_forest.py``: the figure draws the numbers of paired_stats.csv."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_paired_rows
from spectramr.infrastructure.reporting.comparison.effect_forest import adjusted_p, make

ARMS = ["a", "b", "c"]


def test_the_marker_reports_the_adjusted_p_when_there_is_one():
    row = pd.Series({"p_ttest": 0.01, "p_holm": 0.04})
    assert adjusted_p(row) == 0.04
    assert adjusted_p(pd.Series({"p_ttest": 0.01, "p_holm": np.nan})) == 0.01
    assert adjusted_p(pd.Series({"p_ttest": 0.01, "p_holm": None})) == 0.01


def test_the_figure_is_written_from_the_paired_rows(tmp_path):
    rows = make_paired_rows()
    assert {"val", "test"} == set(rows["phase"])
    out = make(
        pd.DataFrame(),
        tmp_path / "f.png",
        paired_df=rows,
        arms=ARMS,
        baseline="a",
        formats=("png",),
        dpi=40,
    )
    assert out is not None and out.is_file()


def test_rows_without_a_finite_interval_are_not_drawn(tmp_path):
    rows = make_paired_rows()
    rows["ci_low"] = np.nan
    assert make(pd.DataFrame(), tmp_path / "f.png", paired_df=rows, arms=ARMS, baseline="a") is None


def test_many_metrics_continue_in_numbered_files(tmp_path):
    rows = make_paired_rows()
    many = pd.concat([rows.assign(metric=f"m{i}") for i in range(5)], ignore_index=True)
    make(
        pd.DataFrame(),
        tmp_path / "f.png",
        paired_df=many,
        arms=ARMS,
        baseline="a",
        max_panels=2,
        formats=("png",),
        dpi=40,
    )
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["f.png", "f_2.png", "f_3.png"]


def test_nothing_to_draw_gives_none(tmp_path):
    assert make(pd.DataFrame(), tmp_path / "f.png", paired_df=None, arms=ARMS, baseline="a") is None
    assert make(pd.DataFrame(), tmp_path / "f.png", paired_df=make_paired_rows(), arms=ARMS) is None
