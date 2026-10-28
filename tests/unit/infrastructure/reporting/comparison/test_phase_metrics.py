"""``comparison/phase_metrics.py``: one panel per metric across training, validation, inference."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from spectramr.infrastructure.reporting import plotters
from spectramr.infrastructure.reporting.comparison.phase_metrics import make


def _table(metrics, phases=("train", "val", "test")):
    rows = [
        {
            "arm": arm,
            "phase": phase,
            "metric": metric,
            "mean": 1.0 + i + 0.1 * j,
            "std": 0.1 if phase != "val" else float("nan"),
            "n": 5,
            "unit": "x",
            "step": 1,
            "direction": "higher",
        }
        for j, arm in enumerate(("a", "b"))
        for i, metric in enumerate(metrics)
        for phase in phases
    ]
    return pd.DataFrame(rows)


def test_the_figure_is_registered_and_written(tmp_path):
    assert "cmp_phase_metrics" in plotters.list_available()
    out = make(
        pd.DataFrame(),
        tmp_path / "cmp_phase_metrics.png",
        phase_df=_table(["psnr", "cjv"]),
        arms=["a", "b"],
        formats=("png",),
        dpi=50,
    )
    assert out is not None and out.is_file()


def test_more_metrics_than_panels_continue_in_numbered_files(tmp_path):
    out = make(
        pd.DataFrame(),
        tmp_path / "cmp_phase_metrics.png",
        phase_df=_table([f"m{i}" for i in range(7)]),
        arms=["a", "b"],
        max_panels=3,
        formats=("png",),
        dpi=50,
    )
    assert out is not None
    assert sorted(p.name for p in tmp_path.glob("*.png")) == [
        "cmp_phase_metrics.png",
        "cmp_phase_metrics_2.png",
        "cmp_phase_metrics_3.png",
    ]


def test_a_metric_logged_in_one_phase_still_gets_a_panel(tmp_path):
    out = make(
        pd.DataFrame(),
        tmp_path / "f.png",
        phase_df=_table(["only_val"], phases=("val",)),
        arms=["a", "b"],
        formats=("png",),
        dpi=50,
    )
    assert out is not None


def test_nothing_to_draw_returns_none_and_a_bad_panel_count_raises(tmp_path):
    assert make(pd.DataFrame(), tmp_path / "f.png", phase_df=None, arms=["a"]) is None
    with pytest.raises(ValueError, match="at least 1"):
        make(
            pd.DataFrame(), tmp_path / "f.png", phase_df=_table(["psnr"]), arms=["a"], max_panels=0
        )
