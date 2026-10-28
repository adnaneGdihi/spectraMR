"""Unit tests for the ``cohort_task_ablation_bars`` plotter."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)  # headless: never open a display backend

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.plotters import get, list_available, task_ablation_bars


def _frame() -> pd.DataFrame:
    rows = []
    for family, role, arm in [
        ("b1", "method", "mrixfields_b1_full"),
        ("b1", "ablation", "mrixfields_b1_ablate_x"),
        ("b2", "method", "mrixfields_b2_full"),
        ("b2", "ablation", "mrixfields_b2_ablate_y"),
    ]:
        for metric, value in [("ssim", 0.8), ("psnr", 30.0), ("lpips", 0.2)]:
            rows.append(
                {"family": family, "role": role, "arm": arm, "metric": metric, "value": value}
            )
    return pd.DataFrame(rows)


def test_registered() -> None:
    assert "cohort_task_ablation_bars" in list_available()
    assert get("cohort_task_ablation_bars") is task_ablation_bars.make


def test_make_writes_pdf_and_png(tmp_path) -> None:
    out = tmp_path / "task1_ablation_bars.pdf"
    result = task_ablation_bars.make(_frame(), out, metrics=("ssim", "psnr", "lpips"))
    assert result is not None
    assert result.exists() and result.stat().st_size > 0
    assert (tmp_path / "task1_ablation_bars.png").exists()


def test_baseline_role_renders_with_distinct_colour(tmp_path) -> None:
    # A frame with a role="baseline" family renders (Task 8 overlay support) and
    # the baseline role gets a colour distinct from method and ablation.
    df = _frame()
    extra = pd.DataFrame(
        [
            {"family": "cut", "role": "baseline", "arm": "baseline_cut", "metric": m, "value": v}
            for m, v in [("ssim", 0.6), ("lpips", 0.35)]
        ],
        columns=pd.Index(["family", "role", "arm", "metric", "value"]),
    )
    df = pd.concat([df, extra], ignore_index=True)
    result = task_ablation_bars.make(df, tmp_path / "task1_with_baselines.pdf",
                                     metrics=("ssim", "psnr", "lpips"))
    assert result is not None and result.exists()
    base = task_ablation_bars._role_colour("baseline")
    assert base != task_ablation_bars._role_colour("method")
    assert base != task_ablation_bars._role_colour("ablation")


def test_empty_frame_returns_none(tmp_path) -> None:
    empty = pd.DataFrame(columns=pd.Index(["family", "role", "arm", "metric", "value"]))
    assert task_ablation_bars.make(empty, tmp_path / "x.pdf") is None


def test_missing_required_columns_returns_none(tmp_path) -> None:
    bad = pd.DataFrame({"family": ["b1"], "value": [0.5]})
    assert task_ablation_bars.make(bad, tmp_path / "x.pdf") is None


def test_no_requested_metric_present_returns_none(tmp_path) -> None:
    result = task_ablation_bars.make(_frame(), tmp_path / "x.pdf", metrics=("dice",))
    assert result is None


def test_missing_metric_cell_does_not_raise(tmp_path) -> None:
    # Drop b2's LPIPS row: the panel must still render with a gap, not crash.
    df = _frame()
    df = df[~((df["arm"] == "mrixfields_b2_full") & (df["metric"] == "lpips"))]
    result = task_ablation_bars.make(df, tmp_path / "y.pdf", metrics=("ssim", "psnr", "lpips"))
    assert result is not None and result.exists()


def test_degenerate_arm_hatched_without_error(tmp_path) -> None:
    result = task_ablation_bars.make(
        _frame(), tmp_path / "z.pdf", degenerate={"mrixfields_b1_ablate_x"}
    )
    assert result is not None and result.exists()


def test_negative_value_renders(tmp_path) -> None:
    # Collapsed arms can report negative PSNR — must not raise (zero-line drawn).
    df = _frame()
    df.loc[(df["arm"] == "mrixfields_b2_full") & (df["metric"] == "psnr"), "value"] = -9.0
    result = task_ablation_bars.make(df, tmp_path / "neg.pdf", metrics=("psnr",))
    assert result is not None and result.exists()


@pytest.mark.parametrize("metrics", [("ssim",), ("ssim", "psnr"), ("ssim", "psnr", "lpips")])
def test_variable_panel_count(tmp_path, metrics) -> None:
    result = task_ablation_bars.make(_frame(), tmp_path / "m.pdf", metrics=metrics)
    assert result is not None and result.exists()


def _captured_figure(monkeypatch, tmp_path, **kwargs):
    """Run ``make`` and return the figure it would have saved."""
    seen = {}

    def _save(fig, out_dir, stem, *, formats, dpi):
        seen["fig"] = fig
        return {formats[0]: tmp_path / f"{stem}.{formats[0]}"}

    monkeypatch.setattr(task_ablation_bars, "save_figure", _save)
    task_ablation_bars.make(_frame(), tmp_path / "t.pdf", metrics=("ssim",), **kwargs)
    return seen["fig"]


def test_unconfirmed_arm_is_marked_and_excluded_arm_is_named(monkeypatch, tmp_path) -> None:
    fig = _captured_figure(
        monkeypatch,
        tmp_path,
        run_notes={"mrixfields_b2_full": "status unknown: no run_summary.json"},
        excluded_runs={"mrixfields_b3_crashed": "crashed: RuntimeError"},
    )
    ticks = [t.get_text() for t in fig.axes[-1].get_xticklabels()]
    assert ticks.count("full ‡") == 1, ticks
    texts = " ".join(t.get_text() for t in fig.texts)
    assert "‡ full: status unknown: no run_summary.json." in texts
    assert "Not plotted: crashed (crashed: RuntimeError)." in texts


def test_no_status_notes_leaves_labels_unmarked(monkeypatch, tmp_path) -> None:
    fig = _captured_figure(monkeypatch, tmp_path)
    ticks = [t.get_text() for t in fig.axes[-1].get_xticklabels()]
    assert not any("‡" in t for t in ticks)
    assert not any("Not plotted" in t.get_text() for t in fig.texts)
