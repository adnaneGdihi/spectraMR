"""Render test for the learning-curves plotter.

Covers the ddof=1 (sample-std) cross-seed band fix — a multi-seed cohort must
render a band without error; the std-convention correctness itself is a
one-line ``ddof=1`` and is exercised here by feeding >1 seed per step.
"""

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.plotters import learning_curves


def _curves() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    rows = []
    for seed in range(3):
        for step in range(0, 100, 10):
            rows.append(
                {
                    "method": "ours",
                    "metric": "val_loss",
                    "step": step,
                    "value": 1.0 / (1 + step / 10) + rng.normal(0, 0.02),
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def test_learning_curves_render_with_band(tmp_path):
    out = tmp_path / "learning_curves.pdf"
    res = learning_curves.make(
        _curves(), out, metrics=["val_loss"], log_y=False, band=True,
        formats=("pdf",), dpi=150,
    )
    assert res is not None and res.exists()


def test_learning_curves_single_seed_no_band_crash(tmp_path):
    # ddof=1 std of a single observation is NaN → fillna(0) → no band, no crash.
    df = _curves()
    df = df[df["seed"] == 0]
    out = tmp_path / "lc_single.pdf"
    res = learning_curves.make(
        df, out, metrics=["val_loss"], log_y=False, band=True,
        formats=("pdf",), dpi=150,
    )
    assert res is not None and res.exists()


def test_learning_curves_multi_metric_grid_renders(tmp_path):
    """2026-07-01 clarity pass: pretty titles + panel letters + bottom-row-only
    x-labels must not break a multi-metric grid."""
    frames = []
    for metric in ("loss", "g_total_loss", "val_loss", "d_loss"):
        df = _curves()
        df["metric"] = metric
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    out = tmp_path / "grid.pdf"
    res = learning_curves.make(
        df, out, metrics=["loss", "g_total_loss", "val_loss", "d_loss"],
        log_y=True, band=True, formats=("pdf",), dpi=150,
    )
    assert res is not None and res.exists()


def test_learning_curves_all_zero_metric_no_log_warning(tmp_path):
    """An all-zero loss (degenerate run) must not emit matplotlib's
    'cannot be log-scaled' UserWarning — pitfall #10, warnings are not OK."""
    import warnings

    df = _curves()
    df["value"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = learning_curves.make(df, tmp_path / "zero.pdf",
                                   metrics=["val_loss"], log_y=True,
                                   band=False, formats=("pdf",), dpi=150)
    assert res is not None and res.exists()

def test_each_method_reaches_its_panel_under_its_own_names():
    from spectramr.infrastructure.reporting.metric_names import annotate_frame
    from spectramr.infrastructure.reporting.plotters._curve_panels import default_panels

    rows = []
    for step in (10, 20):
        rows += [
            {"method": "a", "run_id": "a", "split": "val", "step": step, "metric": "val_psnr", "value": 20.0},
            {"method": "b", "run_id": "b", "split": "val", "step": step, "metric": "val_psnr_mean", "value": 21.0},
            {"method": "a", "run_id": "a", "split": "train", "step": step, "metric": "loss", "value": 0.5},
            {"method": "b", "run_id": "b", "split": "train", "step": step, "metric": "g_total_loss", "value": 0.4},
        ]
    panels = {p.title: p for p in default_panels(annotate_frame(pd.DataFrame(rows)))}
    assert set(panels["psnr"].val["method"]) == {"a", "b"}
    assert set(panels["loss"].train["method"]) == {"a", "b"}
