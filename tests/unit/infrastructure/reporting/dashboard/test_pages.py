"""Tests for :mod:`spectramr.infrastructure.reporting.dashboard.pages`."""

from __future__ import annotations

import pandas as pd

from spectramr.infrastructure.reporting.dashboard.pages import (
    comparison_page,
    figure_ids,
    headline_metrics,
)


def _comparison(folder):
    (folder / "data").mkdir(parents=True)
    (folder / "figures").mkdir()
    from PIL import Image

    for fid in ("cmp_mosaic", "fig_1_2_learning_curves", "zz_custom"):
        Image.new("RGB", (64, 32)).save(folder / "figures" / f"{fid}.png")
    rows = [
        ("a", "psnr", 30.0, "higher"),
        ("b", "psnr", 31.0, "higher"),
        ("a", "ssim", 0.9, "higher"),
        ("b", "ssim", 0.8, "higher"),
        ("a", "band_x", 1.0, "lower"),
        ("b", "band_x", 2.0, "lower"),
    ]
    pd.DataFrame(
        [
            {"arm": a, "phase": "val", "metric": m, "mean": v, "std": 0.1, "n": 4, "direction": d}
            for a, m, v, d in rows
        ]
    ).to_csv(folder / "data" / "metrics_long.csv", index=False)
    return folder


def test_headline_metrics_prefer_the_named_set():
    long_df = pd.DataFrame(
        [
            {"arm": "a", "phase": "val", "metric": m, "mean": 1.0, "direction": "higher"}
            for m in ("band_x", "ssim", "psnr")
        ]
    )
    assert headline_metrics(long_df, "val") == ["psnr", "ssim"]


def test_figures_are_grouped_into_families(tmp_path):
    folder = _comparison(tmp_path / "comparison")
    assert set(figure_ids(folder)) == {"cmp_mosaic", "fig_1_2_learning_curves", "zz_custom"}
    doc = comparison_page(folder, baseline="a").read_text()
    for family in ("Images", "Training", "Other figures", "Leaderboard", "Results table"):
        assert family in doc
    assert "<tr class='base'><td>a</td>" in doc
