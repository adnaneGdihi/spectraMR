"""Tests for :mod:`spectramr.infrastructure.reporting.plotters.dynamics._series`."""

from __future__ import annotations

import pandas as pd

from spectramr.infrastructure.reporting.plotters.dynamics import _series


def test_series_averages_repeats_and_sorts_by_step():
    df = pd.DataFrame(
        {
            "split": ["val"] * 3,
            "metric": ["val_psnr"] * 3,
            "step": [20, 10, 10],
            "value": [3.0, 1.0, 2.0],
            "method": ["a"] * 3,
        }
    )
    out = _series.series(df, "val", "val_psnr")
    assert out["step"].tolist() == [10, 20] and out["value"].tolist() == [1.5, 3.0]


def test_missing_metric_or_frame_is_empty():
    assert _series.series(None, "val", "x").empty
    assert not _series.has(pd.DataFrame({"split": [], "metric": [], "value": []}), "val", "x")
