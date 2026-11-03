"""Tests for the interactive figure builders.

Targets ``spectramr.infrastructure.reporting.interactive.figures``: each ``*_div``
returns an inline plotly div on real data and ``None`` on empty input or when
plotly is monkeypatched absent. Also asserts no global numpy RNG is touched.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from spectramr.infrastructure.reporting.interactive import figures as figs

_HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def _per_case(n=16):
    v = np.linspace(30, 34, n)
    return pd.DataFrame({"case_id": [f"c{i}" for i in range(n)], "split": "val",
                         "step": range(n), "psnr": v, "ssim": np.linspace(0.8, 0.95, n),
                         "nmse": np.linspace(0.03, 0.01, n)})


def _predictions(methods=("proposed", "baseline")):
    rows = []
    for m in methods:
        for s in range(8):
            for metric, val in (("psnr", 30 + s * 0.3), ("ssim", 0.8 + s * 0.01)):
                rows.append({"method": m, "metric": metric, "value": val,
                             "subject_id": f"s{s}", "split": "test"})
    return pd.DataFrame(rows)


def _curves():
    rows = []
    for step in range(0, 100, 10):
        for split in ("train", "val"):
            for metric, val in (("loss", 1 - step / 200), ("psnr", 25 + step / 20)):
                rows.append({"run_id": "r", "method": "proposed", "split": split,
                             "step": step, "metric": metric, "value": val})
    return pd.DataFrame(rows)


def _assert_div_or_none(div):
    if _HAS_PLOTLY:
        assert isinstance(div, str) and "plotly-graph-div" in div
        assert 'src="https://cdn' not in div  # inline-once, not per-div CDN
    else:
        assert div is None


def test_group_iqm_div_from_wide_and_long():
    _assert_div_or_none(figs.group_iqm_div(per_call_df=_per_case()))
    _assert_div_or_none(figs.group_iqm_div(predictions_df=_predictions()))


def test_group_iqm_div_none_on_empty():
    assert figs.group_iqm_div(per_call_df=pd.DataFrame()) is None
    assert figs.group_iqm_div() is None


def test_learning_curve_div():
    _assert_div_or_none(figs.learning_curve_div(_curves()))
    assert figs.learning_curve_div(pd.DataFrame()) is None
    # no train/val rows -> None
    only_test = pd.DataFrame({"split": ["test"], "step": [0], "metric": ["psnr"],
                              "value": [30.0]})
    assert figs.learning_curve_div(only_test) is None


def test_metric_distribution_div():
    _assert_div_or_none(figs.metric_distribution_div(predictions_df=_predictions()))
    _assert_div_or_none(figs.metric_distribution_div(per_call_df=_per_case()))
    assert figs.metric_distribution_div() is None


def test_comparison_div_needs_two_methods():
    # single method -> None (soft-skip, honest)
    assert figs.comparison_div(predictions_df=_predictions(methods=("only",))) is None
    _assert_div_or_none(figs.comparison_div(predictions_df=_predictions()))
    assert figs.comparison_div(predictions_df=pd.DataFrame()) is None


def test_no_global_rng_touch():
    before = repr(np.random.get_state())
    figs.group_iqm_div(per_call_df=_per_case())
    figs.metric_distribution_div(predictions_df=_predictions())
    figs.comparison_div(predictions_df=_predictions())
    figs.learning_curve_div(_curves())
    assert repr(np.random.get_state()) == before


def test_all_none_when_plotly_absent(monkeypatch):
    monkeypatch.setattr(figs, "has_plotly", lambda: False)
    assert figs.group_iqm_div(per_call_df=_per_case()) is None
    assert figs.learning_curve_div(_curves()) is None
    assert figs.metric_distribution_div(predictions_df=_predictions()) is None
    assert figs.comparison_div(predictions_df=_predictions()) is None


def test_learning_curve_pairs_prefixed_val_metric():
    # Real data: training logs `loss`, validation logs `val_loss` (prefixed). The
    # div must strip the val_ prefix so both land on one base metric as train
    # (solid) + val (dotted) — otherwise the validation loss never shows paired.
    rows = []
    for step in range(0, 50, 10):
        rows.append({"run_id": "r", "method": "proposed", "split": "train",
                     "step": step, "metric": "loss", "value": 1 - step / 100})
        rows.append({"run_id": "r", "method": "proposed", "split": "val",
                     "step": step, "metric": "val_loss", "value": 1.1 - step / 100})
    div = figs.learning_curve_div(pd.DataFrame(rows))
    if _HAS_PLOTLY:
        assert div is not None
        # hovertemplate carries "<method> <split> <base-metric>" for each trace.
        assert "proposed train loss" in div and "proposed val loss" in div
    else:
        assert div is None


def test_large_case_sets_keep_exact_boxes_but_thin_the_dots():
    pytest.importorskip("plotly")

    from spectramr.infrastructure.reporting.interactive import figures as F

    n = 5000
    vals = [30.0 + (i % 100) / 100 for i in range(n)] + [10.0]  # one far outlier
    df = pd.DataFrame({"case_id": [f"c{i}" for i in range(n + 1)], "psnr": vals})
    div = F.group_iqm_div(per_call_df=df)
    assert "c5000" in div  # the outlier is always drawn
    import re

    assert len(re.findall(r'"c\d+"', div)) == F.MAX_POINTS + 1  # sample + the outlier, not all 5001
    assert f"a random {F.MAX_POINTS}" in div
    st = F._box_stats(pd.Series(vals))
    assert st["median"] == pytest.approx(pd.Series(vals).median())
