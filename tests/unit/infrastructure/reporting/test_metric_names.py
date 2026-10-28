"""Parsing logged metric names into base / acceleration / aggregate."""

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.metric_names import annotate_frame, parse_metric_name


@pytest.mark.parametrize(
    ("raw", "base", "accel", "agg", "headline"),
    [
        ("val_psnr", "psnr", None, None, True),
        ("val_psnr_8x", "psnr", 8.0, None, False),
        ("val_psnr_2.5x", "psnr", 2.5, None, False),
        ("val_psnr_mean", "psnr", None, "_mean", True),
        ("val_psnr_accel_gap", "psnr", None, "_accel_gap", False),
        ("train_ssim", "ssim", None, None, True),
        ("val_zf_psnr", "zf_psnr", None, None, True),
        ("g_total_loss_32x", "g_total_loss", 32.0, None, False),
        ("loss", "loss", None, None, True),
        ("mean", "mean", None, None, True),
    ],
)
def test_parse(raw, base, accel, agg, headline):
    parsed = parse_metric_name(raw)
    assert (parsed.base, parsed.acceleration, parsed.aggregate) == (base, accel, agg)
    assert parsed.is_headline is headline


def test_a_name_that_merely_contains_x_is_not_an_acceleration():
    assert parse_metric_name("val_max").acceleration is None
    assert parse_metric_name("val_pred_abs_max").base == "pred_abs_max"


def test_annotate_frame_adds_the_three_columns():
    df = pd.DataFrame({"metric": ["val_psnr_8x", "val_ssim"], "value": [1.0, 2.0]})
    out = annotate_frame(df)
    assert list(out["base_metric"]) == ["psnr", "ssim"]
    assert out["acceleration"].iloc[0] == 8.0 and pd.isna(out["acceleration"].iloc[1])
    assert list(out["headline"]) == [False, True]
    assert "base_metric" not in df.columns  # input untouched


def test_annotate_frame_passes_through_frames_without_metrics():
    df = pd.DataFrame({"x": [1]})
    assert annotate_frame(df) is df
