"""Per-slice test rows reduce to stats overall, per contrast and per acquisition situation."""

from __future__ import annotations

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.test_split.summary import (
    SUMMARY_CSV,
    long_metrics,
    write_summary,
)

pytestmark = pytest.mark.unit


def _rows(test_dir):
    rows = pd.DataFrame(
        {
            "batch_index": range(4),
            "subject_id": ["s1", "s1", "s2", "s2"],
            "contrast": ["T1", "T2", "T1", "T2"],
            "slice_index": [0, 0, 0, 0],
            "val_psnr": [20.0, 22.0, 24.0, 26.0],
            "val_psnr_8x": [18.0, 19.0, 20.0, 21.0],
        }
    )
    rows.to_csv(test_dir / "validation_batches.csv", index=False)
    (test_dir / "slices").mkdir()
    pd.DataFrame(
        {
            "batch_index": range(4),
            "image": [f"slices/{i}.png" for i in range(4)],
            "realised_acceleration": [4.0] * 4,
        }
    ).to_csv(test_dir / "slices" / "slice_index.csv", index=False)


def test_a_rung_suffix_becomes_the_acceleration_situation() -> None:
    long = long_metrics(pd.DataFrame({"batch_index": [0], "val_psnr_8x": [1.0], "val_ssim": [0.5]}))
    assert set(zip(long["metric"], long["acceleration"].fillna(0), strict=True)) == {
        ("psnr", 8.0),
        ("ssim", 0.0),
    }


def test_summary_covers_overall_contrast_and_acceleration(tmp_path) -> None:
    _rows(tmp_path)
    table, _, summary = write_summary(tmp_path)
    assert table["image"].tolist() == [f"slices/{i}.png" for i in range(4)]
    assert set(summary["grouping"]) == {
        "overall",
        "contrast",
        "acceleration",
        "contrast+acceleration",
    }
    t1 = summary[
        (summary.grouping == "contrast")
        & (summary.metric == "psnr")
        & (summary.group.str.contains("T1"))
    ]
    # the T1 group pools the full-rate and 8x psnr columns of the two T1 slices
    assert t1["n"].iloc[0] == 4
    r8 = summary[(summary.grouping == "acceleration") & (summary.group == "acceleration=8.0")]
    assert r8["mean"].iloc[0] == pytest.approx(19.5)
    assert (tmp_path / SUMMARY_CSV).is_file()


def test_no_rows_writes_nothing(tmp_path) -> None:
    table, _, summary = write_summary(tmp_path)
    assert table.empty and summary.empty
