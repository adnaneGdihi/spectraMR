"""``comparison/phase_tables.py``: one value per arm, metric and phase, from the right rows."""

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.comparison import load_arms
from spectramr.infrastructure.reporting.comparison.phase_tables import (
    inference_unavailable,
    phase_table,
    wide,
)

from ._runs import add_inference_results, add_training_metrics, make_run


@pytest.fixture
def runs(tmp_path):
    out = [make_run(tmp_path, "base"), make_run(tmp_path, "better", shift=1.5)]
    for run, level in zip(out, (20.0, 21.0), strict=True):
        add_training_metrics(run, level=level)
    for run, offset in zip(out, (0.0, 0.5), strict=True):
        add_inference_results(run, offset=offset)
    return out


def _load(runs):
    return load_arms(runs, labels=["base", "better"], with_images=False)


def _value(table, arm, phase, metric, column="mean"):
    row = table[(table["arm"] == arm) & (table["phase"] == phase) & (table["metric"] == metric)]
    return row[column].iloc[0]


def test_each_phase_reads_its_own_rows(runs):
    wanted = {"val": ["psnr"], "train": ["psnr"], "test": ["psnr"]}
    table = phase_table(_load(runs), wanted, train_window=10)
    assert _value(table, "better", "val", "psnr") == pytest.approx(25 + 1.5 + 2.0)  # pass 200
    # last 10 training rows at or before the pass (steps 191..200)
    assert _value(table, "better", "train", "psnr") == pytest.approx(21.0 + 0.01 * 195.5)
    assert _value(table, "better", "test", "psnr") == pytest.approx(30 + 3.5 + 0.5)
    assert _value(table, "better", "test", "psnr", "n") == 8
    assert _value(table, "better", "val", "psnr", "step") == 200


def test_training_rows_after_the_pass_are_not_averaged(runs):
    table = phase_table(_load(runs), {"train": ["psnr"]}, train_window=300)
    # 200 rows exist at or before the pass; steps 201..300 must not enter the mean
    assert _value(table, "base", "train", "psnr", "n") == 200


def test_a_restarted_training_log_keeps_the_last_row_of_each_step(tmp_path):
    run = make_run(tmp_path, "base")
    make_run(tmp_path, "better", shift=1.0)
    add_training_metrics(run, level=20.0, repeats_first_rows=True)
    add_training_metrics(tmp_path / "better", level=21.0)
    arms = load_arms([run, tmp_path / "better"], labels=["base", "better"], with_images=False)
    table = phase_table(arms, {"train": ["psnr"]}, train_window=200)
    assert _value(table, "base", "train", "psnr") == pytest.approx(20.0 + 0.01 * 100.5)
    assert _value(table, "base", "train", "psnr", "n") == 200


def test_direction_comes_from_the_metric_ssot(runs):
    table = phase_table(_load(runs), {"test": ["psnr", "nmse"]})
    assert _value(table, "base", "test", "psnr", "direction") == "higher"
    assert _value(table, "base", "test", "nmse", "direction") == "lower"


def test_wide_has_one_row_per_arm(runs):
    table = phase_table(_load(runs), {"test": ["psnr", "nmse"]})
    frame = wide(table, "test")
    assert list(frame["arm"]) == ["base", "better"] and {"psnr", "nmse"} <= set(frame.columns)


def test_a_bad_window_or_phase_raises(runs):
    with pytest.raises(ValueError, match="at least 1"):
        phase_table(_load(runs), {"train": ["psnr"]}, train_window=0)
    with pytest.raises(ValueError, match="unknown phase"):
        phase_table(_load(runs), {"inference": ["psnr"]})


def test_the_recorded_reason_for_a_missing_inference_metric_is_returned(runs):
    (reason,) = inference_unavailable(runs[0], "base")
    assert reason["metric"] == "ssim" and "no_reference_available_on_this_path" in reason["reason"]
    pd.DataFrame([{"metric": "psnr", "status": "computed", "reason": ""}]).to_csv(
        runs[0] / "inference_metrics.csv", index=False
    )
    assert inference_unavailable(runs[0], "base") == []
    assert inference_unavailable(runs[0] / "absent", "base") == []


def test_validation_row_carries_slice_spread_from_the_read_pass(runs):
    from ._runs import add_slice_rows

    for seed, run in enumerate(runs):
        add_slice_rows(run, offset=0.0, seed=seed)
    table = phase_table(_load(runs), {"val": ["psnr"]})
    assert _value(table, "base", "val", "psnr", "n") > 1
    assert _value(table, "base", "val", "psnr", "std") > 0
    # the pass value is still the mean the arm was selected on
    assert _value(table, "better", "val", "psnr") == pytest.approx(25 + 1.5 + 2.0)


def test_validation_row_without_slice_rows_has_no_spread(runs):
    table = phase_table(_load(runs), {"val": ["psnr"]})
    assert _value(table, "base", "val", "psnr", "n") == 1
    assert pd.isna(_value(table, "base", "val", "psnr", "std"))
