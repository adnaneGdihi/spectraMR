"""``comparison/contrast_data.py``: the tables a contrast folder keeps under ``data/``."""

import pytest

from spectramr.infrastructure.reporting.comparison import load_arms
from spectramr.infrastructure.reporting.comparison.contrast_data import (
    all_metrics,
    inference_skips,
    write_phase_tables,
)
from spectramr.infrastructure.reporting.comparison.contrasts import ContrastSet

from ._runs import add_inference_results, add_validation_columns, make_run


@pytest.fixture
def arms(tmp_path):
    runs = [make_run(tmp_path, "base"), make_run(tmp_path, "better", shift=1.0)]
    for run in runs:
        add_validation_columns(run, ["cjv"])
        add_inference_results(run, offset=0.0)
    return load_arms(runs, labels=["base", "better"], with_images=False)


def _set(**over):
    base = {"contrasts": [{"name": "c", "arms": ["a", "b"]}]}
    return ContrastSet.model_validate({**base, **over})


def test_all_metrics_has_one_row_per_arm_at_the_pass_read(arms):
    frame = all_metrics(arms).set_index("arm")
    assert list(frame.index) == ["base", "better"] and (frame["step"] == 200).all()
    assert {"val_psnr", "val_ssim", "val_cjv"} <= set(frame.columns)


def test_skipped_inference_metrics_are_listed_once_per_arm(arms):
    listed = inference_skips(arms, ("val", "test"), known=[])
    assert [(u["variant"], u["metric"]) for u in listed] == [("base", "ssim"), ("better", "ssim")]
    known = [{"phase": "test", "metric": "ssim", "variant": "base", "reason": "declared"}]
    assert [u["variant"] for u in inference_skips(arms, ("test",), known)] == ["better"]
    assert inference_skips(arms, ("val",), known=[]) == []  # no inference phase requested


def test_phase_tables_use_every_metric_unless_the_set_names_some(arms, tmp_path):
    used = write_phase_tables(tmp_path, arms, _set(), ("val", "test"), 50)
    assert {"psnr", "cjv"} <= set(used["val"]) and used["test"] == ["psnr", "nmse"]
    assert (tmp_path / "metrics_long.csv").is_file() and (tmp_path / "metrics_test.csv").is_file()
    (tmp_path / "n").mkdir()
    named = write_phase_tables(tmp_path / "n", arms, _set(metrics=["val_cjv"]), ("val", "test"), 50)
    assert named == {"val": ["cjv"], "test": []}
