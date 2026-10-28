"""``comparison/metric_selection.py``: every logged metric, per phase, or a checked list."""

import pytest

from spectramr.infrastructure.reporting.comparison import load_arms
from spectramr.infrastructure.reporting.comparison.metric_selection import (
    check_phases,
    discover_metrics,
    select_metrics,
    union,
)

from ._runs import add_inference_results, add_training_metrics, make_run


@pytest.fixture
def arms(tmp_path):
    runs = [make_run(tmp_path, "base"), make_run(tmp_path, "better", shift=1.0)]
    for run, level in zip(runs, (20.0, 21.0), strict=True):
        add_training_metrics(run, level=level)
        add_inference_results(run, offset=0.0)
    return load_arms(runs, labels=["base", "better"], with_images=False)


def test_every_phase_reports_what_it_logged_and_no_reference_columns(arms):
    found = discover_metrics(arms)
    assert found["val"] == ["psnr", "ssim"]  # val_zf_psnr is a reference line, not a result
    assert {"psnr", "nmse", "loss"} <= set(found["train"])
    assert found["test"] == ["psnr", "nmse"]


def test_all_keeps_each_phases_own_metrics(arms):
    selected = select_metrics("all", discover_metrics(arms))
    assert selected["test"] == ["psnr", "nmse"] and "ssim" in selected["val"]
    assert union(selected)[:2] == ["psnr", "ssim"]  # the leading metrics stay first


def test_a_list_is_checked_and_filtered_per_phase(arms):
    selected = select_metrics(["ssim", "nmse"], discover_metrics(arms))
    assert selected == {"val": ["ssim"], "train": ["nmse"], "test": ["nmse"]}


def test_a_name_no_phase_logged_raises(arms):
    with pytest.raises(ValueError, match="no phase logged"):
        select_metrics(["not_a_metric"], discover_metrics(arms))
    with pytest.raises(ValueError, match="'all' or a list"):
        select_metrics("everything", discover_metrics(arms))


def test_no_spec_means_the_callers_default(arms):
    assert select_metrics(None, discover_metrics(arms)) is None


def test_unknown_phases_raise_and_duplicates_collapse():
    assert check_phases(["val", "val", "test"]) == ("val", "test")
    with pytest.raises(ValueError, match="unknown phase"):
        check_phases(["validation"])
