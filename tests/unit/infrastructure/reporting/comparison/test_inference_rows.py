"""``comparison/inference_rows.py``: arms paired on the case ids of their inference results."""

import pytest

from spectramr.infrastructure.reporting.comparison import load_arms
from spectramr.infrastructure.reporting.comparison.inference_rows import pair_on_cases

from ._runs import add_inference_results, make_run


def _arms(tmp_path, cases=(8, 8)):
    runs = [make_run(tmp_path, "base"), make_run(tmp_path, "better")]
    for run, n, offset in zip(runs, cases, (0.0, 0.5), strict=True):
        add_inference_results(run, offset=offset, cases=n)
    return load_arms(runs, labels=["base", "better"], with_images=False)


def test_cases_are_joined_on_id_and_the_difference_is_exact(tmp_path):
    paired = pair_on_cases(_arms(tmp_path), "better", "base", "psnr")
    assert paired.n_subjects == 8 and paired.scale == "metric"
    assert (paired.table["a"] - paired.table["b"]).tolist() == pytest.approx([0.5] * 8)


def test_a_case_in_one_arm_only_is_counted_unpaired(tmp_path):
    paired = pair_on_cases(_arms(tmp_path, cases=(8, 5)), "better", "base", "psnr")
    assert paired.n_subjects == 5
    assert paired.unpaired_slices == {"a": 0, "b": 3}


def test_an_arm_without_the_metric_gives_none(tmp_path):
    assert pair_on_cases(_arms(tmp_path), "better", "base", "ssim") is None
