"""``comparison/contrast_stats.py``: paired subject tests, and Holm across a family."""

import math

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.comparison import load_arms
from spectramr.infrastructure.reporting.comparison.contrast_stats import apply_holm, contrast_stats

from ._runs import add_inference_results, add_slice_rows, make_run

OFFSET = 0.5


@pytest.fixture
def arms(tmp_path):
    base, better = make_run(tmp_path, "base"), make_run(tmp_path, "better")
    add_slice_rows(base, offset=0.0, seed=1)
    add_slice_rows(better, offset=OFFSET, seed=2)
    return load_arms([base, better], labels=["base", "better"], with_images=False)


def test_the_difference_is_recovered_at_the_selected_pass(arms):
    out = contrast_stats("c", arms, ("val_psnr",), "base", "fam")
    (row,) = out.rows
    assert row["mean_diff"] == pytest.approx(OFFSET, abs=0.1)  # iteration 300 would add 0
    assert row["iteration_variant"] == row["iteration_baseline"] == 200
    assert row["ci_low"] < OFFSET < row["ci_high"]
    assert row["n_subjects"] == 6 and row["n_slices"] == 24
    assert row["p_ttest"] < 1e-3


def test_without_a_declared_list_every_shared_metric_is_tested(arms):
    out = contrast_stats("c", arms, None, "base", None)
    assert sorted(r["metric"] for r in out.rows) == [
        "psnr",
        "reference_noise_corrected_mse",
    ]
    mse = next(r for r in out.rows if r["metric"].endswith("corrected_mse"))
    assert mse["mean_diff"] < 0  # the planted offset lowers it: 0.001 * offset, sign carried


def test_a_text_column_is_not_a_metric(tmp_path):
    runs = [make_run(tmp_path, "base"), make_run(tmp_path, "better")]
    for run, offset in zip(runs, (0.0, OFFSET), strict=True):
        rows = add_slice_rows(run, offset=offset, seed=3)
        frame = pd.read_csv(rows)
        frame["note"] = "hello"
        frame.to_csv(rows, index=False)
    arms = load_arms(runs, labels=["base", "better"], with_images=False)
    tested = {r["metric"] for r in contrast_stats("c", arms, None, "base", None).rows}
    assert tested == {"psnr", "reference_noise_corrected_mse"}


def test_a_run_without_slice_rows_is_unavailable_with_a_reason_and_no_p_value(tmp_path):
    base, better = make_run(tmp_path, "base"), make_run(tmp_path, "better")
    add_slice_rows(base, offset=0.0, seed=1)
    arms = load_arms([base, better], labels=["base", "better"], with_images=False)
    out = contrast_stats("c", arms, ("val_psnr",), "base", None)
    assert out.rows == []
    assert "wrote no per-slice rows" in out.unavailable[0]["reason"]


def test_a_metric_missing_from_the_rows_is_unavailable(arms):
    out = contrast_stats("c", arms, ("val_nex_equivalent",), "base", None)
    assert out.rows == [] and "not a column" in out.unavailable[0]["reason"]


def _row(p, family, metric="m", phase="val"):
    return {
        "p_ttest": p,
        "holm_family": family,
        "metric": metric,
        "phase": phase,
        "p_holm": None,
        "holm_m": None,
    }


def test_holm_corrects_within_a_family_and_metric_only():
    rows = [
        _row(0.01, "f"),
        _row(0.04, "f"),
        _row(0.01, "g"),
        _row(0.01, None),
        _row(0.01, "f", "n"),
    ]
    apply_holm(rows)
    # (f, m): sorted 0.01, 0.04 -> 0.02, max(0.02, 0.04) = 0.04
    assert rows[0]["p_holm"] == pytest.approx(0.02) and rows[1]["p_holm"] == pytest.approx(0.04)
    assert rows[0]["holm_m"] == 2
    assert rows[2]["p_holm"] == pytest.approx(0.01) and rows[2]["holm_m"] == 1  # alone in (g, m)
    assert rows[4]["p_holm"] == pytest.approx(0.01)  # alone in (f, n)
    assert rows[3]["p_holm"] is None and rows[3]["holm_m"] is None  # no family: uncorrected


def test_holm_never_mixes_phases():
    rows = [_row(0.01, "f", phase="val"), _row(0.04, "f", phase="test")]
    apply_holm(rows)
    assert [r["holm_m"] for r in rows] == [1, 1]
    assert [r["p_holm"] for r in rows] == pytest.approx([0.01, 0.04])


def test_inference_cases_are_paired_on_case_id(tmp_path):
    runs = [make_run(tmp_path, "base"), make_run(tmp_path, "better")]
    for run, offset in zip(runs, (0.0, OFFSET), strict=True):
        add_inference_results(run, offset=offset)
    arms = load_arms(runs, labels=["base", "better"], with_images=False)
    out = contrast_stats("c", arms, None, "base", None, phases=("test",))
    by_metric = {r["metric"]: r for r in out.rows}
    assert set(by_metric) == {"psnr", "nmse"}
    assert by_metric["psnr"]["mean_diff"] == pytest.approx(OFFSET)
    assert by_metric["psnr"]["unit"] == "cases" and by_metric["psnr"]["n_subjects"] == 8
    assert by_metric["nmse"]["mean_diff"] < 0


def test_a_declared_metric_missing_from_inference_says_why(tmp_path):
    runs = [make_run(tmp_path, "base"), make_run(tmp_path, "better")]
    for run, offset in zip(runs, (0.0, OFFSET), strict=True):
        add_inference_results(run, offset=offset)
    arms = load_arms(runs, labels=["base", "better"], with_images=False)
    out = contrast_stats("c", arms, ("ssim",), "base", None, phases=("test",))
    assert out.rows == []
    assert "no_reference_available_on_this_path" in out.unavailable[0]["reason"]


def test_a_case_scored_by_one_arm_only_does_not_enter_the_pairing(tmp_path):
    runs = [make_run(tmp_path, "base"), make_run(tmp_path, "better")]
    add_inference_results(runs[0], offset=0.0, cases=8)
    add_inference_results(runs[1], offset=OFFSET, cases=5)
    arms = load_arms(runs, labels=["base", "better"], with_images=False)
    (row, *_) = contrast_stats("c", arms, ("psnr",), "base", None, phases=("test",)).rows
    assert row["n_subjects"] == 5


def test_holm_is_monotone_and_capped_at_one():
    rows = [_row(0.4, "f"), _row(0.6, "f"), _row(0.001, "f")]
    apply_holm(rows)
    assert [r["p_holm"] for r in rows] == pytest.approx([0.8, 0.8, 0.003])
    assert all(not math.isnan(r["p_holm"]) and r["p_holm"] <= 1 for r in rows)
