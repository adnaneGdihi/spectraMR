"""``comparison/subject_lines.py``: each subject's baseline and arm values, joined."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_paired_subjects
from spectramr.infrastructure.reporting.comparison.subject_lines import make, subject_differences

ARMS = ["a", "b", "c"]


def test_the_figure_is_written_for_the_requested_phase(tmp_path):
    out = make(
        pd.DataFrame(),
        tmp_path / "s.png",
        paired=make_paired_subjects(),
        arms=ARMS,
        baseline="a",
        phase="test",
        formats=("png",),
        dpi=40,
    )
    assert out is not None and out.is_file()


def test_a_phase_with_no_pairs_gives_none(tmp_path):
    only_val = {k: v for k, v in make_paired_subjects().items() if k[0] == "val"}
    assert (
        make(
            pd.DataFrame(),
            tmp_path / "s.png",
            paired=only_val,
            arms=ARMS,
            baseline="a",
            phase="test",
        )
        is None
    )
    assert make(pd.DataFrame(), tmp_path / "s.png", paired={}, arms=ARMS, baseline="a") is None


def test_many_metrics_continue_in_numbered_files(tmp_path):
    paired = {}
    for i in range(5):
        for (phase, _metric, variant), pair in make_paired_subjects().items():
            paired[(phase, f"m{i}", variant)] = pair
    make(
        pd.DataFrame(),
        tmp_path / "s.png",
        paired=paired,
        arms=ARMS,
        baseline="a",
        max_panels=2,
        formats=("png",),
        dpi=40,
    )
    assert (tmp_path / "s_2.png").is_file() and (tmp_path / "s_3.png").is_file()


def test_differences_are_arm_minus_baseline_per_subject():
    paired = make_paired_subjects()
    diffs = subject_differences(paired, "val", "psnr", ["b", "c"])
    table = paired[("val", "psnr", "b")].table
    assert diffs["b"].tolist() == (table["a"] - table["b"]).tolist()
    assert list(diffs.columns) == ["b", "c"] and len(diffs) == 8


def test_a_variant_with_no_pairs_is_left_out_and_unpaired_subjects_are_nan():
    paired = make_paired_subjects()
    paired[("val", "psnr", "c")] = paired[("val", "psnr", "c")].__class__(
        table=paired[("val", "psnr", "c")].table.iloc[:5],
        n_slices=5,
        nonfinite_dropped={"a": 0, "b": 0},
        unpaired_slices={"a": 0, "b": 0},
        scale="metric",
    )
    diffs = subject_differences(paired, "val", "psnr", ["b", "c", "missing"])
    assert list(diffs.columns) == ["b", "c"]
    assert diffs["c"].isna().sum() == 3  # subjects 5..7 were not paired under arm c
