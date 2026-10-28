"""``comparison/metric_ladder.py``: one metric per arm against named reference values."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_paired_subjects
from spectramr.infrastructure.reporting.comparison.contrasts import Ladder
from spectramr.infrastructure.reporting.comparison.metric_ladder import (
    arm_values,
    make,
    on_metric_scale,
)

ARMS = ["a", "b", "c"]


def test_a_reciprocal_metric_is_averaged_on_its_reciprocal_and_shown_back():
    # N_eq of 2 on every subject is 1/N_eq = 0.5, which is N_eq = 2 back on its own scale
    mean, lo, hi = on_metric_scale(np.array([0.5, 0.5, 0.5]), "nex_equivalent")
    assert (mean, lo, hi) == (2.0, 2.0, 2.0)
    # the mean of 1/N_eq, not of N_eq: subjects at N_eq 1 and 4 average to 1/0.625 = 1.6, not 2.5
    mean, *_ = on_metric_scale(np.array([1.0, 0.25]), "nex_equivalent")
    assert mean == pytest.approx(1.6)


def test_a_subject_with_zero_reciprocal_is_an_infinite_n_eq_not_a_crash():
    mean, _lo, hi = on_metric_scale(np.array([0.0, 0.0]), "nex_equivalent")
    assert mean == np.inf and hi == np.inf


def test_an_ordinary_metric_passes_through():
    mean, lo, hi = on_metric_scale(np.array([1.0, 2.0, 3.0]), "psnr")
    assert mean == 2.0 and lo <= mean <= hi


def test_arm_values_take_the_baseline_from_the_paired_tables():
    values = arm_values(make_paired_subjects(), "val", "psnr", "a")
    assert set(values) == {"a", "b", "c"} and all(len(v) == 8 for v in values.values())


def test_the_figure_is_written_and_unknown_metrics_give_none(tmp_path):
    ladder = Ladder(metric="psnr", references={"identity": 30.0})
    out = make(
        pd.DataFrame(),
        tmp_path / "l.png",
        paired=make_paired_subjects(),
        arms=ARMS,
        baseline="a",
        ladder=ladder,
        formats=("png",),
        dpi=40,
    )
    assert out is not None and out.is_file()
    nothing = Ladder(metric="not_logged")
    assert (
        make(
            pd.DataFrame(),
            tmp_path / "n.png",
            paired=make_paired_subjects(),
            arms=ARMS,
            baseline="a",
            ladder=nothing,
        )
        is None
    )
    assert (
        make(
            pd.DataFrame(),
            tmp_path / "n.png",
            paired=make_paired_subjects(),
            arms=ARMS,
            baseline="a",
            ladder=None,
        )
        is None
    )
