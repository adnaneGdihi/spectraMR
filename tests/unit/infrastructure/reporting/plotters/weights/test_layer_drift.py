import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_checkpoint_runs
from spectramr.infrastructure.reporting.plotters.weights import layer_drift
from spectramr.infrastructure.reporting.weights import WeightSample


def test_drift_is_relative_to_the_first_checkpoint_per_group():
    first = WeightSample(np.ones(6), np.array([0, 0, 0, 1, 1, 1]), ("a", "b"))
    later = WeightSample(np.array([1, 1, 1, 2, 2, 2.0]), first.group, first.group_names)
    np.testing.assert_allclose(layer_drift.group_drift(first, later), [0.0, 1.0])


def test_a_zero_norm_group_is_nan_not_a_guess():
    first = WeightSample(np.zeros(2), np.zeros(2, dtype=int), ("a",))
    assert np.isnan(layer_drift.group_drift(first, first)).all()


def test_one_run_gives_a_column_per_later_checkpoint_and_arms_one_each(tmp_path):
    runs = make_checkpoint_runs(2)
    _, solo = layer_drift._columns({"arm0": runs["arm0"]})
    _, both = layer_drift._columns(runs)
    assert list(solo) == ["2000", "4000", "best"]
    assert list(both) == ["arm0", "arm1"]
    out = layer_drift.make(
        pd.DataFrame(), tmp_path / "f.pdf", weight_runs=runs, formats=("pdf",), dpi=72
    )
    assert out is not None and out.exists()


def test_soft_skips_without_runs(tmp_path):
    assert layer_drift.make(pd.DataFrame(), tmp_path / "f.pdf", weight_runs={}) is None
