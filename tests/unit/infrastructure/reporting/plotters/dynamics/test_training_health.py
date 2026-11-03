"""Tests for :mod:`spectramr.infrastructure.reporting.plotters.dynamics.training_health`."""

from __future__ import annotations

import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_dynamics import make_dynamics_df
from spectramr.infrastructure.reporting.plotters.dynamics import training_health


def test_draws_from_step_rows(tmp_path):
    out = training_health.make(make_dynamics_df(), tmp_path / "f.png", formats=("png",), dpi=40)
    assert out is not None and out.exists()


def test_soft_skips_without_its_series(tmp_path):
    empty = pd.DataFrame({"split": ["train"], "metric": ["other"], "value": [1.0], "method": ["a"]})
    assert training_health.make(empty, tmp_path / "f.png", formats=("png",)) is None


def test_spikes_need_both_the_robust_band_and_a_real_jump():
    flat = pd.DataFrame({"step": range(40), "value": [1.0 + 1e-6 * (i % 3) for i in range(40)]})
    assert training_health.spikes(flat).empty
    jumpy = flat.copy()
    jumpy.loc[20, "value"] = 5.0
    assert training_health.spikes(jumpy)["step"].tolist() == [20]
