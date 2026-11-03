"""Tests for :mod:`spectramr.infrastructure.reporting.plotters.dynamics.gain_over_input`."""

from __future__ import annotations

import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_dynamics import make_dynamics_df
from spectramr.infrastructure.reporting.plotters.dynamics import gain_over_input


def test_draws_from_step_rows(tmp_path):
    out = gain_over_input.make(make_dynamics_df(), tmp_path / "f.png", formats=("png",), dpi=40)
    assert out is not None and out.exists()


def test_soft_skips_without_its_series(tmp_path):
    empty = pd.DataFrame({"split": ["train"], "metric": ["other"], "value": [1.0], "method": ["a"]})
    assert gain_over_input.make(empty, tmp_path / "f.png", formats=("png",)) is None


def test_gain_is_output_minus_input_per_step():
    df = make_dynamics_df(n_arms=1)
    g = gain_over_input.gains(df, "psnr")
    out = df[(df.metric == "val_psnr")].groupby("step")["value"].mean()
    inp = df[(df.metric == "val_zf_psnr")].groupby("step")["value"].mean()
    assert g.set_index("step")["value"].round(9).equals((out - inp).round(9).rename("value"))
