"""Tests for :mod:`spectramr.infrastructure.reporting.plotters.dynamics.spectral_recovery`."""

from __future__ import annotations

import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_dynamics import make_dynamics_df
from spectramr.infrastructure.reporting.plotters.dynamics import spectral_recovery


def test_draws_from_step_rows(tmp_path):
    out = spectral_recovery.make(make_dynamics_df(), tmp_path / "f.png", formats=("png",), dpi=40)
    assert out is not None and out.exists()


def test_soft_skips_without_its_series(tmp_path):
    empty = pd.DataFrame({"split": ["train"], "metric": ["other"], "value": [1.0], "method": ["a"]})
    assert spectral_recovery.make(empty, tmp_path / "f.png", formats=("png",)) is None
