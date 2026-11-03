"""Tests for :mod:`spectramr.infrastructure.reporting.plotters.differences.win_rate`."""

from __future__ import annotations

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_slice_df
from spectramr.infrastructure.reporting.plotters.differences import win_rate


def test_draws_from_identity_keyed_slices(tmp_path):
    out = win_rate.make(
        None, tmp_path / "f.png", slice_df=make_slice_df(), formats=("png",), dpi=40
    )
    assert out is not None and out.exists()


def test_soft_skips_without_slices(tmp_path):
    assert win_rate.make(None, tmp_path / "f.png", slice_df=None, formats=("png",)) is None


def test_win_share_is_complementary_and_holm_flags_a_clear_order():
    import pandas as pd

    subj = pd.DataFrame({"a": range(10, 20), "b": range(0, 10), "c": [5.0] * 10}, dtype=float)
    win, sig = win_rate.win_matrix(subj, higher_is_better=True)
    assert win.loc["a", "b"] == 1.0 and win.loc["b", "a"] == 0.0
    assert sig.loc["a", "b"] and sig.loc["b", "a"]
