"""Tests for :mod:`spectramr.infrastructure.reporting.plotters.differences.slice_profile`."""

from __future__ import annotations

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_slice_df
from spectramr.infrastructure.reporting.plotters.differences import slice_profile


def test_draws_from_identity_keyed_slices(tmp_path):
    out = slice_profile.make(
        None, tmp_path / "f.png", slice_df=make_slice_df(), formats=("png",), dpi=40
    )
    assert out is not None and out.exists()


def test_soft_skips_without_slices(tmp_path):
    assert slice_profile.make(None, tmp_path / "f.png", slice_df=None, formats=("png",)) is None


def test_positional_rows_are_refused(tmp_path):
    rows = make_slice_df().assign(case_id=lambda d: "val#" + d.index.astype(str))
    assert slice_profile.make(None, tmp_path / "f.png", slice_df=rows, formats=("png",)) is None
