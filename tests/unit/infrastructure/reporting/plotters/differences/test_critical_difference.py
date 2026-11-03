"""Tests for :mod:`spectramr.infrastructure.reporting.plotters.differences.critical_difference`."""

from __future__ import annotations

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_slice_df
from spectramr.infrastructure.reporting.plotters.differences import critical_difference


def test_draws_from_identity_keyed_slices(tmp_path):
    out = critical_difference.make(
        None, tmp_path / "f.png", slice_df=make_slice_df(), formats=("png",), dpi=40
    )
    assert out is not None and out.exists()


def test_soft_skips_without_slices(tmp_path):
    assert (
        critical_difference.make(None, tmp_path / "f.png", slice_df=None, formats=("png",)) is None
    )


def test_needs_three_arms(tmp_path):
    two = make_slice_df(n_arms=2)
    assert (
        critical_difference.make(None, tmp_path / "f.png", slice_df=two, formats=("png",)) is None
    )
