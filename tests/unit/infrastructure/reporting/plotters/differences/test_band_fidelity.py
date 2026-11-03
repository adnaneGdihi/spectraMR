"""Tests for :mod:`spectramr.infrastructure.reporting.plotters.differences.band_fidelity`."""

from __future__ import annotations

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_slice_df
from spectramr.infrastructure.reporting.plotters.differences import band_fidelity


def test_draws_from_identity_keyed_slices(tmp_path):
    out = band_fidelity.make(
        None, tmp_path / "f.png", slice_df=make_slice_df(), formats=("png",), dpi=40
    )
    assert out is not None and out.exists()


def test_soft_skips_without_slices(tmp_path):
    assert band_fidelity.make(None, tmp_path / "f.png", slice_df=None, formats=("png",)) is None
