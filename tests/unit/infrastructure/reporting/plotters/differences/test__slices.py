"""Tests for :mod:`spectramr.infrastructure.reporting.plotters.differences._slices`."""

from __future__ import annotations

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_slice_df
from spectramr.infrastructure.reporting.plotters.differences import _slices


def test_primary_prefers_psnr_with_its_direction():
    assert _slices.primary(_slices.table(make_slice_df())) == ("psnr", True)


def test_identity_needs_subject_contrast_slice():
    tab = _slices.table(make_slice_df())
    assert _slices.has_identity(tab)
    positional = make_slice_df().assign(case_id=lambda d: "val_step10#" + d.index.astype(str))
    assert not _slices.has_identity(_slices.table(positional))
    assert _slices.slice_of("sub1|T1|7|0") == 7 and _slices.subject_of("sub1|T1|7|0") == "sub1"


def test_subject_means_keep_only_subjects_every_arm_has():
    df = make_slice_df(n_arms=2, n_subjects=3)
    df = df[~((df["method"] == "arm1") & df["case_id"].str.startswith("sub2|"))]
    assert list(_slices.subject_means(_slices.table(df), "psnr").index) == ["sub0", "sub1"]
