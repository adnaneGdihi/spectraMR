"""UniversalMRIDataset sample metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.data.datasets.universal_dataset import UniversalMRIDataset


def _dataset(record: dict) -> UniversalMRIDataset:
    ds = UniversalMRIDataset(index=[record], normalize_kspace=False, transform=None)
    ds.subject_builder = SimpleNamespace(build=lambda rec: {})
    return ds


def test_subject_id_is_stamped_when_the_record_carries_one() -> None:
    """VFU-09: held-out conformal calibration keys residuals on subject_id."""
    sample = _dataset({"primary_path": "/x/sub-07_T1w_ulf.nii.gz", "subject_id": 7})[0]
    assert sample["subject_id"] == "7"


def test_no_subject_id_key_without_one_in_the_record() -> None:
    sample = _dataset({"primary_path": "/x/a.nii.gz"})[0]
    assert "subject_id" not in sample


def test_record_field_strength_is_stamped_when_asked_and_present() -> None:
    record = {"primary_path": "/x/a.h5", "field_strength": 0.3}
    ds = UniversalMRIDataset(index=[record], normalize_kspace=False, record_field_strength=True)
    ds.subject_builder = SimpleNamespace(build=lambda rec: {})
    assert float(ds[0]["field_strength"]) == pytest.approx(0.3)
    assert "field_strength" not in _dataset(record)[0]  # off unless data.expose.field_strength
    ds.index = [{"primary_path": "/x/b.h5"}]  # no field: left to attach_field_strength
    assert "field_strength" not in ds[0]


def test_a_non_positive_record_field_strength_raises() -> None:
    ds = UniversalMRIDataset(
        index=[{"primary_path": "/x/a.h5", "field_strength": 0.0}],
        normalize_kspace=False,
        record_field_strength=True,
    )
    ds.subject_builder = SimpleNamespace(build=lambda rec: {})
    with pytest.raises(ValueError, match="not > 0"):
        ds[0]
