"""UniversalMRIDataset sample metadata."""

from __future__ import annotations

from types import SimpleNamespace

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
