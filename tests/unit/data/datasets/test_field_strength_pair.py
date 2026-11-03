"""Per-record B0 stamps on the unpaired NIfTI dataset and the stamp helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torchio as tio

from spectramr.data.datasets.contrast_aware import ContrastConfig
from spectramr.data.datasets.field_strength_pair import stamp_field_strength_pair
from spectramr.data.datasets.unpaired_domain_pair import UnpairedDomainDataset

_IO_FACTORY_PATCH = "spectramr.data.io_strategies.IOStrategyFactory.get"


def _rec(**extra) -> dict:
    rec = {
        "subject_id": "sub-0001",
        "primary_path": "/a/ulf1.nii.gz",
        "target_path": "/b/hf1.nii.gz",
        "file_id": "sub-0001_T1w",
        "input_field": 0.064,
        "target_field": 3.0,
    }
    rec.update(extra)
    return rec


def _blank() -> tio.Subject:
    return tio.Subject(image=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)))


def _io() -> MagicMock:
    io = MagicMock()
    io.load.side_effect = lambda path: torch.ones(1, 4, 4, 2)
    return io


def _dataset(index: list[dict], expose: bool) -> UnpairedDomainDataset:
    contrast = ContrastConfig(name="ULF_64mT", normalization="none")
    with patch(_IO_FACTORY_PATCH, return_value=_io()):
        ds = UnpairedDomainDataset(
            index=index,
            input_contrast=contrast,
            target_contrast=ContrastConfig(name="HF_3T", normalization="none"),
            expose_field_strength_pair=expose,
        )
    ds.builder.io = _io()
    return ds


def test_stamp_helper_writes_both_domains() -> None:
    subject = _blank()
    stamp_field_strength_pair(subject, _rec(), _rec())
    assert subject["field_strength_source_t"].item() == pytest.approx(0.064)
    assert subject["field_strength_t"].item() == pytest.approx(3.0)


def test_stamp_helper_swap_exchanges_domains() -> None:
    subject = _blank()
    stamp_field_strength_pair(subject, _rec(), _rec(), swap=True)
    assert subject["field_strength_source_t"].item() == pytest.approx(3.0)
    assert subject["field_strength_t"].item() == pytest.approx(0.064)


def test_stamp_helper_raises_when_target_field_absent() -> None:
    rec = _rec()
    del rec["target_field"]
    with pytest.raises(ValueError, match="target_field"):
        stamp_field_strength_pair(_blank(), _rec(), rec)


def test_unpaired_dataset_stamps_when_exposed() -> None:
    ds = _dataset([_rec()], expose=True)
    sample = ds[0]
    assert sample["field_strength_source_t"].item() == pytest.approx(0.064)
    assert sample["field_strength_t"].item() == pytest.approx(3.0)


def test_unpaired_dataset_without_flag_emits_no_stamp() -> None:
    sample = _dataset([_rec()], expose=False)[0]
    assert "field_strength_t" not in sample


def test_unpaired_dataset_raises_on_missing_field_when_exposed() -> None:
    rec = _rec()
    del rec["input_field"]
    ds = _dataset([rec], expose=True)
    with pytest.raises(ValueError, match="input_field"):
        _ = ds[0]


def test_manifest_metadata_labels_stamp_tesla() -> None:
    """The ulf_paired manifest states fields as ``metadata`` labels; they stamp as Tesla."""
    rec = _rec(metadata={"field_strength_input": "64mT", "field_strength_target": "3T"})
    del rec["input_field"], rec["target_field"]
    subject = _blank()
    stamp_field_strength_pair(subject, rec, rec)
    assert subject["field_strength_source_t"].item() == pytest.approx(0.064)
    assert subject["field_strength_t"].item() == pytest.approx(3.0)


@pytest.mark.parametrize("label", ["64 mT", "3.0T", "0.3T"])
def test_field_label_parses(label: str) -> None:
    from spectramr.data.datasets.field_strength_pair import field_label_to_tesla

    assert field_label_to_tesla(label) == pytest.approx({"64 mT": 0.064, "3.0T": 3.0, "0.3T": 0.3}[label])


@pytest.mark.parametrize("label", ["ULF", "64", "3 Tesla"])
def test_unparseable_field_label_raises(label: str) -> None:
    rec = _rec(metadata={"field_strength_input": label, "field_strength_target": "3T"})
    del rec["input_field"]
    with pytest.raises(ValueError, match="field label"):
        stamp_field_strength_pair(_blank(), rec, rec)


def test_paired_dataset_stamps_a_v6_manifest_record() -> None:
    """exp_52's route: ContrastAwarePairedDataset on a record shaped as
    preprocess_ulf_paired.py writes it (fields as metadata labels only)."""
    from spectramr.data.datasets.contrast_aware import ContrastAwarePairedDataset

    rec = _rec(metadata={"field_strength_input": "64mT", "field_strength_target": "3T"})
    del rec["input_field"], rec["target_field"]
    t1w = ContrastConfig(name="T1w", normalization="none", keywords=["t1w"])
    with patch(_IO_FACTORY_PATCH, return_value=_io()):
        ds = ContrastAwarePairedDataset(
            index=[rec], input_contrast=t1w, target_contrast=t1w, expose_field_strength_pair=True
        )
    ds.builder.io = _io()
    sample = ds[0]
    assert sample["field_strength_source_t"].item() == pytest.approx(0.064)
    assert sample["field_strength_t"].item() == pytest.approx(3.0)
