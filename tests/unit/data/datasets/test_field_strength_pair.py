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
