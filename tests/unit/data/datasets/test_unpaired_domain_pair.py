"""Unit tests for UnpairedDomainDataset (independent domain-A / domain-B draws).

IO is mocked: every file loads as a constant tensor whose value identifies the
file, so each sample's provenance can be read back from its tensor.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.config.schemas.data import DataPairingConfigSchema
from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
from spectramr.data.datasets.contrast_aware import ContrastAwarePairedDataset, ContrastConfig
from spectramr.data.datasets.unpaired_domain_pair import UnpairedDomainDataset
from tests.utils.data_config_stub import DataConfigStub

_IO_FACTORY_PATCH = "spectramr.data.io_strategies.IOStrategyFactory.get"

# Subject i owns ULF file /a/ulf{i} (domain A) and HF file /b/hf{i} (domain B).
# Subject 5 is unpaired: ULF only.
_RECORDS = [
    {
        "subject_id": f"sub-000{i}",
        "primary_path": f"/a/ulf{i}.nii.gz",
        "target_path": f"/b/hf{i}.nii.gz" if i <= 4 else None,
        "file_id": f"sub-000{i}_T1w",
    }
    for i in range(1, 6)
]
_VALUE = {f"/a/ulf{i}.nii.gz": float(i) for i in range(1, 6)}
_VALUE.update({f"/b/hf{i}.nii.gz": float(i) for i in range(1, 5)})


def _fake_load(path: str) -> torch.Tensor:
    return torch.full((1, 4, 4, 2), _VALUE[str(path)])


def _io():
    io = MagicMock()
    io.load.side_effect = _fake_load
    return io


def _contrast(name: str) -> ContrastConfig:
    return ContrastConfig(name=name, normalization="none")


@pytest.fixture
def ds_factory():
    def make(index=_RECORDS, **kw):
        with patch(_IO_FACTORY_PATCH, return_value=_io()):
            return UnpairedDomainDataset(
                index=index,
                input_contrast=_contrast("ULF_64mT"),
                target_contrast=_contrast("HF_3T"),
                **kw,
            )

    return make


def _values(ds, n: int = 200):
    out = []
    with patch(_IO_FACTORY_PATCH, return_value=_io()):
        ds.builder.io = _io()
        for k in range(n):
            s = ds[k % len(ds)]
            out.append((float(s["input"].data.flatten()[0]), float(s["target"].data.flatten()[0])))
    return out


class TestIndependentDraws:
    def test_pools_hold_each_domain_and_no_pair_is_forced(self, ds_factory):
        ds = ds_factory()
        assert len(ds.a_pool) == 5  # every ULF record, paired and unpaired
        assert len(ds.b_pool) == 4  # only records that carry an HF volume
        draws = _values(ds)
        assert {a for a, _ in draws} <= {1.0, 2.0, 3.0, 4.0, 5.0}
        assert {b for _, b in draws} <= {1.0, 2.0, 3.0, 4.0}
        # Independent: some sample pairs a ULF subject with a different HF subject.
        assert any(a != b for a, b in draws)

    def test_disjoint_subjects_keeps_domains_apart(self, ds_factory):
        ds = ds_factory(disjoint_subjects=True)
        # Sorted subjects 1..5: A-half = {1,3,5}, B-half = {2,4}.
        assert {r["subject_id"] for r in ds.a_pool} == {"sub-0001", "sub-0003", "sub-0005"}
        assert {r["subject_id"] for r in ds.b_pool} == {"sub-0002", "sub-0004"}
        for a, b in _values(ds):
            assert a in {1.0, 3.0, 5.0}
            assert b in {2.0, 4.0}

    def test_held_out_subject_in_train_index_raises(self, ds_factory):
        with pytest.raises(ValueError, match="held-out subjects"):
            ds_factory(exclude_subjects={"sub-0002"})

    def test_missing_domain_raises(self, ds_factory):
        unpaired_only = [dict(r, target_path=None) for r in _RECORDS]
        with pytest.raises(ValueError, match="needs both domains"):
            ds_factory(index=unpaired_only)


class TestInstantiatorBranch:
    @staticmethod
    def _cfg(unpaired: bool, disjoint: bool = False):
        cfg = DataConfigStub()
        cfg.pairing = DataPairingConfigSchema(
            unpaired_training=unpaired, disjoint_subjects=disjoint
        )
        cfg.input_contrast = SimpleNamespace(
            model_dump=lambda: {"name": "ULF_64mT", "normalization": "none"}
        )
        cfg.target_contrast = SimpleNamespace(
            model_dump=lambda: {"name": "HF_3T", "normalization": "none"}
        )
        return cfg

    def test_unpaired_training_builds_domain_dataset_with_paired_val(self):
        train_index = [r for r in _RECORDS if r["subject_id"] != "sub-0001"]
        val_index = [_RECORDS[0]]  # sub-0001 is the held-out subject
        with patch(_IO_FACTORY_PATCH, return_value=_io()):
            train_ds, val_ds = DatasetInstantiator._create_contrast_aware(
                self._cfg(unpaired=True, disjoint=True), train_index, val_index, None, None
            )
            assert isinstance(train_ds, UnpairedDomainDataset)
            assert isinstance(val_ds, ContrastAwarePairedDataset)
            assert all(r["subject_id"] != "sub-0001" for r in train_ds.a_pool + train_ds.b_pool)

    def test_paired_default_keeps_paired_dataset(self):
        train_index = [r for r in _RECORDS if r["subject_id"] != "sub-0001"]
        with patch(_IO_FACTORY_PATCH, return_value=_io()):
            train_ds, _ = DatasetInstantiator._create_contrast_aware(
                self._cfg(unpaired=False), train_index, [_RECORDS[0]], None, None
            )
        assert isinstance(train_ds, ContrastAwarePairedDataset)

    def test_leak_into_training_raises_through_instantiator(self):
        with (
            patch(_IO_FACTORY_PATCH, return_value=_io()),
            pytest.raises(ValueError, match="held-out subjects"),
        ):
            DatasetInstantiator._create_contrast_aware(
                self._cfg(unpaired=True), _RECORDS, [_RECORDS[1]], None, None
            )
