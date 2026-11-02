"""Tests for the offline 64 mT synthetic k-space corpus generator."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.datasets.universal_dataset import parse_fastmri_index
from spectramr.data.io_strategies import FastMRIH5Strategy
from spectramr.data.synthesis.ulf_kspace_corpus import (
    ULFCorpusParams,
    build_ulf_corpus,
    rewrite_ismrmrd_header,
    synthesize_ulf_kspace,
    volume_seed,
)

_HEADER = (
    "<ismrmrdHeader><acquisitionSystemInformation>"
    "<systemFieldStrength_T>2.89</systemFieldStrength_T>"
    "<receiverChannels>4</receiverChannels>"
    "</acquisitionSystemInformation></ismrmrdHeader>"
)


def _write_source(path, seed=0):
    g = np.random.default_rng(seed)
    k = (g.standard_normal((2, 3, 16, 16)) + 1j * g.standard_normal((2, 3, 16, 16))).astype(
        np.complex64
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=k)
        f.create_dataset("ismrmrd_header", data=_HEADER)


def test_synthesize_shapes_and_determinism():
    k = torch.randn(2, 3, 16, 16, dtype=torch.complex64)
    p = ULFCorpusParams()
    k1, rss1 = synthesize_ulf_kspace(k, p, seed=7)
    k2, _ = synthesize_ulf_kspace(k, p, seed=7)
    k3, _ = synthesize_ulf_kspace(k, p, seed=8)
    assert k1.shape == (2, 1, 16, 16) and k1.dtype == torch.complex64
    assert rss1.shape == (2, 16, 16) and rss1.dtype == torch.float32
    assert torch.equal(k1, k2)
    assert not torch.equal(k1, k3)


def test_volume_seed_is_stable():
    assert volume_seed("file_a") == volume_seed("file_a") != volume_seed("file_b")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_field_strength_t": 3.5},
        {"resolution_scale": 0.0},
        {"b0_warp_strength": -0.1},
        {"contrast_remapping": True},
    ],
)
def test_params_reject_invalid(kwargs):
    with pytest.raises(ValueError):
        ULFCorpusParams(**kwargs)


def test_off_field_source_is_skipped_and_missing_field_raises(tmp_path):
    _write_source(tmp_path / "v15.h5")
    with h5py.File(tmp_path / "v15.h5", "a") as f:
        del f["ismrmrd_header"]
        f.create_dataset("ismrmrd_header", data=_HEADER.replace("2.89", "1.494"))
    records, skipped = build_ulf_corpus(
        [{"relative_path": "v15.h5"}],
        source_root=tmp_path,
        out_root=tmp_path / "o",
        params=ULFCorpusParams(),
    )
    assert records == [] and skipped == ["v15"]
    with h5py.File(tmp_path / "v15.h5", "a") as f:
        del f["ismrmrd_header"]
        f.create_dataset("ismrmrd_header", data="<ismrmrdHeader/>")
    with pytest.raises(ValueError, match="systemFieldStrength_T"):
        build_ulf_corpus(
            [{"relative_path": "v15.h5"}],
            source_root=tmp_path,
            out_root=tmp_path / "o",
            params=ULFCorpusParams(),
        )


def test_header_rewrite_and_missing_tag():
    out = rewrite_ismrmrd_header(_HEADER, field_strength_t=0.064, receiver_channels=1)
    assert "<systemFieldStrength_T>0.064</systemFieldStrength_T>" in out
    assert "<receiverChannels>1</receiverChannels>" in out
    with pytest.raises(ValueError, match="receiverChannels"):
        rewrite_ismrmrd_header(
            "<systemFieldStrength_T>3</systemFieldStrength_T>",
            field_strength_t=0.064,
            receiver_channels=1,
        )


def test_corpus_is_read_by_the_production_manifest_parser(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src_root = tmp_path / "databases" / "fastmri"
    _write_source(src_root / "vol_a.h5", 0)
    _write_source(src_root / "vol_b.h5", 1)
    out_root = tmp_path / "databases" / "ulf_synthetic_64mT"
    records, skipped = build_ulf_corpus(
        [
            {"relative_path": "vol_a.h5", "file_id": "vol_a", "subject_id": "S1"},
            {"relative_path": "vol_b.h5"},
        ],
        source_root=src_root,
        out_root=out_root,
        params=ULFCorpusParams(),
    )
    assert skipped == []
    assert [r["subject_id"] for r in records] == ["S1", "vol_b"]
    assert all(r["field_strength"] == 0.064 for r in records)
    manifest = tmp_path / "ulf.json"
    manifest.write_text(
        json.dumps({"data_root": "databases/ulf_synthetic_64mT", "records": records})
    )
    index = parse_fastmri_index(str(manifest))
    assert {r["field_strength"] for r in index} == {0.064}
    for rec in index:
        with h5py.File(rec["primary_path"], "r") as f:
            assert f["kspace"].shape == (2, 1, 16, 16)
            prov = json.loads(f.attrs["synthetic_ulf_provenance"])
            assert prov["target_field_strength_t"] == 0.064 and prov["contrast_remapping"] is False
            assert b"0.064" in f["ismrmrd_header"][()]
        loaded = FastMRIH5Strategy().load(rec["primary_path"], {"slice_index": 1})
        assert loaded["kspace"].shape == (1, 16, 16) and torch.is_complex(loaded["kspace"])
        assert loaded["data"].shape == (16, 16)


def test_target_is_cropped_like_the_source_rss(tmp_path):
    _write_source(tmp_path / "v.h5")
    with h5py.File(tmp_path / "v.h5", "a") as f:
        f.create_dataset("reconstruction_rss", data=np.ones((2, 8, 8), np.float32))
    records, _ = build_ulf_corpus(
        [{"relative_path": "v.h5"}],
        source_root=tmp_path,
        out_root=tmp_path / "o",
        params=ULFCorpusParams(),
    )
    assert records[0]["shape"] == [2, 16, 16]
    with h5py.File(tmp_path / "o" / "v.h5", "r") as f:
        assert f["reconstruction_rss"].shape == (2, 8, 8)
        assert f["kspace"].shape == (2, 1, 16, 16)
