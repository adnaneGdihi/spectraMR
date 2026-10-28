"""Unit tests for IndexBuilder.load_paired_bids_manifest (v4 JSON).

Tests cover:
  - FileNotFoundError on missing manifest
  - ValueError on wrong manifest version
  - Split filtering (train / val)
  - Pairing status filter (allow_unpaired=True / False)
  - Contrast filtering
  - HF resolution filtering (highres / lowres / None)
  - Bidirectional swap (hf_to_ulf mode)
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from spectramr.config.schemas.data import DataPairingConfigSchema
from tests.utils.data_config_stub import DataConfigStub
from unittest.mock import patch

import pytest

from spectramr.data.metadata.index_builder import IndexBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_record(
    sub: str = "sub-0001",
    contrast: str = "T1w",
    split_hint: str = "train",
    pairing_status: str = "paired",
    primary: str = "/data/ulf_T1w.nii.gz",
    target: str | None = "/data/hf_T1w.nii.gz",
    hf_resolution: str | None = "highres",
) -> dict:
    return {
        "subject_id": sub,
        "contrast": contrast,
        "pairing_status": pairing_status,
        "split_hint": split_hint,
        "primary_path": primary,
        "target_path": target,
        "hf_resolution": hf_resolution,
        "file_id": f"{sub}_{contrast}",
        "shape": [1, 256, 256, 48],
        "metadata": {},
    }


def _make_manifest_file(tmp_path: Path, records: list[dict], version: str = "4.0") -> Path:
    manifest = {
        "manifest_version": version,
        "records": records,
    }
    p = tmp_path / "test_manifest_v4.json"
    with open(p, "w") as f:
        json.dump(manifest, f)
    return p


def _no_op_resolve(path):
    """PathResolver mock that returns the path unchanged."""
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _split(**kw):
    """The `data.split` sub-block a real config always carries (phase 9b).

    `IndexBuilder.load_paired_bids_manifest` reads `config.split.type` to decide
    whether to apply the split_hint filter, so a stand-in without the block is a
    shape no config produces.
    """
    return SimpleNamespace(
        **{"type": "auto", "holdout_subject": None, "loso_fold": None, **kw}
    )


@patch("spectramr.data.metadata.index_builder.PathResolver.resolve", side_effect=_no_op_resolve)
class TestLoadPairedBidsManifest:
    def test_raises_on_missing_file(self, mock_resolve):
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False,
            bidirectional_mode="ulf_to_hf",
            contrasts=None,
            hf_resolution=None,
        )
        with pytest.raises(FileNotFoundError, match="v4 paired manifest not found"):
            IndexBuilder.load_paired_bids_manifest(
                manifest_path=Path("/nonexistent/manifest.json"),
                split="train",
                config=cfg,
            )

    def test_raises_on_wrong_version(self, mock_resolve, tmp_path):
        p = _make_manifest_file(tmp_path, records=[], version="3.0")
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False, bidirectional_mode="ulf_to_hf", contrasts=None,
            hf_resolution=None,
        )
        with pytest.raises(ValueError, match="Expected manifest_version='4.0'"):
            IndexBuilder.load_paired_bids_manifest(
                manifest_path=p, split="train", config=cfg
            )

    def test_split_filter_train(self, mock_resolve, tmp_path):
        records = [
            _fake_record(split_hint="train"),
            _fake_record(sub="sub-0002", split_hint="val"),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False, bidirectional_mode="ulf_to_hf", contrasts=None,
            hf_resolution=None,
        )
        result = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        assert len(result) == 1
        assert result[0]["split_hint"] == "train"

    def test_split_filter_val(self, mock_resolve, tmp_path):
        records = [
            _fake_record(split_hint="train"),
            _fake_record(sub="sub-0002", split_hint="val"),
            _fake_record(
                sub="sub-0003",
                split_hint="val",
                pairing_status="unpaired_ulf",
                target=None,
                hf_resolution=None,
            ),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=True, bidirectional_mode="ulf_to_hf", contrasts=None,
            hf_resolution=None,
        )
        result = IndexBuilder.load_paired_bids_manifest(p, "val", cfg)
        assert len(result) == 2

    def test_allow_unpaired_false_removes_unpaired(self, mock_resolve, tmp_path):
        records = [
            _fake_record(split_hint="val", pairing_status="paired"),
            _fake_record(
                sub="sub-0002",
                split_hint="val",
                pairing_status="unpaired_ulf",
                target=None,
                hf_resolution=None,
            ),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False, bidirectional_mode="ulf_to_hf", contrasts=None,
            hf_resolution=None,
        )
        result = IndexBuilder.load_paired_bids_manifest(p, "val", cfg)
        assert all(r["pairing_status"] == "paired" for r in result)
        assert len(result) == 1

    def test_contrast_filter(self, mock_resolve, tmp_path):
        records = [
            _fake_record(contrast="T1w"),
            _fake_record(sub="sub-0002", contrast="T2w"),
            _fake_record(sub="sub-0003", contrast="FLAIR"),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False, bidirectional_mode="ulf_to_hf", contrasts=["T1w", "FLAIR"],
            hf_resolution=None,
        )
        result = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        assert len(result) == 2
        contrasts = {r["contrast"] for r in result}
        assert contrasts == {"T1w", "FLAIR"}

    def test_bidirectional_swap_hf_to_ulf(self, mock_resolve, tmp_path):
        """In hf_to_ulf mode, manifest does NOT swap paths (dataset handles that).

        It only drops unpaired records that lack a target_path.
        """
        records = [
            _fake_record(
                primary="/data/ulf_T1w.nii.gz", target="/data/hf_T1w.nii.gz"
            )
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False, bidirectional_mode="hf_to_ulf", contrasts=None,
            hf_resolution=None,
        )
        result = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        assert len(result) == 1
        # Paths are NOT swapped at manifest level — dataset does this at __getitem__
        assert result[0]["primary_path"] == "/data/ulf_T1w.nii.gz"
        assert result[0]["target_path"] == "/data/hf_T1w.nii.gz"

    def test_bidirectional_swap_drops_unpaired(self, mock_resolve, tmp_path):
        """In HF→ULF mode, unpaired records (no target) can't be swapped and are dropped."""
        records = [
            _fake_record(primary="/data/ulf_0.nii.gz", target="/data/hf_0.nii.gz"),
            _fake_record(
                sub="sub-0002",
                pairing_status="unpaired_ulf",
                target=None,
                primary="/data/ulf_1.nii.gz",
                hf_resolution=None,
            ),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=True, bidirectional_mode="hf_to_ulf", contrasts=None,
            hf_resolution=None,
        )
        result = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        # Only the paired record survives the swap (unpaired has no target to promote)
        assert len(result) == 1

    def test_no_records_after_filter_returns_empty_list(self, mock_resolve, tmp_path):
        records = [_fake_record(contrast="T1w", split_hint="train")]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False, bidirectional_mode="ulf_to_hf", contrasts=["FLAIR"],
            hf_resolution=None,
        )
        result = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        assert result == []

    # -----------------------------------------------------------------------
    # HF resolution filter  (NEW — Rule 5)
    # -----------------------------------------------------------------------

    def test_hf_resolution_filter_highres_only(self, mock_resolve, tmp_path):
        """hf_resolution='highres' drops lowres records, keeps highres + unpaired (None)."""
        records = [
            _fake_record(sub="sub-0001", hf_resolution="highres"),
            _fake_record(sub="sub-0002", hf_resolution="lowres"),
            _fake_record(
                sub="sub-0003",
                pairing_status="unpaired_ulf",
                target=None,
                hf_resolution=None,
            ),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False, bidirectional_mode="ulf_to_hf", contrasts=None,
            hf_resolution="highres",
        )
        cfg.pairing = DataPairingConfigSchema(unpaired_training=True, hf_resolution="highres")
        result = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        # sub-0002 (lowres) should be dropped; sub-0001 (highres) + sub-0003 (None) survive
        assert len(result) == 2
        resolutions = {r.get("hf_resolution") for r in result}
        assert "lowres" not in resolutions

    def test_hf_resolution_none_passes_all(self, mock_resolve, tmp_path):
        """hf_resolution=None means no filtering — all resolution variants pass."""
        records = [
            _fake_record(sub="sub-0001", hf_resolution="highres"),
            _fake_record(sub="sub-0002", hf_resolution="lowres"),
            _fake_record(sub="sub-0003", hf_resolution="unknown"),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False, bidirectional_mode="ulf_to_hf", contrasts=None,
            hf_resolution=None,
        )
        result = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        assert len(result) == 3


@patch("spectramr.data.metadata.index_builder.PathResolver.resolve", side_effect=_no_op_resolve)
class TestUnpairedTrainingKeepsOnlyTrainUnpaired:
    def test_train_keeps_unpaired_val_stays_paired(self, mock_resolve, tmp_path):
        records = [
            _fake_record(sub="sub-0001", split_hint="train"),
            _fake_record(sub="sub-0002", split_hint="train", target=None, pairing_status="unpaired_ulf"),
            _fake_record(sub="sub-0003", split_hint="val"),
            _fake_record(sub="sub-0004", split_hint="val", target=None, pairing_status="unpaired_ulf"),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False,
            bidirectional_mode="ulf_to_hf",
            contrasts=None,
            hf_resolution=None,
        )
        cfg.pairing = DataPairingConfigSchema(unpaired_training=True)
        train = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        val = IndexBuilder.load_paired_bids_manifest(p, "val", cfg)
        assert sorted(r["subject_id"] for r in train) == ["sub-0001", "sub-0002"]
        assert [r["subject_id"] for r in val] == ["sub-0003"]

    def test_default_drops_unpaired_from_train(self, mock_resolve, tmp_path):
        records = [
            _fake_record(sub="sub-0001", split_hint="train"),
            _fake_record(sub="sub-0002", split_hint="train", target=None, pairing_status="unpaired_ulf"),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(),
            allow_unpaired=False,
            bidirectional_mode="ulf_to_hf",
            contrasts=None,
            hf_resolution=None,
        )
        train = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        assert [r["subject_id"] for r in train] == ["sub-0001"]


@patch("spectramr.data.metadata.index_builder.PathResolver.resolve", side_effect=_no_op_resolve)
class TestUnpairedAdmissionPerSplit:
    """allow_unpaired never admits target-less records into train (identity-pair defect)."""

    def _records(self):
        return [
            _fake_record(sub="sub-0001", split_hint="train"),
            _fake_record(sub="sub-0002", split_hint="train", target=None, pairing_status="unpaired_ulf"),
        ]

    def test_allow_unpaired_alone_drops_unpaired_from_train(self, mock_resolve, tmp_path):
        p = _make_manifest_file(tmp_path, self._records())
        cfg = DataConfigStub(
            split=_split(), allow_unpaired=True, bidirectional_mode="ulf_to_hf",
            contrasts=None, hf_resolution=None,
        )
        train = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        assert [r["subject_id"] for r in train] == ["sub-0001"]

    def test_unpaired_training_admits_unpaired_to_train_only(self, mock_resolve, tmp_path):
        records = [
            *self._records(),
            _fake_record(sub="sub-0003", split_hint="val", target=None, pairing_status="unpaired_ulf"),
        ]
        p = _make_manifest_file(tmp_path, records)
        cfg = DataConfigStub(
            split=_split(), allow_unpaired=False, bidirectional_mode="ulf_to_hf",
            contrasts=None, hf_resolution=None,
        )
        cfg.pairing = DataPairingConfigSchema(unpaired_training=True)
        train = IndexBuilder.load_paired_bids_manifest(p, "train", cfg)
        assert sorted(r["subject_id"] for r in train) == ["sub-0001", "sub-0002"]
        val = IndexBuilder.load_paired_bids_manifest(p, "val", cfg)
        assert val == []


@patch("spectramr.data.metadata.index_builder.PathResolver.resolve", side_effect=_no_op_resolve)
@patch("spectramr.data.builders.manifest_loader.PathResolver.resolve", side_effect=_no_op_resolve)
def test_contrast_aware_route_admits_no_targetless_record(_m1, _m2, tmp_path):
    """The contrast-aware route (`load_paired_nifti_splits`) filters target-less records
    out of train and val, so the dataset's TargetlessSampleError never fires on a correct index."""
    from spectramr.data.builders.manifest_loader import ManifestLoader

    records = [
        _fake_record(sub="sub-0001", split_hint="train"),
        _fake_record(sub="sub-0002", split_hint="train", target=None, pairing_status="unpaired_ulf"),
        _fake_record(sub="sub-0003", split_hint="val"),
        _fake_record(sub="sub-0004", split_hint="val", target=None, pairing_status="unpaired_ulf"),
    ]
    p = _make_manifest_file(tmp_path, records)
    from spectramr.config.schemas.data import DataSplitConfigSchema

    cfg = DataConfigStub(
        split=DataSplitConfigSchema(), allow_unpaired=False,
        bidirectional_mode="ulf_to_hf", contrasts=None, hf_resolution=None,
        enable_test_split=False,
    )
    cfg.source = cfg.source.model_copy(update={"paired_manifest_path": str(p)})
    train, val = ManifestLoader.load_paired_nifti_splits(cfg)
    assert [r["subject_id"] for r in train] == ["sub-0001"]
    assert [r["subject_id"] for r in val] == ["sub-0003"]
    assert all(r.get("target_path") for r in train + val)
