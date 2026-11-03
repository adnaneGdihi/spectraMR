"""Unit tests for acquisition_metadata transform.

Covers:
  - LoadAcquisitionMetadata: yaml_inline source copies defaults through,
    strict mode raises on missing field, lenient mode fills NaN,
    subject gains 'acquisition_metadata' dict.
  - BIDS-JSON reader: seconds-to-milliseconds conversion, missing sidecar
    returns empty dict.
  - _coalesce: merge logic for parsed vs defaults vs strict.
  - parse helpers: _from_bids_json key-mapping.
  - MissingAcquisitionMetadata is a KeyError subclass.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torchio as tio

from spectramr.config.schemas.data import AcquisitionMetadataConfigSchema
from spectramr.data.transforms.acquisition_metadata import (
    LoadAcquisitionMetadata,
    MissingAcquisitionMetadata,
    _coalesce,
    _from_bids_json,
    _read_metadata,
)


# ── MissingAcquisitionMetadata ────────────────────────────────────────────────


def test_missing_metadata_is_key_error_subclass() -> None:
    assert issubclass(MissingAcquisitionMetadata, KeyError)


# ── yaml_inline source ────────────────────────────────────────────────────────


def test_canary_yaml_inline_attaches_metadata_to_subject() -> None:
    """Minimal end-to-end: yaml_inline copies all requested defaults to subject."""
    cfg = AcquisitionMetadataConfigSchema(
        source="yaml_inline",
        fields=["TE", "TR", "FA"],
        defaults={"TE": 4.5, "TR": 8.0, "FA": 15.0},
    )
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 4)))
    out = LoadAcquisitionMetadata(cfg)(subj)
    assert "acquisition_metadata" in out
    md = out["acquisition_metadata"]
    assert md["TE"] == pytest.approx(4.5)
    assert md["TR"] == pytest.approx(8.0)
    assert md["FA"] == pytest.approx(15.0)


def test_yaml_inline_passes_all_listed_fields() -> None:
    cfg = AcquisitionMetadataConfigSchema(
        source="yaml_inline",
        fields=["TE", "TR", "B0"],
        defaults={"TE": 2.5, "TR": 5.0, "B0": 3.0},
    )
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)))
    out = LoadAcquisitionMetadata(cfg)(subj)
    md = out["acquisition_metadata"]
    assert set(md.keys()) == {"TE", "TR", "B0"}


# ── Strict / lenient handling ─────────────────────────────────────────────────


def test_strict_mode_raises_on_missing_field() -> None:
    """strict=True and a required field absent → MissingAcquisitionMetadata."""
    cfg = AcquisitionMetadataConfigSchema(
        source="yaml_inline",
        fields=["TE", "TR", "TI"],
        defaults={"TE": 2.5, "TR": 5.0},  # TI missing
        strict=True,
    )
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)))
    with pytest.raises(MissingAcquisitionMetadata, match="TI"):
        LoadAcquisitionMetadata(cfg)(subj)


def test_lenient_mode_fills_nan_on_missing_field() -> None:
    """strict=False and a field absent from defaults → NaN inserted."""
    cfg = AcquisitionMetadataConfigSchema(
        source="yaml_inline",
        fields=["TE", "TI"],
        defaults={"TE": 2.5},  # TI not in defaults
        strict=False,
    )
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)))
    out = LoadAcquisitionMetadata(cfg)(subj)
    assert math.isnan(out["acquisition_metadata"]["TI"])


# ── _coalesce unit tests ──────────────────────────────────────────────────────


def test_coalesce_uses_parsed_over_defaults() -> None:
    result = _coalesce(
        parsed={"TE": 3.0},
        defaults={"TE": 10.0},
        fields=["TE"],
        strict=False,
        subject_id=None,
    )
    assert result["TE"] == pytest.approx(3.0)


def test_coalesce_falls_back_to_defaults_when_parsed_missing() -> None:
    result = _coalesce(
        parsed={},
        defaults={"TE": 10.0},
        fields=["TE"],
        strict=False,
        subject_id=None,
    )
    assert result["TE"] == pytest.approx(10.0)


def test_coalesce_strict_raises_when_both_absent() -> None:
    with pytest.raises(MissingAcquisitionMetadata, match="FA"):
        _coalesce(
            parsed={},
            defaults={},
            fields=["FA"],
            strict=True,
            subject_id="sub-001",
        )


def test_coalesce_lenient_inserts_nan_when_both_absent() -> None:
    result = _coalesce(
        parsed={},
        defaults={},
        fields=["FA"],
        strict=False,
        subject_id=None,
    )
    assert math.isnan(result["FA"])


# ── _from_bids_json ───────────────────────────────────────────────────────────


def test_bids_json_seconds_to_milliseconds(tmp_path: Path) -> None:
    """BIDS stores TE/TR in seconds; the reader must convert to ms."""
    sidecar = tmp_path / "sub-001_T1w.json"
    sidecar.write_text(
        json.dumps({"EchoTime": 0.0045, "RepetitionTime": 0.008, "FlipAngle": 15.0})
    )
    result = _from_bids_json(sidecar)
    assert result["TE"] == pytest.approx(4.5, rel=1e-4)
    assert result["TR"] == pytest.approx(8.0, rel=1e-4)
    assert result["FA"] == pytest.approx(15.0)


def test_bids_json_missing_key_is_silently_omitted(tmp_path: Path) -> None:
    """Keys absent from the sidecar are simply not emitted (not set to NaN)."""
    sidecar = tmp_path / "sub-002_T1w.json"
    sidecar.write_text(json.dumps({"EchoTime": 0.003}))
    result = _from_bids_json(sidecar)
    assert "TE" in result
    assert "TR" not in result


# ── _read_metadata dispatch ───────────────────────────────────────────────────


def test_read_metadata_yaml_inline_copies_defaults() -> None:
    defaults = {"TE": 5.0, "TR": 10.0}
    result = _read_metadata("yaml_inline", image_path=None, defaults=defaults)
    assert result == defaults


def test_read_metadata_none_path_returns_defaults() -> None:
    defaults = {"TE": 2.0}
    result = _read_metadata("bids_json", image_path=None, defaults=defaults)
    assert result == defaults


def test_read_metadata_missing_bids_sidecar_returns_empty(tmp_path: Path) -> None:
    """No sidecar present → empty dict (not an error)."""
    fake_img = tmp_path / "sub-001_T1w.nii.gz"
    result = _read_metadata("bids_json", image_path=fake_img, defaults={})
    assert result == {}


def test_read_metadata_unknown_source_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        _read_metadata("csv_sidecar", image_path=None, defaults={})  # type: ignore[arg-type]


def test_ismrmrd_reader_is_the_data_layer_one() -> None:
    """The header file read moved out of physics (#3421); the transform binds the data-layer reader."""
    from spectramr.data.datasets import kspace_volume_stack
    from spectramr.data.transforms import acquisition_metadata

    assert acquisition_metadata.read_acquisition_params is kspace_volume_stack.read_acquisition_params
