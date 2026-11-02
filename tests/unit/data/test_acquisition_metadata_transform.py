"""Phase 4b — LoadAcquisitionMetadata transform.

Exercises the four ingestion sources end-to-end using on-disk fixtures
written into ``tmp_path``. The DICOM path is skipped when pydicom is
unavailable (acceptable for CI on minimal environments).
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
)


def _make_subject_with_path(tmp_path: Path, basename: str = "sub-001_T1w") -> tio.Subject:
    """Create a TorchIO Subject whose ``input`` image has a real path."""
    img_path = tmp_path / f"{basename}.nii.gz"
    # Write a tiny dummy NIfTI so tio.ScalarImage(path=...) can be path-aware.
    # We don't actually use the data — only the .path attribute.
    try:
        import nibabel as nib
        import numpy as np

        nib.save(  # type: ignore[no-untyped-call]
            nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), affine=np.eye(4)),
            str(img_path),
        )
        img = tio.ScalarImage(str(img_path))
    except ImportError:
        # Fall back to a tensor image with a fake path attribute.
        img = tio.ScalarImage(tensor=torch.zeros(1, 4, 4, 4))
        img.path = img_path  # type: ignore[attr-defined]
    return tio.Subject(input=img)


# ── yaml_inline source ──────────────────────────────────────────────────────


def test_yaml_inline_copies_defaults_through() -> None:
    cfg = AcquisitionMetadataConfigSchema(
        enabled=True,
        source="yaml_inline",
        fields=["TE", "TR", "FA"],
        defaults={"TE": 4.5, "TR": 8.0, "FA": 15.0},
    )
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)))
    out = LoadAcquisitionMetadata(cfg)(subj)
    md = out["acquisition_metadata"]
    assert md == {"TE": 4.5, "TR": 8.0, "FA": 15.0}


def test_yaml_inline_strict_raises_on_missing_field() -> None:
    cfg = AcquisitionMetadataConfigSchema(
        enabled=True,
        source="yaml_inline",
        fields=["TE", "TR"],
        defaults={"TE": 4.5},  # TR missing!
        strict=True,
    )
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)))
    with pytest.raises(MissingAcquisitionMetadata, match="TR"):
        LoadAcquisitionMetadata(cfg)(subj)


def test_yaml_inline_lenient_falls_back_to_nan() -> None:
    cfg = AcquisitionMetadataConfigSchema(
        enabled=True,
        source="yaml_inline",
        fields=["TE", "TR"],
        defaults={"TE": 4.5},
        strict=False,
    )
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)))
    out = LoadAcquisitionMetadata(cfg)(subj)
    md = out["acquisition_metadata"]
    assert md["TE"] == 4.5
    assert math.isnan(md["TR"])


# ── bids_json source ────────────────────────────────────────────────────────


def test_bids_json_reads_sidecar_and_converts_seconds_to_ms(tmp_path: Path) -> None:
    """BIDS stores TE in seconds; the transform must convert to ms."""
    # Write a sidecar.
    sidecar = tmp_path / "sub-001_T1w.json"
    sidecar.write_text(
        json.dumps(
            {
                "EchoTime": 0.0045,  # 4.5 ms
                "RepetitionTime": 2.5,  # 2500 ms
                "FlipAngle": 15.0,
                "MagneticFieldStrength": 3.0,
            }
        )
    )
    subj = _make_subject_with_path(tmp_path, "sub-001_T1w")

    cfg = AcquisitionMetadataConfigSchema(
        enabled=True,
        source="bids_json",
        fields=["TE", "TR", "FA", "B0"],
    )
    out = LoadAcquisitionMetadata(cfg)(subj)
    md = out["acquisition_metadata"]
    assert md["TE"] == pytest.approx(4.5)
    assert md["TR"] == pytest.approx(2500.0)
    assert md["FA"] == pytest.approx(15.0)
    assert md["B0"] == pytest.approx(3.0)


def test_bids_json_missing_sidecar_strict_raises(tmp_path: Path) -> None:
    """No sidecar found AND strict=True ⇒ MissingAcquisitionMetadata."""
    subj = _make_subject_with_path(tmp_path, "sub-002_T1w")
    # No sidecar written.
    cfg = AcquisitionMetadataConfigSchema(
        enabled=True,
        source="bids_json",
        fields=["TE"],
        strict=True,
    )
    with pytest.raises(MissingAcquisitionMetadata, match="TE"):
        LoadAcquisitionMetadata(cfg)(subj)


def test_bids_json_missing_sidecar_lenient_uses_default(tmp_path: Path) -> None:
    """strict=False + default present ⇒ default wins."""
    subj = _make_subject_with_path(tmp_path, "sub-003_T1w")
    cfg = AcquisitionMetadataConfigSchema(
        enabled=True,
        source="bids_json",
        fields=["TE"],
        defaults={"TE": 2.0},
        strict=False,
    )
    out = LoadAcquisitionMetadata(cfg)(subj)
    assert out["acquisition_metadata"]["TE"] == 2.0


def test_bids_json_partial_sidecar_fills_with_defaults(tmp_path: Path) -> None:
    """A sidecar with TE only + defaults filling TR ⇒ both populated."""
    sidecar = tmp_path / "sub-004_T1w.json"
    sidecar.write_text(json.dumps({"EchoTime": 0.003}))
    subj = _make_subject_with_path(tmp_path, "sub-004_T1w")

    cfg = AcquisitionMetadataConfigSchema(
        enabled=True,
        source="bids_json",
        fields=["TE", "TR"],
        defaults={"TR": 1000.0},
    )
    out = LoadAcquisitionMetadata(cfg)(subj)
    md = out["acquisition_metadata"]
    assert md["TE"] == pytest.approx(3.0)
    assert md["TR"] == pytest.approx(1000.0)


# ── Attached to TorchIO Compose ──────────────────────────────────────────────


def test_works_inside_torchio_compose() -> None:
    """The transform must be composable like any other ``tio.Transform``."""
    cfg = AcquisitionMetadataConfigSchema(
        enabled=True,
        source="yaml_inline",
        fields=["TE", "TR"],
        defaults={"TE": 5.0, "TR": 9.0},
    )
    transform = tio.Compose([LoadAcquisitionMetadata(cfg)])
    subj = tio.Subject(input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 2)))
    out = transform(subj)
    assert out["acquisition_metadata"]["TE"] == 5.0


# ── ismrmrd_header (M4Raw) ───────────────────────────────────────────────────


def _write_ismrmrd_h5(path: Path, *, ti: str | None = "0") -> Path:
    """An h5 whose header carries M4Raw's ``ns0:``-prefixed sequence parameters."""
    import h5py

    ti_tag = f"<ns0:TI>{ti}</ns0:TI>" if ti is not None else ""
    xml = (
        '<?xml version="1.0"?><ns0:ismrmrdHeader xmlns:ns0="http://www.ismrm.org/ISMRMRD">'
        "<ns0:acquisitionSystemInformation><ns0:systemFieldStrength_T>0.3"
        "</ns0:systemFieldStrength_T></ns0:acquisitionSystemInformation>"
        f"<ns0:sequenceParameters><ns0:TR>500</ns0:TR><ns0:TE>18.4</ns0:TE>{ti_tag}"
        "<ns0:flipAngle_deg>180</ns0:flipAngle_deg></ns0:sequenceParameters>"
        "</ns0:ismrmrdHeader>"
    )
    with h5py.File(str(path), "w") as f:
        f["ismrmrd_header"] = xml.encode()
    return path


def _header_cfg(**kw) -> AcquisitionMetadataConfigSchema:
    return AcquisitionMetadataConfigSchema(
        enabled=True, source="ismrmrd_header", fields=["TE", "TR", "TI", "FA", "B0"], **kw
    )


def test_ismrmrd_header_reads_all_five_fields_via_source_path(tmp_path: Path) -> None:
    """An in-memory subject names its file under ``source_path``; the header fills phi."""
    path = _write_ismrmrd_h5(tmp_path / "2022_T101.h5")
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 1)), source_path=str(path)
    )
    md = LoadAcquisitionMetadata(_header_cfg(strict=True))(subj)["acquisition_metadata"]
    assert md == {"TE": 18.4, "TR": 500.0, "TI": 0.0, "FA": 180.0, "B0": 0.3}


def test_ismrmrd_header_absent_tag_raises_in_strict_mode(tmp_path: Path) -> None:
    path = _write_ismrmrd_h5(tmp_path / "2022_T201.h5", ti=None)
    subj = tio.Subject(
        input=tio.ScalarImage(tensor=torch.zeros(1, 2, 2, 1)), source_path=str(path)
    )
    with pytest.raises(MissingAcquisitionMetadata, match="TI"):
        LoadAcquisitionMetadata(_header_cfg(strict=True))(subj)
