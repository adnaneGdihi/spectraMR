"""Tests for the noise-unit M4Raw triplet dataset (exp 32, gain-calibrated maps)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.datasets.m4raw_triplet_dataset import M4RawTripletDataset, family_of

# Header values as the carya census found them (2026-10-05).
_PROTO = {
    "T1": {"TR": 500, "TE": 18.4, "TI": 0, "fa": 180, "etl": 1, "esp": "N/A"},
    "T2": {"TR": 5500, "TE": 128, "TI": 0, "fa": 160, "etl": 15, "esp": "16.0ms"},
    "FLAIR": {"TR": 7500, "TE": 98, "TI": 1655, "fa": 160, "etl": 11, "esp": "16.3ms"},
}


def _header(p: dict) -> str:
    return (
        "<ismrmrdHeader><sequenceParameters>"
        f"<TR>{p['TR']}</TR><TE>{p['TE']}</TE><TI>{p['TI']}</TI>"
        f"<flipAngle_deg>{p['fa']}</flipAngle_deg>"
        f"<echo_train_length>{p['etl']}</echo_train_length>"
        f"<echo_spacing>{p['esp']}</echo_spacing>"
        "</sequenceParameters></ismrmrdHeader>"
    )


def _kspace(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    k = torch.complex(
        torch.randn(3, 4, 32, 32, generator=g), torch.randn(3, 4, 32, 32, generator=g)
    )
    k[..., 12:20, 12:20] += 30.0
    return k


def _write(
    path: Path,
    contrast: str,
    *,
    registered: bool,
    nex: int | None = 3,
    kspace: torch.Tensor | None = None,
    source: Path | None = None,
    **override,
) -> None:
    proto = {**_PROTO[contrast], **override}
    k = _kspace(len(path.name)) if kspace is None else kspace
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=k.numpy().astype(np.complex64))
        f["ismrmrd_header"] = np.bytes_(_header(proto).encode())
        if nex is not None:
            f.attrs["nex_average_count"] = nex
        if registered:
            f.attrs["registration_valid_slices"] = np.array([True, True, False])
        if source is not None:
            f.attrs["registration_source_file"] = str(source)


def _tree(tmp_path: Path, *, s2_flair_registered=False, s2_t2_override=None) -> tuple[Path, Path]:
    a, b = tmp_path / "A" / "split", tmp_path / "B" / "split"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    for s, flair_ok, t2_override in (
        ("s1", True, {}),
        ("s2", s2_flair_registered, s2_t2_override or {}),
    ):
        _write(a / f"{s}_T100.h5", "T1", registered=False)
        _write(a / f"{s}_T200.h5", "T2", registered=True, **t2_override)
        _write(b / f"{s}_FLAIR00.h5", "FLAIR", registered=flair_ok)
    return tmp_path / "A", tmp_path / "B"


def test_items_are_noise_unit_triplets_with_header_protocols(tmp_path: Path) -> None:
    ds = M4RawTripletDataset(*_tree(tmp_path), split="split")
    item = ds[0]
    x = item["input"].data
    assert x.shape == (3, 32, 32, 1)
    assert item["acq_tr"].tolist() == [500.0, 5500.0, 7500.0]
    assert item["acq_etl"].tolist() == [1, 15, 11]
    assert item["sequence_family"].tolist() == [0, 0, 1]
    assert 0.5 < float(x[:, :4, :4].mean()) < 3.5  # noise floor near the RSS of unit noise


def test_single_echo_t1_takes_its_echo_at_te(tmp_path: Path) -> None:
    """ETL 1 has no echo spacing; its only echo is at TE (definition, not a fallback)."""
    item = M4RawTripletDataset(*_tree(tmp_path), split="split")[0]
    assert item["acq_esp"].tolist() == pytest.approx([18.4, 16.0, 16.3])


def test_unregistered_subject_is_excluded(tmp_path: Path) -> None:
    ds = M4RawTripletDataset(*_tree(tmp_path), split="split")
    assert {r[0] for r in ds.records} == {"s1"}
    assert len(ds) == 2


def test_zero_refocusing_header_excludes_the_subject_with_a_reason(tmp_path: Path) -> None:
    """25 of 699 M4Raw T2 headers report 0 deg refocusing; render them at 0 deg never."""
    trees = _tree(tmp_path, s2_flair_registered=True, s2_t2_override={"fa": 0.0})
    ds = M4RawTripletDataset(*trees, split="split")
    assert {r[0] for r in ds.records} == {"s1"}
    assert ds.excluded["s2"] == "header refocusing angle 0"


def test_mixed_protocols_in_one_split_raise(tmp_path: Path) -> None:
    trees = _tree(tmp_path, s2_flair_registered=True, s2_t2_override={"TR": 6000})
    with pytest.raises(ValueError, match="mixes protocols"):
        M4RawTripletDataset(*trees, split="split")


def test_family_of_refuses_gre(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GRE"):
        family_of({"TR": 270.0, "TE": 7.0, "flip_angle": 70.0}, tmp_path / "x.h5")


def test_an_empty_tree_raises(tmp_path: Path) -> None:
    (tmp_path / "A" / "split").mkdir(parents=True)
    (tmp_path / "B" / "split").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no registered triplet"):
        M4RawTripletDataset(tmp_path / "A", tmp_path / "B", split="split")


def _one_subject(tmp_path: Path, *, t2_kspace=None, t2_source=None, t2_nex=3) -> tuple[Path, Path]:
    a, b = tmp_path / "A" / "split", tmp_path / "B" / "split"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    _write(a / "s1_T100.h5", "T1", registered=False)
    _write(a / "s1_T200.h5", "T2", registered=True, kspace=t2_kspace, source=t2_source, nex=t2_nex)
    _write(b / "s1_FLAIR00.h5", "FLAIR", registered=True, nex=2)
    return tmp_path / "A", tmp_path / "B"


def test_a_registered_series_takes_its_noise_support_from_its_source(tmp_path: Path) -> None:
    """Registration un-zeroes the never-acquired lines (#2613); its own support reads as
    fully sampled and the noise comes out ~0.77x. The source file's support must be used."""
    from spectramr.data.contrast_registration import RigidFit, apply_fit
    from spectramr.data.m4raw_noise_units import noise_power

    source_k = _kspace(7)
    source_k[..., 24:] = 0  # 25 % of phase-encode columns never acquired
    source = tmp_path / "source_T200.h5"
    _write(source, "T2", registered=False, kspace=source_k)
    registered, _ = apply_fit(
        source_k, RigidFit(shift_px=(0.0, 0.0, 0.0), slice_offset=0, rotation_deg=0.0)
    )
    ds = M4RawTripletDataset(
        *_one_subject(tmp_path, t2_kspace=registered, t2_source=source), split="split"
    )
    t2_path = tmp_path / "A" / "split" / "s1_T200.h5"
    assert abs(ds._sigma2[t2_path] / noise_power(source_k) - 1) < 0.05


def test_images_are_in_per_repetition_noise_units(tmp_path: Path) -> None:
    """An N-average series is sqrt(N) brighter in its own noise units; dividing by
    sqrt(N) puts every series in the per-repetition units the gate's ratios use."""
    k = _kspace(11)
    plain = M4RawTripletDataset(*_one_subject(tmp_path / "x", t2_kspace=k, t2_nex=1), split="split")
    four = M4RawTripletDataset(*_one_subject(tmp_path / "y", t2_kspace=k, t2_nex=4), split="split")
    t2_plain, t2_four = plain[0]["input"].data[1], four[0]["input"].data[1]
    assert torch.allclose(t2_four, t2_plain / 2.0, rtol=1e-4)


def test_a_series_without_its_averaging_count_excludes_the_subject(tmp_path: Path) -> None:
    trees = _one_subject(tmp_path / "x", t2_nex=None)
    (tmp_path / "y" / "A" / "split").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no registered triplet"):
        M4RawTripletDataset(*trees, split="split")
