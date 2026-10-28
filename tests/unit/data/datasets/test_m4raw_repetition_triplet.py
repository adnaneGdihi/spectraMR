"""Repetition-split M4Raw triplets: input and target are independent repetitions, co-registered."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from spectramr.data.datasets.m4raw_repetition_triplet import M4RawRepetitionTripletDataset
from spectramr.data.m4raw_noise_units import noise_unit_rss, scan_noise_power

_PROTO = {
    "T1": {"TR": 500, "TE": 18.4, "TI": 0, "fa": 180, "etl": 1, "esp": "N/A"},
    "T2": {"TR": 5500, "TE": 128, "TI": 0, "fa": 160, "etl": 15, "esp": "16.0ms"},
    "FLAIR": {"TR": 7500, "TE": 98, "TI": 1655, "fa": 160, "etl": 11, "esp": "16.3ms"},
}
_SHIFT = {"T2": (0.0, 3.0, -2.0), "FLAIR": (1.0, 0.0, 4.0)}  # (slice, y, x) pixels
_OFFSET = {"T2": 0, "FLAIR": 1}


def _header(p: dict) -> str:
    return (
        "<ismrmrdHeader><sequenceParameters>"
        f"<TR>{p['TR']}</TR><TE>{p['TE']}</TE><TI>{p['TI']}</TI>"
        f"<flipAngle_deg>{p['fa']}</flipAngle_deg>"
        f"<echo_train_length>{p['etl']}</echo_train_length>"
        f"<echo_spacing>{p['esp']}</echo_spacing>"
        "</sequenceParameters></ismrmrdHeader>"
    )


def _kspace(seed: int, skipped_columns: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    k = torch.complex(
        torch.randn(4, 4, 32, 32, generator=g), torch.randn(4, 4, 32, 32, generator=g)
    )
    k[..., 12:20, 12:20] += 30.0
    if skipped_columns:  # M4Raw ships ~24 % of its phase-encode columns as exact zeros
        k[..., 32 - skipped_columns :] = 0
    return k


def _write(path: Path, contrast: str, k: torch.Tensor, **attrs) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=k.numpy().astype(np.complex64))
        f["ismrmrd_header"] = np.bytes_(_header(_PROTO[contrast]).encode())
        f.attrs.update(attrs)


def _tree(tmp_path: Path, *, flair_reps: int = 2, skipped_columns: int = 0) -> dict:
    a, b, raw = (tmp_path / d / "split" for d in ("A", "B", "raw"))
    for d in (a, b, raw):
        d.mkdir(parents=True)
    seed = 0
    for contrast, n_reps in (("T1", 2), ("T2", 2), ("FLAIR", flair_reps)):
        for r in range(1, n_reps + 1):
            seed += 1
            _write(raw / f"s1_{contrast}0{r}.h5", contrast, _kspace(seed, skipped_columns))
    valid = np.array([False, True, True, False])
    _write(a / "s1_T100.h5", "T1", _kspace(100), nex_average_count=2)
    for contrast, root in (("T2", a), ("FLAIR", b)):
        _write(
            root / f"s1_{contrast}00.h5",
            contrast,
            _kspace(200 + len(contrast)),
            nex_average_count=2,
            registration_valid_slices=valid,
            registration_shift_px=np.asarray(_SHIFT[contrast]),
            registration_slice_offset=_OFFSET[contrast],
        )
    return {"t2_root": tmp_path / "A", "flair_root": tmp_path / "B", "raw_root": tmp_path / "raw"}


def _ds(tmp_path: Path, **kw) -> M4RawRepetitionTripletDataset:
    return M4RawRepetitionTripletDataset(**_tree(tmp_path, **kw), split="split", deterministic=True)


def test_items_pair_two_distinct_repetitions_per_contrast(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    item = ds[0]
    x, y = item["input"].data[..., 0], item["target"].data[..., 0]
    assert x.shape == y.shape == (3, 32, 32)
    assert not torch.allclose(x, y)  # independent noise
    assert item["noise_power"].tolist() == [4.0, 4.0, 4.0]
    assert item["acq_tr"].tolist() == [500.0, 5500.0, 7500.0]


def test_the_t1_repetition_is_served_unshifted_in_noise_units(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    _, s = ds.records[0]
    raw = tmp_path / "raw" / "split" / "s1_T101.h5"
    with h5py.File(raw, "r") as f:
        k = torch.from_numpy(f["kspace"][s])
    expected = noise_unit_rss(k, scan_noise_power(raw))
    assert torch.allclose(ds[0]["input"].data[0, ..., 0], expected, atol=1e-4)


def test_a_registered_repetition_takes_its_average_s_shift_and_slice_offset(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    _, s = ds.records[0]
    raw = tmp_path / "raw" / "split" / "s1_FLAIR01.h5"
    with h5py.File(raw, "r") as f:
        k = torch.from_numpy(f["kspace"][s - _OFFSET["FLAIR"]])
    image = noise_unit_rss(k, scan_noise_power(raw))
    dy, dx = int(_SHIFT["FLAIR"][1]), int(_SHIFT["FLAIR"][2])
    expected = torch.roll(image, shifts=(dy, dx), dims=(0, 1))  # integer shift = circular roll
    assert torch.allclose(ds[0]["input"].data[2, ..., 0], expected, atol=1e-3)


def test_the_noise_power_counts_only_the_acquired_samples(tmp_path: Path) -> None:
    # Per coil, an image pixel carries the noise of the acquired samples only, so in noise
    # units E[M^2] = A^2 + f C with f the acquired fraction; C alone over-subtracts.
    ds = _ds(tmp_path, skipped_columns=8)
    assert ds[0]["noise_power"].tolist() == [3.0, 3.0, 3.0]  # 4 coils x 24/32


def test_validation_pairs_are_deterministic(tmp_path: Path) -> None:
    ds = _ds(tmp_path)
    assert torch.equal(ds[0]["input"].data, ds[0]["input"].data)
    assert torch.equal(ds[0]["target"].data, ds[0]["target"].data)


def test_a_subject_with_one_repetition_of_a_contrast_is_excluded(tmp_path: Path) -> None:
    tree = _tree(tmp_path, flair_reps=1)
    try:
        M4RawRepetitionTripletDataset(**tree, split="split", deterministic=True)
    except FileNotFoundError:
        pass  # no subject left is the right outcome for this one-subject tree
    else:  # pragma: no cover
        raise AssertionError("a subject without two FLAIR repetitions must be excluded")
