"""Unit tests for spectramr.data.nex_averaging.

Each synthetic repetition carries its own global receiver phase, as M4Raw's do. The
average must be the loader's ``phase_aligned_mean`` target, which keeps the signal a
plain complex mean cancels, and must land as one parseable file per contrast.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.datasets.m4raw_dataset import _average_reps
from spectramr.data.datasets.m4raw_identity import parse_m4raw_file_id
from spectramr.data.nex_averaging import average_split, group_split
from tests.fixtures import m4raw_phantom as ph

SUBJECT = "2022000001"


def _rep(path: Path, phase: float, *, real_storage: bool = False, rss: bool = False) -> Path:
    """One repetition: the phantom's k-space rotated by a global ``phase``, plus a header."""
    data = ph.kspace(ph.phantom(1.0)) * np.exp(1j * phase).astype(np.complex64)
    if real_storage:
        data = np.stack([data.real, data.imag], axis=-1).astype(np.float32)
    with h5py.File(path, "w") as f:
        f["kspace"] = data
        f["ismrmrd_header"] = ph.header()
        if rss:
            f["reconstruction_rss"] = np.ones((ph.SLICES, ph.SIZE, ph.SIZE), np.float32)
        f.attrs["patient_id"] = SUBJECT
    return path


def _split(src: Path, phases: tuple[float, ...] = (0.0, np.pi), **kw) -> Path:
    src.mkdir(parents=True, exist_ok=True)
    for i, phase in enumerate(phases, start=1):
        _rep(src / f"{SUBJECT}_T2{i:02d}.h5", phase, **kw)
    return src


def _kspace(path: Path) -> torch.Tensor:
    with h5py.File(path) as f:
        return torch.from_numpy(f["kspace"][()])


def test_opposed_receiver_phases_average_to_the_signal_not_to_zero(tmp_path: Path) -> None:
    src = _split(tmp_path / "src")
    average_split(src, tmp_path / "dst")
    averaged = _kspace(tmp_path / "dst" / f"{SUBJECT}_T200.h5")
    reps = [_kspace(p) for p in sorted(src.glob("*.h5"))]
    assert torch.allclose(averaged, reps[0], atol=1e-5)  # aligned onto rep 01's frame
    assert _average_reps(reps, "complex_mean").abs().max() < 1e-5  # what alignment prevents


def test_the_average_is_the_loaders_own_nex_target(tmp_path: Path) -> None:
    src = _split(tmp_path / "src", (0.3, 2.1, -1.2))
    average_split(src, tmp_path / "dst")
    reps = [_kspace(p) for p in sorted(src.glob("*.h5"))]
    expected = _average_reps(reps, "phase_aligned_mean")
    assert torch.equal(_kspace(tmp_path / "dst" / f"{SUBJECT}_T200.h5"), expected)


def test_one_file_per_contrast_named_as_an_average_with_its_provenance(tmp_path: Path) -> None:
    src = _split(tmp_path / "src", (0.0, 1.0, 2.0), rss=True)
    _rep(src / f"{SUBJECT}_FLAIR01.h5", 0.0)
    records = average_split(src, tmp_path / "dst")
    assert sorted(p.name for p in (tmp_path / "dst").iterdir()) == [
        f"{SUBJECT}_FLAIR00.h5",
        f"{SUBJECT}_T200.h5",
    ]
    assert [(r["contrast"], len(r["sources"]), r["status"]) for r in records] == [
        ("FLAIR", 1, "averaged"),
        ("T2", 3, "averaged"),
    ]
    assert parse_m4raw_file_id(f"{SUBJECT}_T200").contrast == "T2"
    with h5py.File(tmp_path / "dst" / f"{SUBJECT}_T200.h5") as f:
        assert f.attrs["nex_average_mode"] == "phase_aligned_mean"
        assert f.attrs["nex_average_count"] == 3
        assert list(f.attrs["nex_average_sources"]) == [f"{SUBJECT}_T20{i}.h5" for i in (1, 2, 3)]
        assert f.attrs["patient_id"] == SUBJECT
        assert f["ismrmrd_header"][()] == ph.header()
        # Reconstructed from rep 01 alone, it would contradict the averaged k-space.
        assert "reconstruction_rss" not in f
        assert list(f.attrs["dropped_kspace_derived_datasets"]) == ["reconstruction_rss"]


def test_real_imag_storage_is_written_back_in_kind(tmp_path: Path) -> None:
    src = _split(tmp_path / "src", real_storage=True)
    average_split(src, tmp_path / "dst")
    with h5py.File(tmp_path / "dst" / f"{SUBJECT}_T200.h5") as f:
        assert f["kspace"].dtype == np.float32 and f["kspace"].shape[-1] == 2


def test_repetitions_on_different_grids_are_refused_and_not_written(tmp_path: Path) -> None:
    src = _split(tmp_path / "src", (0.0,))
    with h5py.File(src / f"{SUBJECT}_T202.h5", "w") as f:
        f["kspace"] = ph.kspace(ph.phantom(1.0)[:-2])
        f["ismrmrd_header"] = ph.header()
    (record,) = average_split(src, tmp_path / "dst")
    assert record["status"].startswith("refused: repetitions on different grids")
    assert not (tmp_path / "dst" / f"{SUBJECT}_T200.h5").exists()


@pytest.mark.parametrize(
    ("name", "match"),
    [(f"{SUBJECT}_T200.h5", "already an average"), ("scan.h5", "is not <subject>")],
)
def test_a_file_that_is_not_an_acquired_repetition_stops_the_split(
    tmp_path: Path, name: str, match: str
) -> None:
    src = _split(tmp_path / "src")
    _rep(src / name, 0.0)
    with pytest.raises(ValueError, match=match):
        average_split(src, tmp_path / "dst")
    assert not (tmp_path / "dst").exists()


def test_a_non_empty_destination_is_refused(tmp_path: Path) -> None:
    src = _split(tmp_path / "src")
    (tmp_path / "dst").mkdir()
    (tmp_path / "dst" / "stale.h5").touch()
    with pytest.raises(FileExistsError, match="not empty"):
        average_split(src, tmp_path / "dst")


def test_grouping_reads_names_in_repetition_order_and_writes_nothing(tmp_path: Path) -> None:
    src = _split(tmp_path / "src", (0.0, 1.0, 2.0))
    _rep(src / f"{SUBJECT}_FLAIR01.h5", 0.0)
    before = sorted(src.iterdir())
    groups = group_split(src)
    assert [p.name for p in groups[(SUBJECT, "T2")]] == [f"{SUBJECT}_T20{i}.h5" for i in (1, 2, 3)]
    assert set(groups) == {(SUBJECT, "T2"), (SUBJECT, "FLAIR")}
    assert sorted(src.iterdir()) == before
