"""One dataset_kind -> ContrastGrid dispatch; the M4Raw law never sees brain files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from spectramr.data.datasets.native_grid import (
    DATASET_KINDS,
    NATIVE_ACQUISITION,
    discover_native_grid,
)


def _brain(root: Path) -> None:
    for pid in ("200_6002001", "200_6002002"):
        with h5py.File(str(root / f"file_brain_AXT1_{pid}.h5"), "w") as f:
            f.create_dataset("kspace", data=np.zeros((1, 4, 4), dtype=np.complex64))
            f.attrs["acquisition"] = "AXT1"


def test_kinds() -> None:
    assert set(DATASET_KINDS) == {"m4raw", "fastmri_brain"}


def test_every_kind_declares_its_native_acquisition() -> None:
    from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode

    assert set(NATIVE_ACQUISITION) == set(DATASET_KINDS)
    assert {AcquisitionMode(v) for v in NATIVE_ACQUISITION.values()} == {
        AcquisitionMode.MULTISLICE_2D
    }


def test_brain_kind_keeps_one_file_per_exam(tmp_path: Path) -> None:
    _brain(tmp_path)
    grid = discover_native_grid("fastmri_brain", tmp_path, max_contrasts=1)
    assert not grid.linked
    assert [len(k.file_ids) for k in grid.keys] == [1, 1]


def test_m4raw_kind_refuses_brain_names(tmp_path: Path) -> None:
    _brain(tmp_path)
    with pytest.raises(ValueError, match="not an M4Raw"):
        discover_native_grid("m4raw", tmp_path)


def test_m4raw_kind_reads_allow_incomplete(tmp_path: Path) -> None:
    for subj, cs in (("s1", ("T1", "T2")), ("s2", ("T1",))):
        for c in cs:
            for r in (1, 2):
                (tmp_path / f"{subj}_{c}{r:02d}.h5").write_bytes(b"")
    with pytest.raises(ValueError, match="s2"):
        discover_native_grid("m4raw", tmp_path, max_contrasts=2)
    grid = discover_native_grid("m4raw", tmp_path, max_contrasts=2, allow_incomplete=True)
    assert grid.dropped_subjects == ("s2",)


def test_unknown_kind_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown dataset_kind"):
        discover_native_grid("fastmri_knee", tmp_path)


def test_allow_incomplete_is_refused_for_unlinked_brain(tmp_path: Path) -> None:
    _brain(tmp_path)
    with pytest.raises(ValueError, match="does not apply to fastmri_brain"):
        discover_native_grid("fastmri_brain", tmp_path, allow_incomplete=True)
