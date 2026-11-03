"""fastMRI-brain grid: unlinked exams, one file per content, conflicts raise (#308, #344)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from spectramr.data.datasets.contrast_grid import ContrastMismatchError
from spectramr.data.datasets.fastmri_brain_groups import brain_contrast_of, discover_brain_grid


def _brain(path: Path, acquisition: str | None) -> Path:
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=np.zeros((1, 4, 4), dtype=np.complex64))
        if acquisition is not None:
            f.attrs["acquisition"] = acquisition
    return path


class TestBrainGrid:
    def test_attr_filename_conflict_raises(self, tmp_path: Path) -> None:
        p = _brain(tmp_path / "file_brain_AXT1_200_2000001.h5", "AXT2")
        with pytest.raises(ContrastMismatchError, match="broken provenance"):
            brain_contrast_of(p)

    def test_unlinked_grid_with_exam_ids(self, tmp_path: Path) -> None:
        for pid in ("200_6002001", "200_6002002", "201_6002011"):
            _brain(tmp_path / f"file_brain_AXT1_{pid}.h5", "AXT1")
        for pid in ("200_6002101", "202_6002102"):
            _brain(tmp_path / f"file_brain_AXT2_{pid}.h5", "AXT2")
        grid = discover_brain_grid(tmp_path, max_subjects=5, max_contrasts=2)
        assert not grid.linked and grid.row_label == "index"
        assert grid.contrasts == ("AXT1", "AXT2")
        assert grid.n_rows == 2
        assert grid.keys[0].subject_id == "200_6002001"
        assert all(len(k.file_ids) == 1 for k in grid.keys)
        assert grid.select(contrast="AXT2").subjects == ("200_6002101", "202_6002102")

    def test_explicit_and_max_contrasts(self, tmp_path: Path) -> None:
        for c, n in (("AXT1", 3), ("AXT2", 2), ("AXT2FLAIR", 1)):
            for i in range(n):
                _brain(tmp_path / f"file_brain_{c}_200_{i:07d}.h5", c)
        assert discover_brain_grid(tmp_path, max_contrasts=1).contrasts == ("AXT1",)
        grid = discover_brain_grid(tmp_path, contrasts=["AXT2FLAIR", "AXT1"])
        assert grid.contrasts == ("AXT2FLAIR", "AXT1") and grid.n_rows == 1
        with pytest.raises(ValueError, match="not found"):
            discover_brain_grid(tmp_path, contrasts=["AXDWI"])

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No H5 files"):
            discover_brain_grid(tmp_path)

    def test_filename_token_when_no_attr(self, tmp_path: Path) -> None:
        p = _brain(tmp_path / "file_brain_AXT2FLAIR_201_6002670.h5", None)
        assert brain_contrast_of(p) == "AXT2FLAIR"

    def test_bytes_attr_is_decoded(self, tmp_path: Path) -> None:
        p = tmp_path / "vol.h5"
        with h5py.File(str(p), "w") as f:
            f.attrs["acquisition"] = np.bytes_(b"AXFLAIR")
        assert brain_contrast_of(p) == "AXFLAIR"

    def test_unrecognisable_contrast_raises(self, tmp_path: Path) -> None:
        p = _brain(tmp_path / "mystery_volume.h5", None)
        with pytest.raises(ValueError, match="Cannot determine a contrast"):
            brain_contrast_of(p)

    def test_imbalanced_buckets_are_capped_to_the_min(self, tmp_path: Path, caplog) -> None:
        for c, n in (("AXT1", 5), ("AXT2", 3), ("AXT2FLAIR", 2)):
            for i in range(n):
                _brain(tmp_path / f"file_brain_{c}_200_{i:07d}.h5", c)
        with caplog.at_level("INFO"):
            grid = discover_brain_grid(tmp_path, max_subjects=10, max_contrasts=3)
        assert grid.n_rows == 2 and len(grid.keys) == 6
        assert any("dropped" in r.getMessage() for r in caplog.records)
        for row in grid.rows():
            assert tuple(k.contrast for k in row) == grid.contrasts
