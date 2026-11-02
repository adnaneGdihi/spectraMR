"""The one content model: aligned (subject, contrast) cells and one contrast parser."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from spectramr.data.datasets.contrast_grid import (
    ContentKey,
    ContrastGrid,
    ContrastMismatchError,
    IncompleteSubjectError,
    brain_filename_contrast,
    contrast_label,
    discover_m4raw_grid,
    parse_m4raw_stem,
    resolve_brain_contrast,
)

_SUBJECTS = ("2022061001", "2022061002", "2022061003")


def _touch(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def _m4raw_tree(root: Path, *, with_incomplete: bool = False) -> Path:
    for subj in _SUBJECTS:
        for c, n in (("FLAIR", 2), ("T1", 3), ("T2", 3)):
            for r in range(1, n + 1):
                _touch(root / f"{subj}_{c}{r:02d}.h5")
    if with_incomplete:
        for c in ("T1", "T2"):
            for r in (1, 2, 3):
                _touch(root / f"2022061004_{c}{r:02d}.h5")
    return root


def _brain(path: Path, acquisition: str | None) -> Path:
    with h5py.File(str(path), "w") as f:
        f.create_dataset("kspace", data=np.zeros((1, 4, 4), dtype=np.complex64))
        if acquisition is not None:
            f.attrs["acquisition"] = acquisition
    return path


def _assert_aligned(grid) -> None:
    for row in grid.rows():
        assert len({k.subject_id for k in row}) == 1, row
        assert tuple(k.contrast for k in row) == grid.contrasts


class TestParser:
    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            ("2022091411_T101", ("2022091411", "T1", 1)),
            ("2022091411_T203", ("2022091411", "T2", 3)),
            ("2022091411_FLAIR02", ("2022091411", "FLAIR", 2)),
        ],
    )
    def test_m4raw_stem(self, stem: str, expected) -> None:
        assert parse_m4raw_stem(Path(f"{stem}.h5")) == expected
        assert contrast_label(Path(f"{stem}.h5")) == expected[1]

    def test_brain_token(self) -> None:
        assert contrast_label(Path("file_brain_AXT2FLAIR_201_6002670.h5")) == "AXT2FLAIR"
        assert brain_filename_contrast("file_brain_AXT1POST_201.h5") == "AXT1POST"

    def test_unparseable_name_raises_instead_of_using_the_stem(self) -> None:
        with pytest.raises(ValueError, match="not an M4Raw"):
            contrast_label(Path("mystery_volume.h5"))

    def test_resolver_cases(self) -> None:
        assert resolve_brain_contrast("file_brain_AXT2_1.h5", b"AXT2") == "AXT2"
        assert resolve_brain_contrast("file_brain_AXFLAIR_1.h5", None) == "AXFLAIR"
        assert resolve_brain_contrast("vol.h5", "axt1") == "AXT1"
        with pytest.raises(ValueError, match="Cannot determine"):
            resolve_brain_contrast("vol.h5", None)

    def test_axial_is_not_an_acquisition_token(self) -> None:
        with pytest.raises(ValueError, match="not an M4Raw"):
            contrast_label(Path("file_brain_AXIAL_201_6002670.h5"))
        assert brain_filename_contrast("file_brain_AXIAL_AXT2_201.h5") == "AXT2"

    def test_two_ax_tokens_raise(self) -> None:
        with pytest.raises(ContrastMismatchError, match="2 recognised"):
            brain_filename_contrast("file_brain_AXT1_AXT2_201.h5")


class TestM4RawGrid:
    def test_incomplete_subject_raises(self, tmp_path: Path) -> None:
        _m4raw_tree(tmp_path, with_incomplete=True)
        with pytest.raises(IncompleteSubjectError, match="2022061004"):
            discover_m4raw_grid(tmp_path, max_subjects=4, max_contrasts=3)

    def test_allow_incomplete_drops_and_records(self, tmp_path: Path) -> None:
        _m4raw_tree(tmp_path, with_incomplete=True)
        grid = discover_m4raw_grid(tmp_path, max_subjects=4, max_contrasts=3, allow_incomplete=True)
        assert grid.subjects == _SUBJECTS
        assert grid.dropped_subjects == ("2022061004",)
        _assert_aligned(grid)

    def test_max_contrasts_two_stays_aligned(self, tmp_path: Path) -> None:
        _m4raw_tree(tmp_path)
        grid = discover_m4raw_grid(tmp_path, max_subjects=2, max_contrasts=2)
        assert grid.contrasts == ("FLAIR", "T1")
        assert grid.n_rows == 2
        _assert_aligned(grid)
        assert grid.linked and grid.row_label == "subject"

    def test_one_contrast_gives_distinct_subjects(self, tmp_path: Path) -> None:
        """C3: "one per subject" must be three anatomies, not one subject's contrasts."""
        _m4raw_tree(tmp_path)
        grid = discover_m4raw_grid(tmp_path, max_subjects=3, max_contrasts=1)
        assert [k.subject_id for k in grid.keys] == list(_SUBJECTS)
        sel = discover_m4raw_grid(tmp_path, max_subjects=3).select(per_subject=1)
        assert [k.subject_id for k in sel.keys] == list(_SUBJECTS)

    def test_contrasts_are_read(self, tmp_path: Path) -> None:
        _m4raw_tree(tmp_path, with_incomplete=True)
        grid = discover_m4raw_grid(tmp_path, contrasts=["T2", "T1"], max_subjects=4)
        assert grid.contrasts == ("T2", "T1")
        assert len(grid.subjects) == 4  # subject 4 is complete for T1/T2
        with pytest.raises(ValueError, match="not found"):
            discover_m4raw_grid(tmp_path, contrasts=["PD"])

    def test_brain_names_are_refused(self, tmp_path: Path) -> None:
        _brain(tmp_path / "file_brain_AXT1_200_6002001.h5", "AXT1")
        with pytest.raises(ValueError, match="not an M4Raw"):
            discover_m4raw_grid(tmp_path)

    def test_provenance_rows_carry_identity_and_basenames(self, tmp_path: Path) -> None:
        _m4raw_tree(tmp_path)
        rows = discover_m4raw_grid(tmp_path, max_subjects=1, max_contrasts=1).provenance()
        assert rows == [
            {
                "content_id": "2022061001_FLAIR",
                "subject_id": "2022061001",
                "contrast": "FLAIR",
                "slice_index": None,
                "file_ids": ["2022061001_FLAIR01.h5", "2022061001_FLAIR02.h5"],
                "n_reps": 2,
            }
        ]

    def test_more_contrasts_than_the_cap_raise(self, tmp_path: Path) -> None:
        _m4raw_tree(tmp_path)
        with pytest.raises(ValueError, match="max-contrasts is 1"):
            discover_m4raw_grid(tmp_path, contrasts=["T1", "T2"], max_contrasts=1)

    def test_grid_rejects_duplicate_and_misaligned_keys(self) -> None:
        a = ContentKey("s1", "T1", None, ("a",))
        b = ContentKey("s2", "T2", None, ("b",))
        with pytest.raises(ValueError, match="not unique"):
            ContrastGrid((a, a), ("s1",), ("T1",), True)
        with pytest.raises(ValueError, match="do not tile"):
            ContrastGrid((a,), ("s1",), ("T1", "T2"), True)
        with pytest.raises(ValueError, match="spans subjects"):
            ContrastGrid(
                (a, ContentKey("s2", "T2", None, ("c",))), ("s1", "s2"), ("T1", "T2"), True
            )
        with pytest.raises(ValueError, match="not in"):
            ContrastGrid((b, a), ("s2", "s1"), ("T1", "T2"), False)

    def test_content_id_is_filename_safe(self) -> None:
        key = ContentKey("200/6002001", "AXT1", 3, ("a.h5",))
        assert key.content_id == "200-6002001_AXT1_s3"
