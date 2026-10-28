"""Subject-grouped splitting (cohort review 2026-09-02, T0.2).

The planted violation comes first: the flat ``split_index`` DOES put one
subject's files on both sides of the boundary, which is the leak the grouped
variants exist to prevent. A test that only showed the grouped split working
would not show that the ungrouped one fails.
"""

from __future__ import annotations

import pytest

from spectramr.data.split_utils import (
    split_index,
    split_index_grouped,
    split_index_three_way_grouped,
    subject_id_from_name,
)


def _key(name: str) -> str:
    return subject_id_from_name(name) or name


def _subject_sets(items: list[str]) -> set[str]:
    return {_key(i) for i in items}


# --- subject_id_from_name -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sub-01_T1w.nii.gz", "sub-01"),
        ("sub-01/anat/sub-01_T2w.nii.gz", "sub-01"),
        ("/data/bids/sub-A1b2/ses-02/anat/sub-A1b2_ses-02_FLAIR.nii", "sub-A1b2"),
        ("2022_T101.h5", None),
        ("file1000000.h5", None),
        ("resub-01_T1w.nii.gz", None),
    ],
)
def test_subject_id_from_name(name: str, expected: str | None) -> None:
    assert subject_id_from_name(name) == expected


# --- the planted violation: the flat split straddles a subject -----------------


def test_flat_split_index_straddles_a_subject() -> None:
    """Three subjects x two contrasts, 1/3 held out: the file-level tail split
    takes two files, i.e. one whole subject -- but with an odd hold-out it takes
    half a subject. Both shapes existed in the corpus; this pins the leaking one."""
    files = [f"sub-{s:02d}_{c}.nii.gz" for s in (1, 2, 3) for c in ("T1w", "T2w")]
    train, val = split_index(files, 0.5)  # 3 of 6 files -> sub-02 straddles
    assert _subject_sets(train) & _subject_sets(val) == {"sub-02"}


def test_grouped_split_never_straddles_a_subject() -> None:
    files = [f"sub-{s:02d}_{c}.nii.gz" for s in (1, 2, 3) for c in ("T1w", "T2w")]
    train, val = split_index_grouped(files, 0.5, key=_key)
    assert not (_subject_sets(train) & _subject_sets(val))
    assert sorted(train + val) == sorted(files)
    # the fraction applies to SUBJECTS: round(3 * 0.5) = 2 held out
    assert _subject_sets(val) == {"sub-02", "sub-03"}


def test_grouped_split_keeps_item_order_within_each_side() -> None:
    files = ["sub-01_a", "sub-02_a", "sub-01_b", "sub-02_b", "sub-03_a"]
    train, val = split_index_grouped(files, 0.34, key=_key)
    assert train == ["sub-01_a", "sub-01_b", "sub-02_a", "sub-02_b"]
    assert val == ["sub-03_a"]


def test_unlabeled_files_are_their_own_groups_so_the_split_is_the_old_one() -> None:
    """An M4Raw-style corpus carries no ``sub-`` label: grouping degrades to the
    file-level split byte for byte, so no existing arm changes behaviour."""
    files = [f"2022_T1{i:02d}.h5" for i in range(10)]
    assert split_index_grouped(files, 0.2, key=_key) == split_index(files, 0.2)


def test_grouped_split_train_only_escape_hatch() -> None:
    files = ["sub-01_a", "sub-02_a"]
    assert split_index_grouped(files, 0.0, key=_key) == (files, [])


def test_grouped_split_single_subject_with_holdout_raises() -> None:
    """One subject cannot be split without leaking it; mirrors ``split_index``."""
    with pytest.raises(ValueError, match="single file"):
        split_index_grouped(["sub-01_a", "sub-01_b"], 0.5, key=_key)


# --- three-way ------------------------------------------------------------------


def test_three_way_grouped_is_pairwise_disjoint_by_subject_and_covers_everything() -> None:
    files = [f"sub-{s:02d}_{c}" for s in range(1, 7) for c in ("T1w", "T2w", "FLAIR")]
    train, val, test = split_index_three_way_grouped(files, 0.2, 0.2, key=_key)
    sides = [_subject_sets(x) for x in (train, val, test)]
    assert all(not (a & b) for i, a in enumerate(sides) for b in sides[i + 1 :])
    assert sorted(train + val + test) == sorted(files)
    assert len(_subject_sets(test)) == 1 and len(_subject_sets(val)) == 1


def test_three_way_grouped_without_test_split_matches_two_way() -> None:
    files = [f"sub-{s:02d}_{c}" for s in range(1, 5) for c in ("a", "b")]
    train, val, test = split_index_three_way_grouped(files, 0.25, 0.0, key=_key)
    assert test == []
    assert (train, val) == split_index_grouped(files, 0.25, key=_key)


def test_record_split_key_prefers_subject_then_file():
    from spectramr.data.split_utils import record_split_key

    assert record_split_key({"patient_id": "p1", "primary_path": "a.h5"}) == "p1"
    assert record_split_key({"subject_id": "s1", "patient_id": "p1"}) == "s1"
    assert record_split_key({"primary_path": "a.h5"}) == "a.h5"
    a, b = {}, {}
    assert record_split_key(a) != record_split_key(b)


def test_record_split_key_reads_paths_and_attributes():
    from pathlib import Path
    from types import SimpleNamespace

    from spectramr.data.split_utils import record_split_key

    # A BIDS label in the file name groups the records that carry no subject id.
    assert record_split_key({"input": "/d/sub-07/anat/sub-07_T1w.nii.gz"}) == "sub-07"
    assert record_split_key("/d/sub-03_T2w.nii.gz") == "sub-03"
    assert record_split_key(Path("/d/vol_12.npy")) == "/d/vol_12.npy"
    assert record_split_key(SimpleNamespace(subject_id="p9", input_path="x")) == "p9"
