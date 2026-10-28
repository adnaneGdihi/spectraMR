"""``reporting/per_subject_stats.py``: two arms paired on slice identity.

The validation loaders shuffle, so ``batch_index`` names a different slice in
each arm. These tests build two arms whose per-slice difference is known, write
them in independent random orders, and check that the slice-keyed pairing
recovers the difference while a planted ``batch_index`` join does not.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectramr.infrastructure.reporting import per_slice
from spectramr.infrastructure.reporting.per_subject_stats import (
    PairedSubjects,
    compare_subjects,
    pair_on_slices,
)

SUBJECTS, SLICES = 6, 8
OFFSET = 0.5
HEADER = [
    "iteration",
    "epoch",
    "batch_index",
    "n_samples",
    "subject_id",
    "contrast",
    "slice_index",
    "repetition_group",
    "file_id",
    "val_psnr",
]


def _arm(path: Path, rng: np.random.Generator, offset: float, drop: set | None = None) -> Path:
    """One arm's rows: a subject-and-slice signal plus ``offset`` plus small noise,
    written in a random order with ``batch_index`` counting rows as written."""
    keys = [(s, z) for s in range(SUBJECTS) for z in range(SLICES) if (s, z) not in (drop or set())]
    order = rng.permutation(len(keys))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for batch_index, j in enumerate(order):
            s, z = keys[j]
            value = 25.0 + 2.0 * s + 1.5 * z + offset + rng.normal(0.0, 0.05)
            writer.writerow([1000, 0, batch_index, 1, f"s{s}", "T1", z, "rep0", "f", value])
    return path


@pytest.fixture
def arms(tmp_path: Path) -> tuple[per_slice.SlicePass, per_slice.SlicePass]:
    rng = np.random.default_rng(7)
    a = _arm(tmp_path / "a" / "logs" / "validation_batches.csv", rng, OFFSET)
    b = _arm(tmp_path / "b" / "logs" / "validation_batches.csv", rng, 0.0)
    return per_slice.read_pass([a]), per_slice.read_pass([b])


def test_shuffled_arms_are_paired_on_slice_identity(arms) -> None:
    a, b = arms
    paired = pair_on_slices(a, b, "val_psnr", "val_psnr")
    assert paired.n_subjects == SUBJECTS and paired.n_slices == SUBJECTS * SLICES
    diff = paired.table["a"] - paired.table["b"]
    assert diff.to_numpy() == pytest.approx(np.full(SUBJECTS, OFFSET), abs=0.05)
    stats = compare_subjects(paired, n_bootstrap=500)
    assert stats["paired_ttest"]["mean_diff"] == pytest.approx(OFFSET, abs=0.03)
    assert stats["paired_ttest"]["p_value"] < 1e-4
    ci = stats["bootstrap_ci"]
    assert ci["ci_lower"] <= ci["observed"] <= ci["ci_upper"]
    assert ci["ci_lower"] == pytest.approx(OFFSET, abs=0.05)
    assert ci["ci_upper"] == pytest.approx(OFFSET, abs=0.05)


def test_planted_batch_index_pairing_gives_a_different_wrong_answer(tmp_path: Path, arms) -> None:
    """The violation this module exists to prevent: join on batch position."""
    a, b = arms
    right_pairs = pair_on_slices(a, b, "val_psnr", "val_psnr")
    right = compare_subjects(right_pairs, n_bootstrap=200)

    # Everything as in pair_on_slices except the key: rows joined on batch_index,
    # the subject taken from arm a, then reduced to subject means.
    frame_a, frame_b = (pd.read_csv(rows.files[0]) for rows in (a, b))
    joined = frame_a.merge(frame_b, on="batch_index", suffixes=("_a", "_b"))
    table = (
        joined.rename(columns={"val_psnr_a": "a", "val_psnr_b": "b", "subject_id_a": "subject_id"})
        .groupby("subject_id")[["a", "b"]]
        .mean()
    )
    planted = PairedSubjects(
        table=table,
        n_slices=len(joined),
        nonfinite_dropped={"a": 0, "b": 0},
        unpaired_slices={"a": 0, "b": 0},
        scale="metric",
    )
    wrong = compare_subjects(planted, n_bootstrap=200)

    right_diff = (right_pairs.table["a"] - right_pairs.table["b"]).to_numpy()
    wrong_diff = (table["a"] - table["b"]).to_numpy()
    assert right_diff == pytest.approx(np.full(SUBJECTS, OFFSET), abs=0.05)
    # Each subject is compared with a random mix of the other arm's slices, so the
    # per-subject differences track the subject's own level, not the offset.
    assert np.max(np.abs(wrong_diff - OFFSET)) > 2.0
    assert wrong["paired_ttest"]["p_value"] > 1e-3 > right["paired_ttest"]["p_value"]


def test_arms_whose_identities_never_match_are_refused(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    a = per_slice.read_pass([_arm(tmp_path / "a.csv", rng, OFFSET)])
    b_path = _arm(tmp_path / "b.csv", rng, 0.0)
    b_path.write_text(b_path.read_text().replace(",rep0,", ",rep1,"))
    b = per_slice.read_pass([b_path])
    with pytest.raises(ValueError, match=r"no slice of .* matches"):
        pair_on_slices(a, b, "val_psnr", "val_psnr")


def test_slices_one_arm_did_not_score_are_counted_not_paired(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    a = per_slice.read_pass([_arm(tmp_path / "a.csv", rng, OFFSET)])
    b = per_slice.read_pass([_arm(tmp_path / "b.csv", rng, 0.0, drop={(0, 0), (1, 3)})])
    paired = pair_on_slices(a, b, "val_psnr", "val_psnr")
    assert paired.unpaired_slices == {"a": 2, "b": 0}
    assert paired.describe()["n_slices_paired"] == SUBJECTS * SLICES - 2


def test_fewer_than_two_subjects_is_refused(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    only_s0 = {(s, z) for s in range(1, SUBJECTS) for z in range(SLICES)}
    a = per_slice.read_pass([_arm(tmp_path / "a.csv", rng, OFFSET, drop=only_s0)])
    b = per_slice.read_pass([_arm(tmp_path / "b.csv", rng, 0.0, drop=only_s0)])
    with pytest.raises(ValueError, match="paired test needs two"):
        compare_subjects(pair_on_slices(a, b, "val_psnr", "val_psnr"))
