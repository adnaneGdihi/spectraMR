"""``reporting/per_slice.py``: one pass of per-slice rows, keyed by slice identity.

The reader's job is to hand a paired test the same slice from every arm, at the
checkpoint the arm's early stopping chose. Each test pins one way that goes
wrong: reading every pass, keeping a replayed row twice, losing a rank file,
keying ``3`` and ``3.0`` as different slices, averaging batch means as slices,
or letting a non-finite value through uncounted.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spectramr.infrastructure.reporting import per_slice
from spectramr.infrastructure.training.validation_rows import validation_batches_csv_for

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
    "val_nex_equivalent",
]


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(iteration: int, subject: str, z: int, psnr: float, neq: float = 2.0, **kw) -> dict:
    return {
        "iteration": iteration,
        "epoch": 0,
        "batch_index": kw.pop("batch_index", 0),
        "n_samples": kw.pop("n_samples", 1),
        "subject_id": subject,
        "contrast": "T1",
        "slice_index": z,
        "repetition_group": "",
        "file_id": f"{subject}.h5",
        "val_psnr": psnr,
        "val_nex_equivalent": neq,
        **kw,
    }


def _run(tmp_path: Path, rows: list[dict], best: int | None = None) -> Path:
    _write(tmp_path / "logs" / "validation_batches.csv", rows)
    if best is not None:
        (tmp_path / "final_metrics.json").write_text(
            json.dumps({"early_stopping_best_iteration": best})
        )
    return tmp_path


def test_rows_file_is_the_one_the_writer_uses(tmp_path: Path) -> None:
    training_csv = tmp_path / "logs" / "training_metrics.csv"
    assert validation_batches_csv_for(training_csv).name == per_slice.ROWS_FILE


def test_the_early_stopping_pass_is_read_not_every_pass(tmp_path: Path) -> None:
    rows = [_row(1000, "s0", 0, 20.0), _row(2000, "s0", 0, 30.0), _row(3000, "s0", 0, 25.0)]
    got = per_slice.load_run(_run(tmp_path, rows, best=2000))
    assert got.iteration == 2000 and got.selection == "early_stopping_best_iteration=2000"
    assert got.frame["val_psnr"].astype(float).tolist() == [30.0]


def test_an_early_stopping_iteration_between_passes_reads_the_pass_before(tmp_path: Path) -> None:
    rows = [_row(1000, "s0", 0, 20.0), _row(2000, "s0", 0, 30.0)]
    assert per_slice.load_run(_run(tmp_path, rows, best=1500)).iteration == 1000


def test_without_a_record_the_last_pass_is_read_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rows = [_row(1000, "s0", 0, 20.0), _row(2000, "s0", 0, 30.0)]
    got = per_slice.load_run(_run(tmp_path, rows))
    assert got.iteration == 2000 and got.selection.startswith("last pass")
    assert "reading the last pass" in caplog.text


def test_a_record_before_every_pass_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no validation pass at or before"):
        per_slice.load_run(_run(tmp_path, [_row(1000, "s0", 0, 20.0)], best=500))


def test_a_pass_replayed_after_resume_keeps_its_last_row(tmp_path: Path) -> None:
    rows = [_row(1000, "s0", 0, 20.0), _row(1000, "s0", 1, 21.0), _row(1000, "s0", 0, 22.0)]
    got = per_slice.load_run(_run(tmp_path, rows))
    assert got.replayed == 1 and len(got.frame) == 2
    assert got.frame.loc[("s0", "T1", "0", ""), "val_psnr"] == "22.0"


def test_rank_files_are_concatenated_in_rank_order(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _write(logs / "validation_batches.csv", [_row(1000, "s0", 0, 20.0)])
    _write(logs / "validation_batches_rank10.csv", [_row(1000, "s2", 0, 22.0)])
    _write(logs / "validation_batches_rank2.csv", [_row(1000, "s1", 0, 21.0)])
    files = per_slice.find_rows_files(tmp_path)
    assert [p.name for p in files] == [
        "validation_batches.csv",
        "validation_batches_rank2.csv",
        "validation_batches_rank10.csv",
    ]
    got = per_slice.read_pass(files)
    assert sorted(got.frame.index.get_level_values("subject_id")) == ["s0", "s1", "s2"]


def test_slice_index_spelled_as_float_is_the_same_slice(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    _write(logs / "validation_batches.csv", [_row(1000, "s0", 3, 20.0)])
    _write(logs / "validation_batches_rank1.csv", [_row(1000, "s0", "3.0", 21.0)])  # type: ignore[arg-type]
    got = per_slice.read_pass(per_slice.find_rows_files(tmp_path))
    assert len(got.frame) == 1 and got.replayed == 1


def test_a_batch_mean_row_is_refused(tmp_path: Path) -> None:
    rows = [_row(1000, "s0|s1", 0, 20.0, n_samples=2)]
    with pytest.raises(ValueError, match="batch_size: 1"):
        per_slice.load_run(_run(tmp_path, rows))


def test_rows_without_a_subject_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="carry no subject_id"):
        per_slice.load_run(_run(tmp_path, [_row(1000, "", 0, 20.0)]))


def test_a_run_without_rows_is_none(tmp_path: Path) -> None:
    assert per_slice.load_run(tmp_path) is None


def test_non_finite_values_are_dropped_and_counted(tmp_path: Path) -> None:
    rows = [
        _row(1000, "s0", 0, 20.0),
        _row(1000, "s0", 1, math.nan),
        _row(1000, "s1", 0, math.inf),
        _row(1000, "s1", 1, 23.0),
    ]
    got = per_slice.load_run(_run(tmp_path, rows))
    values, dropped = per_slice.slice_values(got, "val_psnr")
    assert dropped == 2 and sorted(values.tolist()) == [20.0, 23.0]


def test_n_eq_is_averaged_as_its_reciprocal_over_slices_then_subjects(tmp_path: Path) -> None:
    # s0: N_eq 2 and 4 -> mean 1/N_eq 0.375; s1: N_eq inf (1/N_eq = 0, kept) and
    # 0 (no reciprocal, dropped) -> 0. Cohort 1/N_eq = 0.1875, N_eq = 5.333.
    rows = [
        _row(1000, "s0", 0, 20.0, neq=2.0),
        _row(1000, "s0", 1, 20.0, neq=4.0),
        _row(1000, "s1", 0, 20.0, neq=math.inf),
        _row(1000, "s1", 1, 20.0, neq=0.0),
    ]
    got = per_slice.load_run(_run(tmp_path, rows))
    values, dropped = per_slice.slice_values(got, "val_nex_equivalent")
    assert dropped == 1
    subjects = per_slice.subject_means(values)
    assert subjects.to_dict() == {"s0": pytest.approx(0.375), "s1": 0.0}
    summary = per_slice.summarise(subjects, "val_nex_equivalent")
    assert summary["mean"] == pytest.approx(1 / 0.1875) and summary["n"] == 2
    # The arithmetic mean of per-slice N_eq is not this number.
    assert summary["mean"] != pytest.approx(np.mean([2.0, 4.0]))


def test_summarise_counts_subjects_not_slices() -> None:
    values = pd.Series([1.0, 3.0], index=pd.Index(["s0", "s1"], name="subject_id"))
    got = per_slice.summarise(values, "val_psnr")
    assert got == {"mean": 2.0, "std": 1.0, "median": 2.0, "n": 2, "unit": "subjects"}
