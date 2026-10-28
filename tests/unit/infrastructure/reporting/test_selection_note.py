"""An arm whose best checkpoint is its first validation pass is marked in the comparison."""

from __future__ import annotations

import json
from pathlib import Path

from spectramr.infrastructure.reporting.selection_note import first_pass_selection_notes


def _run(root: Path, name: str, best: int | None, steps=(2000, 4000, 6000)) -> Path:
    logs = root / name / "logs"
    logs.mkdir(parents=True)
    (root / name / "final_metrics.json").write_text(
        json.dumps({"early_stopping_best_iteration": best})
    )
    (logs / "validation_metrics.csv").write_text(
        "iteration,val_psnr\n" + "".join(f"{s},1.0\n" for s in steps)
    )
    return root / name


def test_a_run_selected_at_its_first_pass_is_marked(tmp_path):
    dirs = {"control": _run(tmp_path, "control", 2000), "ei": _run(tmp_path, "ei", 6000)}
    notes = first_pass_selection_notes(dirs)
    assert list(notes) == ["control"] and "step 2000" in notes["control"]


def test_a_run_selected_later_is_left_alone(tmp_path):
    assert first_pass_selection_notes({"ei": _run(tmp_path, "ei", 4000)}) == {}


def test_a_run_with_no_selection_record_is_left_alone(tmp_path):
    run = _run(tmp_path, "x", None)
    assert first_pass_selection_notes({"x": run}) == {}


def test_a_run_without_final_metrics_is_left_alone(tmp_path):
    (tmp_path / "bare").mkdir()
    assert first_pass_selection_notes({"bare": tmp_path / "bare"}) == {}
