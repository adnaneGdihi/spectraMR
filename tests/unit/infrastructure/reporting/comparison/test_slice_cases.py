"""Per-slice rows stand in for per-call rows in the cross-run comparison (#2500).

The n2n arms write ``validation_batches.csv`` (one row per slice) and no
``per_call_metrics.csv``; the comparison read only the latter, so their figure
had no error bars. Pinned with the violation that turns each red: pairing by
batch position, which the shuffled loaders make wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.comparison.arms import load_arms, resolve_runs
from spectramr.infrastructure.reporting.comparison.slice_cases import slice_case_rows

from ._runs import make_run

_SUBJECTS, _SLICES, _PASSES = ("s1", "s2", "s3"), 4, (100, 200, 300)


def _psnr(subject: str, slice_index: int, shift: float) -> float:
    return 30.0 + shift + int(subject[1:]) + 0.5 * slice_index


def _write_rows(run: Path, *, shift: float, seed: int) -> None:
    """One row per slice per pass, in a different shuffled order per arm."""
    rows = [
        {
            "iteration": it,
            "epoch": 0,
            "batch_index": 0,
            "n_samples": 1,
            "subject_id": subj,
            "contrast": "T1",
            "slice_index": k,
            "repetition_group": f"{subj}_T1",
            "file_id": f"{subj}_T101",
            "val_psnr": _psnr(subj, k, shift) + it / 1000,
        }
        for it in _PASSES
        for subj in _SUBJECTS
        for k in range(_SLICES)
    ]
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=seed)
    frame["batch_index"] = frame.groupby("iteration").cumcount()
    frame.to_csv(run / "logs" / "validation_batches.csv", index=False)


def _run(root: Path, name: str, *, shift: float, seed: int, selected: int = 200) -> Path:
    run = make_run(root, name, shift=shift, selected=selected, per_call=False, pngs=False)
    _write_rows(run, shift=shift, seed=seed)
    (run / "final_metrics.json").write_text(json.dumps({"early_stopping_best_iteration": selected}))
    return run


def test_one_case_per_slice_at_the_selected_pass(tmp_path: Path) -> None:
    run = _run(tmp_path, "exp_a_run", shift=0.0, seed=0)
    rows = slice_case_rows(run, step=200, run_id="a", method="a")
    psnr = rows[rows["base_metric"] == "psnr"]
    assert len(psnr) == len(_SUBJECTS) * _SLICES
    assert set(psnr["step"]) == {200}
    one = psnr[psnr["case_id"] == "s2|T1|3|s2_T1"]["value"].item()
    assert one == _psnr("s2", 3, 0.0) + 0.2


def test_load_arms_pairs_shuffled_arms_on_slice_identity(tmp_path: Path) -> None:
    _run(tmp_path, "exp_a_run", shift=0.0, seed=0)
    _run(tmp_path, "exp_b_run", shift=1.0, seed=1)
    arms = load_arms(resolve_runs([tmp_path]), with_images=False)
    psnr = arms.per_case[arms.per_case["base_metric"] == "psnr"]
    a = psnr[psnr["method"] == "a"].set_index("pair_key")["value"]
    b = psnr[psnr["method"] == "b"].set_index("pair_key")["value"]
    assert set(a.index) == set(b.index) and len(a) == len(_SUBJECTS) * _SLICES
    np.testing.assert_allclose((b - a.reindex(b.index)).to_numpy(), 1.0)


def test_planted_pairing_by_batch_index_is_wrong(tmp_path: Path) -> None:
    """The shape this module refuses: batch i of one arm is not batch i of the other."""
    for name, seed in (("exp_a_run", 0), ("exp_b_run", 1)):
        _run(tmp_path, name, shift=0.0 if seed == 0 else 1.0, seed=seed)
    frames = {
        name: pd.read_csv(tmp_path / name / "logs" / "validation_batches.csv").query(
            "iteration == 200"
        )
        for name in ("exp_a_run", "exp_b_run")
    }
    by_batch = (
        frames["exp_b_run"].set_index("batch_index")["val_psnr"]
        - frames["exp_a_run"].set_index("batch_index")["val_psnr"]
    )
    assert not np.allclose(by_batch.to_numpy(), 1.0)


def test_a_run_with_per_call_rows_keeps_them(tmp_path: Path) -> None:
    make_run(tmp_path, "exp_a_run")
    make_run(tmp_path, "exp_b_run")
    arms = load_arms(resolve_runs([tmp_path]), with_images=False)
    assert not arms.per_case["pair_key"].str.startswith("slice#").any()
