"""``CampaignEvaluator`` reads per-slice rows, and refuses pass rows as samples (#2478).

Before this, an arm with no per-sample NPZ had each metric's whole column of
``validation_metrics.csv`` (one row per validation pass) averaged, bootstrapped
and paired position by position with another arm's passes. These tests pin the
replacement: per-slice rows paired on slice identity and reduced to subjects,
and a refusal when an arm has nothing but pass rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from spectramr.config.schemas.campaign import EvaluationConfigSchema
from spectramr.infrastructure.orchestration.campaign_evaluator import CampaignEvaluator
from spectramr.infrastructure.orchestration.campaign_state import CampaignState, ExperimentStatus

SUBJECTS, SLICES = 5, 6
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


def _pass_rows(run: Path, psnrs: list[float]) -> Path:
    path = run / "logs" / "validation_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "val_psnr"])
        writer.writerows([[1000 * (i + 1), p] for i, p in enumerate(psnrs)])
    return path


def _slice_rows(run: Path, offset: float, seed: int, best: int = 2000) -> None:
    """Two passes of per-slice rows in a shuffled order, early stopping at ``best``."""
    rng = np.random.default_rng(seed)
    keys = [(s, z) for s in range(SUBJECTS) for z in range(SLICES)]
    path = run / "logs" / "validation_batches.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(HEADER)
        for iteration, shift in ((1000, -5.0), (2000, 0.0)):
            for batch_index, j in enumerate(rng.permutation(len(keys))):
                s, z = keys[j]
                value = 25.0 + 2.0 * s + z + offset + shift + rng.normal(0.0, 0.05)
                writer.writerow([iteration, 0, batch_index, 1, f"s{s}", "T1", z, "", "f", value])
    (run / "final_metrics.json").write_text(json.dumps({"early_stopping_best_iteration": best}))


def _state(tmp_path: Path, arms: dict[str, Path]) -> CampaignState:
    state = CampaignState(campaign_name="c", campaign_dir=str(tmp_path))
    for i, (name, run) in enumerate(arms.items()):
        csv_path = run / "logs" / "validation_metrics.csv"
        state.experiments.append(
            ExperimentStatus(
                name=name,
                config_path=f"{name}.yaml",
                role="baseline" if i == 0 else "variant",
                status="completed",
                checkpoint_path=str(run / "best.pt"),
                output_dir=str(run),
                metrics_csv_path=str(csv_path) if csv_path.exists() else None,
            )
        )
    return state


def _evaluator() -> CampaignEvaluator:
    return CampaignEvaluator(
        EvaluationConfigSchema(metrics=["psnr"], n_bootstrap=200, per_sample_inference=False)
    )


def test_only_pass_rows_is_refused_naming_the_issue(tmp_path: Path) -> None:
    """Planted violation: the shape #2478 reported, two arms with pass rows only."""
    a, b = tmp_path / "a", tmp_path / "b"
    _pass_rows(a, [20.0, 25.0, 26.0])
    _pass_rows(b, [21.0, 24.0])
    with pytest.raises(ValueError, match=r"passes are not samples.*#2478"):
        _evaluator().evaluate(_state(tmp_path, {"a": a, "b": b}))


def test_the_refusal_reaches_the_campaign_evaluate_command(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The production path: ``spectramr campaign evaluate`` exits non-zero and says why."""
    from argparse import Namespace

    from spectramr.cli.app import campaign_evaluate

    a, b = tmp_path / "a", tmp_path / "b"
    _pass_rows(a, [20.0, 25.0, 26.0])
    _pass_rows(b, [21.0, 24.0])
    _state(tmp_path, {"a": a, "b": b}).save()
    assert campaign_evaluate(Namespace(campaign_dir=str(tmp_path), latex=False)) == 1
    assert "#2478" in caplog.text and "passes are not samples" in caplog.text


def test_per_slice_rows_are_paired_by_subject_at_the_selected_pass(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for run in (a, b):
        _pass_rows(run, [20.0, 25.0])
    _slice_rows(a, 0.0, seed=1)
    _slice_rows(b, 0.4, seed=2)
    report = _evaluator().evaluate(_state(tmp_path, {"a": a, "b": b}))

    board = report.leaderboard.set_index("experiment")
    assert board.loc["a", "n"] == SUBJECTS and board.loc["a", "unit"] == "subjects"
    assert board.loc["a", "iteration"] == 2000
    assert board.loc["a", "nonfinite_dropped"] == 0
    # Mean over subjects of the 2000-iteration pass, not over passes or the 1000 pass.
    expected_a = 25.0 + 2.0 * np.mean(range(SUBJECTS)) + np.mean(range(SLICES))
    assert board.loc["a", "mean"] == pytest.approx(expected_a, abs=0.05)

    (pair,) = report.pairwise_results
    assert pair.paired_ttest["mean_diff"] == pytest.approx(-0.4, abs=0.05)
    assert pair.paired_ttest["p_value"] < 1e-3
    assert pair.provenance["n_subjects"] == SUBJECTS
    assert pair.provenance["n_slices_paired"] == SUBJECTS * SLICES
    assert report.significance_matrix["psnr"].loc["a", "b"] < 1e-3


def test_an_arm_with_only_pass_rows_beside_row_arms_is_refused(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _pass_rows(a, [20.0, 25.0])
    _slice_rows(a, 0.0, seed=1)
    _pass_rows(b, [21.0, 24.0])
    with pytest.raises(ValueError, match=r"\['b'\] have only validation_metrics.csv"):
        _evaluator().evaluate(_state(tmp_path, {"a": a, "b": b}))


def test_discovery_records_the_rank_files_the_evaluator_reads(tmp_path: Path) -> None:
    """The production path: ``_discover_results`` finds the rows, the evaluator uses them."""
    from spectramr.infrastructure.orchestration.campaign_orchestrator import (
        CampaignOrchestrator,
    )

    a, b = tmp_path / "a", tmp_path / "b"
    for seed, run in enumerate((a, b)):
        _pass_rows(run, [20.0, 25.0])
        _slice_rows(run, 0.3 * seed, seed=seed)
    (b / "logs" / "validation_batches.csv").rename(b / "logs" / "validation_batches_rank1.csv")
    state = CampaignState(campaign_name="c", campaign_dir=str(tmp_path))
    orchestrator = CampaignOrchestrator.__new__(CampaignOrchestrator)
    for name, run in (("a", a), ("b", b)):
        exp = ExperimentStatus(
            name=name,
            config_path=f"{name}.yaml",
            status="completed",
            checkpoint_path=str(run / "best.pt"),
            output_dir=str(run),
        )
        orchestrator._discover_results(exp)
        state.experiments.append(exp)
    assert [Path(p).name for p in state.experiments[1].slice_rows_paths] == [
        "validation_batches_rank1.csv"
    ]
    restored = ExperimentStatus.model_validate(state.experiments[1].model_dump())
    assert restored.slice_rows_paths == state.experiments[1].slice_rows_paths
    (pair,) = _evaluator().evaluate(state).pairwise_results
    assert pair.provenance["n_subjects"] == SUBJECTS


def test_npz_arms_and_row_arms_are_not_mixed(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _slice_rows(a, 0.0, seed=1)
    b.mkdir()
    npz = b / "per_sample_metrics.npz"
    np.savez(npz, psnr=np.linspace(20.0, 30.0, 10))
    state = _state(tmp_path, {"a": a, "b": b})
    state.experiments[1].per_sample_metrics_path = str(npz)
    with pytest.raises(ValueError, match="cannot be paired"):
        _evaluator().evaluate(state)
