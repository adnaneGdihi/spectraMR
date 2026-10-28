"""Tests for the tidy-frame aggregator (``reporting.aggregator``)."""

from __future__ import annotations

import json

import pytest


def test_discover_artifacts_finds_report_cases_dir(tmp_path):
    """``discover_artifacts`` exposes the recorded ``report_cases/`` directory."""

    from spectramr.infrastructure.reporting.aggregator import discover_artifacts

    # No report_cases dir → None.
    arts = discover_artifacts(tmp_path)
    assert arts.report_cases_dir is None

    # With report_cases dir → discovered path.
    (tmp_path / "report_cases").mkdir()
    arts = discover_artifacts(tmp_path)
    assert arts.report_cases_dir == tmp_path / "report_cases"


# --------------------------------------------------------------------------- #
# final_metrics.json + run_summary.json folding (2026-07-01 data-sourcing fix)
# --------------------------------------------------------------------------- #


def test_aggregate_folds_final_metrics_json(tmp_path):
    """``final_metrics.json`` (the per-arm campaign contract) becomes
    ``split='best'`` / ``split='final'`` rows; the ``_best`` suffix is stripped
    because the split already encodes it. Regression: the aggregator used to
    look only for ``final_eval.json``, which the training loop never writes."""

    from spectramr.infrastructure.reporting.aggregator import aggregate

    (tmp_path / "final_metrics.json").write_text(json.dumps({
        "schema_version": "1",
        "best": {"ssim_best": 0.86, "psnr_best": 28.4, "loss_best": 0.0},
        "final_loss": 0.12,
        "early_stopping_best_value": 9.11,
        "early_stopping_best_iteration": 100,
    }))

    df = aggregate(tmp_path, method_name="armA")
    best = df[df["split"] == "best"].set_index("metric")["value"]
    assert best["ssim"] == pytest.approx(0.86)
    assert best["psnr"] == pytest.approx(28.4)
    final = df[df["split"] == "final"].set_index("metric")["value"]
    assert final["final_loss"] == pytest.approx(0.12)
    assert final["early_stopping_best_value"] == pytest.approx(9.11)


def test_aggregate_folds_run_summary_facts(tmp_path):
    """Run-level facts (params → params_m, throughput, wall-time) land as
    ``split='run'`` rows — this is what lets the headline Pareto (cost axis
    ``params_m``) and the computational profile fire on real runs."""

    from spectramr.infrastructure.reporting.aggregator import aggregate

    (tmp_path / "run_summary.json").write_text(json.dumps({
        "run_id": "arm-20260701-abc",
        "model_params": 12_963_937,
        "iterations_per_sec": 5.767,
        "duration_sec": 120.0,
        "effective_batch": 4,
        "host": "compute-9-4",
    }))

    df = aggregate(tmp_path, method_name="armA")
    run = df[df["split"] == "run"].set_index("metric")["value"]
    assert run["params_m"] == pytest.approx(12.963937)
    assert run["iterations_per_sec"] == pytest.approx(5.767)
    assert run["duration_min"] == pytest.approx(2.0)
    assert run["effective_batch"] == pytest.approx(4.0)


def test_aggregate_accepts_run_summary_under_logs(tmp_path):
    """Historic layout kept ``run_summary.json`` under ``logs/`` — still found."""

    from spectramr.infrastructure.reporting.aggregator import discover_artifacts

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run_summary.json").write_text(json.dumps({"model_params": 1_000_000}))
    arts = discover_artifacts(tmp_path)
    assert arts.run_summary_json == logs / "run_summary.json"


def test_aggregate_survives_corrupt_summary_jsons(tmp_path, caplog):
    """A present-but-corrupt JSON logs a warning and contributes no rows —
    it must never crash the end-of-run report."""

    from spectramr.infrastructure.reporting.aggregator import aggregate

    (tmp_path / "final_metrics.json").write_text("{not json")
    (tmp_path / "run_summary.json").write_text("[1,")

    with caplog.at_level("WARNING"):
        df = aggregate(tmp_path, method_name="armA")
    assert df.empty
    assert "final_metrics.json unreadable" in caplog.text
    assert "run_summary.json unreadable" in caplog.text


def test_final_metrics_skips_non_numeric_values(tmp_path):
    """Strings / nulls / bools in final_metrics.json are skipped, not coerced."""

    from spectramr.infrastructure.reporting.aggregator import aggregate

    (tmp_path / "final_metrics.json").write_text(json.dumps({
        "best": {"ssim_best": 0.9, "note_best": "n/a", "flag_best": True},
        "final_loss": None,
    }))
    df = aggregate(tmp_path, method_name="armA")
    assert set(df[df["split"] == "best"]["metric"]) == {"ssim"}
    assert df[df["split"] == "final"].empty


def test_melt_drops_all_nan_metric_columns(tmp_path):
    """A metric column that is entirely empty (e.g. the val_loss / val_total_loss
    headers pre-seeded into the training CSV but never written) is dropped, so it
    never surfaces as a phantom all-NaN series in fig_1_2_learning_curves."""
    import pandas as pd

    from spectramr.infrastructure.reporting.aggregator import _melt_metrics_csv

    csv = tmp_path / "training_metrics.csv"
    pd.DataFrame({
        "iteration": [0, 1, 2],
        "g_total_loss": [0.5, 0.4, 0.3],
        "val_loss": [None, None, None],  # phantom header, never populated
    }).to_csv(csv, index=False)

    melted = _melt_metrics_csv(csv, "train", "r", "m")
    metrics = set(melted["metric"].unique())
    assert "g_total_loss" in metrics
    assert "val_loss" not in metrics


def test_the_rerun_test_pass_is_read_in_preference_and_named(tmp_path):
    from spectramr.infrastructure.reporting.aggregator import (
        aggregate,
        discover_artifacts,
        eval_source,
    )

    assert eval_source(tmp_path) == "none"
    (tmp_path / "final_eval.json").write_text(json.dumps({"psnr": {"s/T1/0": 20.0}}))
    assert eval_source(tmp_path) == "training"
    (tmp_path / "inference").mkdir()
    (tmp_path / "inference" / "final_eval.json").write_text(json.dumps({"psnr": {"s/T1/0": 30.0}}))
    arts = discover_artifacts(tmp_path)
    assert arts.eval_source == "inference" and arts.final_eval_json == tmp_path / "inference" / "final_eval.json"
    frame = aggregate(tmp_path)
    assert frame.loc[frame["split"] == "test", "value"].tolist() == [30.0]
