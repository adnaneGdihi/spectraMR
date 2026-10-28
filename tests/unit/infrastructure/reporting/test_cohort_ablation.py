"""Unit tests for cohort task-ablation figure assembly."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import pandas as pd

from spectramr.infrastructure.reporting import cohort_ablation as ca


def _make_arm(
    root: Path,
    arm: str,
    task: str,
    csv_rows: list[dict],
    *,
    baseline: str | None = None,
    primary: str = "ssim",
    with_csv: bool = True,
    tag_arm: str | None = None,
) -> Path:
    """Materialise a fake arm result dir (resolved_config.json + val CSV)."""
    d = root / arm
    (d / "logs").mkdir(parents=True)
    cfg = {
        "metadata": {
            "tags": {
                "task": task,
                "arm": tag_arm if tag_arm is not None else ca._family_of(arm),
            },
            "baseline": baseline,
            "primary_metric": primary,
        }
    }
    (d / "resolved_config.json").write_text(json.dumps(cfg))
    if with_csv:
        pd.DataFrame(csv_rows).to_csv(
            d / "logs" / "validation_metrics.csv", index=False
        )
    return d


def test_family_parsing() -> None:
    assert ca._family_of("mrixfields_b16_field_fno") == "b16"
    assert ca._family_of("mrixfields_b110_doob_bridge") == "b110"
    assert ca._family_of("mrixfields_b21_ulf_map") == "b21"


def test_family_parsing_non_bnn_falls_back_to_tag_arm() -> None:
    # Paradigm-named arms have no bNN token, so a method/ablation pair must
    # collapse onto one family via tags.arm (else each becomes its own
    # one-arm group and the figure's family-header labels collide).
    assert ca._family_of("mrixfields_bloch_synth_7t", "bloch_synth") == "bloch_synth"
    assert (
        ca._family_of("mrixfields_bloch_synth_ablate_dispersion", "bloch_synth_ablate")
        == "bloch_synth"
    )
    assert (
        ca._family_of("mrixfields_field_cocycle_ablate_cocycle", "field_cocycle_ablate")
        == "field_cocycle"
    )
    # No tags.arm at all -> final fallback is still the raw arm name.
    assert ca._family_of("mrixfields_ref_restormer") == "mrixfields_ref_restormer"


def test_discover_derives_task_family_role(tmp_path) -> None:
    _make_arm(
        tmp_path,
        "mrixfields_b16_field_fno",
        "3",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.3,
                "val_psnr": 18.0,
                "val_lpips": 0.6,
            }
        ],
    )
    _make_arm(
        tmp_path,
        "mrixfields_b16_ablate_spectral",
        "3",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.2,
                "val_psnr": 15.0,
                "val_lpips": 0.7,
            }
        ],
        baseline="mrixfields_b16_field_fno",
    )
    by = {a.arm: a for a in ca.discover_cohort_arms(tmp_path)}

    method = by["mrixfields_b16_field_fno"]
    assert method.role == "method"
    assert method.family == "b16"
    assert method.task == "3"

    ablate = by["mrixfields_b16_ablate_spectral"]
    assert ablate.role == "ablation"
    assert ablate.family == "b16"
    assert ablate.baseline == "mrixfields_b16_field_fno"


def test_discover_groups_non_bnn_siblings_by_tag_arm(tmp_path) -> None:
    # mrixfields_bloch_synth_7t (method) / mrixfields_bloch_synth_ablate_dispersion
    # (ablation) share no bNN token, so without the tags.arm fallback each
    # becomes its own one-arm family -- adjacent single-bar groups whose
    # header labels collide in the rendered figure.
    _make_arm(
        tmp_path,
        "mrixfields_bloch_synth_7t",
        "1",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.6,
                "val_psnr": 16.0,
                "val_lpips": 0.3,
            }
        ],
        tag_arm="bloch_synth",
    )
    _make_arm(
        tmp_path,
        "mrixfields_bloch_synth_ablate_dispersion",
        "1",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.6,
                "val_psnr": 16.0,
                "val_lpips": 0.3,
            }
        ],
        baseline="mrixfields_bloch_synth_7t",
        tag_arm="bloch_synth_ablate",
    )
    by = {a.arm: a for a in ca.discover_cohort_arms(tmp_path)}
    assert by["mrixfields_bloch_synth_7t"].family == "bloch_synth"
    assert by["mrixfields_bloch_synth_ablate_dispersion"].family == "bloch_synth"


def test_best_val_metrics_reduces_best_over_iterations(tmp_path) -> None:
    d = _make_arm(
        tmp_path,
        "mrixfields_b16_field_fno",
        "3",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.3,
                "val_psnr": 18.0,
                "val_lpips": 0.6,
            },
            {
                "iteration": 200,
                "epoch": 4,
                "val_ssim": 0.7,
                "val_psnr": 25.0,
                "val_lpips": 0.4,
            },
        ],
    )
    best = ca.best_val_metrics(d, ("ssim", "psnr", "lpips"))
    assert best["ssim"] == 0.7  # max (higher is better)
    assert best["psnr"] == 25.0  # max
    assert best["lpips"] == 0.4  # min (lower is better)


def test_build_task_frame_partitions_by_task(tmp_path) -> None:
    _make_arm(
        tmp_path,
        "mrixfields_b16_field_fno",
        "3",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.7,
                "val_psnr": 25.0,
                "val_lpips": 0.4,
            }
        ],
    )
    _make_arm(
        tmp_path,
        "mrixfields_b21_ulf_map",
        "2",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.5,
                "val_psnr": 20.0,
                "val_lpips": 0.5,
            }
        ],
    )
    arms = ca.discover_cohort_arms(tmp_path)

    frame3 = ca.build_task_frame(arms, "3", ("ssim", "psnr", "lpips"))
    assert set(frame3["arm"].unique()) == {"mrixfields_b16_field_fno"}
    assert set(frame3["metric"].unique()) == {"ssim", "psnr", "lpips"}
    assert set(frame3.columns) == {"family", "role", "arm", "metric", "value"}

    frame2 = ca.build_task_frame(arms, "2", ("ssim", "psnr", "lpips"))
    assert set(frame2["arm"].unique()) == {"mrixfields_b21_ulf_map"}


def test_arm_without_val_csv_is_skipped(tmp_path) -> None:
    # A calibration-only arm (no validation_metrics.csv) is discovered but
    # contributes no rows — mirrors b17_dice_risk_calibration.
    _make_arm(tmp_path, "mrixfields_b17_dice_risk_calibration", "1", [], with_csv=False)
    arms = ca.discover_cohort_arms(tmp_path)
    assert any(a.arm == "mrixfields_b17_dice_risk_calibration" for a in arms)
    assert ca.best_val_metrics(tmp_path / "mrixfields_b17_dice_risk_calibration") == {}
    frame = ca.build_task_frame(arms, "1", ("ssim", "psnr", "lpips"))
    assert frame.empty


def test_partial_metrics_arm(tmp_path) -> None:
    # b29-style arm: only val_psnr present (no ssim/lpips) -> only psnr row.
    _make_arm(
        tmp_path,
        "mrixfields_b29_heteroscedastic_ulf",
        "2",
        [{"iteration": 100, "epoch": 2, "val_psnr": -9.0}],
    )
    arms = ca.discover_cohort_arms(tmp_path)
    frame = ca.build_task_frame(arms, "2", ("ssim", "psnr", "lpips"))
    assert set(frame["metric"].unique()) == {"psnr"}
    assert float(frame["value"].iloc[0]) == -9.0


def test_generate_task_ablation_figures_end_to_end(tmp_path) -> None:
    _make_arm(
        tmp_path,
        "mrixfields_b16_field_fno",
        "3",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.7,
                "val_psnr": 25.0,
                "val_lpips": 0.4,
            }
        ],
    )
    _make_arm(
        tmp_path,
        "mrixfields_b16_ablate_spectral",
        "3",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.5,
                "val_psnr": 20.0,
                "val_lpips": 0.6,
            }
        ],
        baseline="mrixfields_b16_field_fno",
    )
    out_dir = tmp_path / "figs"
    written = ca.generate_task_ablation_figures(tmp_path, out_dir)

    # Task 3 has data -> a figure; tasks 1 & 2 have no arms -> None.
    assert written["3"] is not None and written["3"].exists()
    assert (out_dir / "task3_ablation_bars.png").exists()
    assert written["1"] is None
    assert written["2"] is None


def _baseline_result(
    method: str, task: int, source: float, ssim: float, lpips: float
) -> dict:
    """A ``baseline_results.json``-shaped record (asdict(BaselineResult))."""
    return {
        "name": f"task{task}_{source}T__{method}",
        "task": task,
        "method": method,
        "source_field": source,
        "target_field": 7.0,
        "contrast": "T1w",
        "metrics_mean": {"nrmse_l2": 0.2, "ssim": ssim, "lpips_alex": lpips},
        "metrics_std": {},
        "n_subjects": 3,
        "provenance": {"metric_impl": "framework_registry"},
    }


def test_merge_baseline_results_adds_role_baseline_rows() -> None:
    # Two cut units (different source fields) -> aggregated to one bar per metric.
    results = [
        _baseline_result("cut", 1, 0.1, ssim=0.60, lpips=0.30),
        _baseline_result("cut", 1, 1.5, ssim=0.80, lpips=0.50),
        _baseline_result("cyclegan", 1, 0.1, ssim=0.55, lpips=0.40),
        _baseline_result(
            "cut", 2, 0.1, ssim=0.99, lpips=0.99
        ),  # different task, ignored
    ]
    base_frame = pd.DataFrame(
        [
            {
                "family": "b16",
                "role": "method",
                "arm": "mrixfields_b16_x",
                "metric": "ssim",
                "value": 0.7,
            }
        ],
        columns=pd.Index(["family", "role", "arm", "metric", "value"]),
    )
    merged = ca.merge_baseline_results(
        base_frame, results, "1", metrics=("ssim", "psnr", "lpips")
    )

    base_rows = merged[merged["role"] == "baseline"]
    assert set(base_rows["family"].unique()) == {"cut", "cyclegan"}
    # lpips_alex is mapped onto the "lpips" panel; psnr has no baseline source -> absent.
    assert set(base_rows["metric"].unique()) == {"ssim", "lpips"}
    # cut ssim bar = mean(0.60, 0.80) over its two task-1 units.
    cut_ssim = base_rows[
        (base_rows["family"] == "cut") & (base_rows["metric"] == "ssim")
    ]
    assert float(cut_ssim["value"].iloc[0]) == 0.70
    # the task-2 cut unit did not leak into the task-1 mean.
    cut_lpips = base_rows[
        (base_rows["family"] == "cut") & (base_rows["metric"] == "lpips")
    ]
    assert float(cut_lpips["value"].iloc[0]) == 0.40


def test_generate_baseline_overlay_figures_end_to_end(tmp_path) -> None:
    _make_arm(
        tmp_path,
        "mrixfields_b11_confluence",
        "1",
        [
            {
                "iteration": 100,
                "epoch": 2,
                "val_ssim": 0.7,
                "val_psnr": 25.0,
                "val_lpips": 0.4,
            }
        ],
    )
    results = [
        _baseline_result("cut", 1, 0.1, ssim=0.6, lpips=0.3),
        _baseline_result("cyclegan", 1, 0.1, ssim=0.55, lpips=0.4),
    ]
    results_json = tmp_path / "baseline_results.json"
    results_json.write_text(json.dumps(results))
    out_dir = tmp_path / "overlay"
    written = ca.generate_baseline_overlay_figures(
        tmp_path, results_json, out_dir, tasks=("1",)
    )
    assert written["1"] is not None and written["1"].exists()
    assert (out_dir / "task1_with_baselines.png").exists()


# ── run completion (#2532) ─────────────────────────────────────────────────────
_ROW = [{"iteration": 100, "epoch": 2, "val_ssim": 0.6, "val_psnr": 22.0, "val_lpips": 0.5}]
_OK, _CRASHED, _UNKNOWN = (
    "mrixfields_b16_ok",
    "mrixfields_b17_crashed",
    "mrixfields_b18_no_summary",
)


def _status_cohort(root: Path) -> None:
    """ok / crashed / no-summary; the crashed arm still holds an earlier attempt's metrics."""
    ok = _make_arm(root, _OK, "1", _ROW)
    (ok / "run_summary.json").write_text(json.dumps({"success": True}))
    crashed = _make_arm(root, _CRASHED, "1", _ROW)
    (crashed / "run_summary.json").write_text(
        json.dumps({"success": False, "error_type": "RuntimeError", "error": "boom"})
    )
    _make_arm(root, _UNKNOWN, "1", _ROW)


def test_discover_reads_each_arms_completion(tmp_path) -> None:
    _status_cohort(tmp_path)
    by = {a.arm: a.status.state for a in ca.discover_cohort_arms(tmp_path)}
    assert by == {_OK: "ok", _CRASHED: "failed", _UNKNOWN: "unknown"}


def test_crashed_arm_is_left_out_of_the_task_frame(tmp_path, caplog) -> None:
    """PLANTED (#2532): red before the fix -- the frame took every arm with val rows."""
    _status_cohort(tmp_path)
    # Not vacuous: the crashed arm has real validation metrics the old path plotted.
    assert ca.best_val_metrics(tmp_path / _CRASHED)
    with caplog.at_level("INFO"):
        frame = ca.build_task_frame(ca.discover_cohort_arms(tmp_path), "1")
    assert set(frame["arm"]) == {_OK, _UNKNOWN}
    assert f"run {_CRASHED}: crashed: RuntimeError (boom); excluded" in caplog.text
    assert f"run {_UNKNOWN}: status unknown: no run_summary.json; kept and marked" in caplog.text


def test_status_kwargs_mark_unknown_and_name_crashed(tmp_path) -> None:
    _status_cohort(tmp_path)
    kwargs = ca.status_kwargs(ca.discover_cohort_arms(tmp_path), "1")
    assert kwargs == {
        "run_notes": {_UNKNOWN: "status unknown: no run_summary.json"},
        "excluded_runs": {_CRASHED: "crashed: RuntimeError"},
    }
    assert ca.status_kwargs(ca.discover_cohort_arms(tmp_path), "2") == {
        "run_notes": {},
        "excluded_runs": {},
    }


def test_figures_receive_the_status_marks(tmp_path, monkeypatch) -> None:
    _status_cohort(tmp_path)
    seen: dict = {}

    def _plot(frame, out_path, **kwargs):
        seen["arms"] = set(frame["arm"])
        seen["run_notes"] = kwargs["run_notes"]
        seen["excluded_runs"] = kwargs["excluded_runs"]
        return out_path

    monkeypatch.setattr(ca, "get_plotter", lambda name: _plot)
    ca.generate_task_ablation_figures(tmp_path, tmp_path / "figs", tasks=("1",))
    assert seen["arms"] == {_OK, _UNKNOWN}
    assert set(seen["run_notes"]) == {_UNKNOWN}
    assert set(seen["excluded_runs"]) == {_CRASHED}
