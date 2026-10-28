"""The cross-run report end to end, and its two plotters."""

import json

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from spectramr.infrastructure.reporting import plotters
from spectramr.infrastructure.reporting.comparison import (
    generate_comparison,
    load_arms,
    metric_bars,
    mosaic,
    resolve_runs,
)
from spectramr.infrastructure.reporting.plotters.distributions import SLICE_FIGURES

from ._runs import (
    add_inference_results,
    add_training_metrics,
    add_validation_columns,
    make_run,
)


@pytest.fixture
def cohort(tmp_path):
    make_run(tmp_path, "exp_base_run", shift=0.0)
    make_run(tmp_path, "exp_better_run", shift=1.5)
    return tmp_path


def _undrawn(manifest):
    """Artifacts that are not ok, bar the figures that need per-slice rows, image cases or checkpoints."""
    from spectramr.infrastructure.reporting.plotters.distributions import (
        CASE_FIGURES,
        SLICE_FIGURES,
    )

    needs_more_data = {
        *SLICE_FIGURES,
        *CASE_FIGURES,
        "fig_4_1_weight_trajectory",
        "fig_4_2_layer_drift",
        "fig_4_4_image_embedding",
    }

    return [
        a
        for a in manifest["artifacts"]
        if a["status"] != "ok"
        and not (
            a["status"] == "skipped" and a.get("reason") == "no_data" and a["id"] in needs_more_data
        )
    ]


def test_both_plotters_are_registered():
    assert {"cmp_mosaic", "cmp_metric_bars"} <= set(plotters.list_available())


def test_generate_comparison_writes_every_figure_and_the_table(cohort, tmp_path):
    out = tmp_path / "cmp"
    res = generate_comparison([cohort], out, baseline="base", formats=("png",), dpi=60)
    assert res["arms"] == ["base", "better"]
    original = ("cmp_metric_bars", "fig_1_2_learning_curves", "acceleration_sweep", "cmp_mosaic")
    assert all(res["figures"][f] is not None and res["figures"][f].exists() for f in original)
    # the per-slice figures are attempted; a cohort without slice rows soft-skips them
    assert set(SLICE_FIGURES) <= set(res["figures"])
    table = pd.read_csv(res["tables"]["tab_2_1_main_results"]["csv"]).set_index("method")
    assert table.loc["better", "psnr_mean"] == pytest.approx(25 + 1.5 + 2.0)
    manifest = json.loads((out / "report_manifest.json").read_text())
    by_status = {
        a["status"]
        for a in manifest["artifacts"]
        if a.get("id") in (*original, "tab_2_1_main_results")
    }
    assert by_status <= {"ok"}
    assert {a["status"] for a in manifest["artifacts"]} <= {"ok", "skipped"}
    assert "| base |" in res["summary"].read_text()


def test_an_arm_note_reaches_the_comparison_table_caption(cohort, tmp_path):
    """End to end, because ``tables.dispatch`` turns a table failure into ``None``."""
    (cohort / "exp_better_run" / "resolved_config.json").write_text(
        json.dumps({"metadata": {"comparison_note": "carries two attention blocks"}})
    )
    res = generate_comparison([cohort], tmp_path / "cmp", formats=("png",), dpi=60)
    tex = res["tables"]["tab_2_1_main_results"]["latex"].read_text()
    caption = next(line for line in tex.splitlines() if line.startswith(r"\caption"))
    assert r"$^\S$ better: carries two attention blocks." in caption


def test_an_unknown_baseline_raises(cohort, tmp_path):
    with pytest.raises(ValueError, match="not an arm"):
        generate_comparison([cohort], tmp_path / "x", baseline="nope", formats=("png",))


def test_metric_bars_test_the_better_arm_against_the_baseline(cohort):
    arms = load_arms([cohort / "exp_base_run", cohort / "exp_better_run"])
    rows = arms.per_case[
        (arms.per_case["base_metric"] == "psnr") & (arms.per_case["acceleration"] == 2.0)
    ]
    p = metric_bars._pvalues(rows, arms.labels, "base")
    assert p["better"] < 0.05


def test_metric_bars_without_per_case_rows_draw_one_value_per_arm(tmp_path):
    make_run(tmp_path, "exp_a_run", per_call=False)
    make_run(tmp_path, "exp_b_run", per_call=False)
    arms = load_arms([tmp_path / "exp_a_run", tmp_path / "exp_b_run"], with_images=False)
    assert arms.per_case.empty
    res = metric_bars.make(arms.df, tmp_path / "b.png", arms=arms, formats=("png",), dpi=60)
    assert res is not None and res.exists()


def test_mosaic_skips_arms_without_a_shared_reference(tmp_path):
    make_run(tmp_path, "exp_a_run", reference_seed=0)
    make_run(tmp_path, "exp_b_run", reference_seed=5)
    arms = load_arms([tmp_path / "exp_a_run", tmp_path / "exp_b_run"])
    assert mosaic.make(arms.df, tmp_path / "m.png", arms=arms, formats=("png",)) is None


def test_mosaic_defaults_to_one_slice_per_acceleration(cohort, tmp_path):
    arms = load_arms([cohort / "exp_base_run", cohort / "exp_better_run"])
    res = mosaic.make(arms.df, tmp_path / "m.png", arms=arms, formats=("png",), dpi=60)
    assert res is not None and res.exists()


def test_an_arm_without_per_case_rows_is_drawn_hollow_not_dropped(tmp_path, monkeypatch):
    from spectramr.infrastructure.reporting.comparison import metric_bars as mb

    make_run(tmp_path, "exp_a_run")
    make_run(tmp_path, "exp_b_run")
    make_run(tmp_path, "exp_c_run", per_call=False)
    arms = load_arms(resolve_runs([tmp_path]), with_images=False)
    seen = {}
    real = mb._per_case_panel

    def spy(ax, rows, fallback, labels, *a, **k):
        seen["labels"], seen["fallback"] = list(labels), set(fallback["method"])
        return real(ax, rows, fallback, labels, *a, **k)

    monkeypatch.setattr(mb, "_per_case_panel", spy)
    res = mb.make(arms.df, tmp_path / "b.png", arms=arms, baseline="c", formats=("png",), dpi=50)
    assert res is not None
    assert seen["labels"] == ["a", "b", "c"]


@pytest.fixture
def phased(cohort):
    """The cohort with training metrics, inference results and extra validation metrics."""
    for name, shift in (("exp_base_run", 0.0), ("exp_better_run", 1.5)):
        run = cohort / name
        add_training_metrics(run, level=20.0 + shift)
        add_inference_results(run, offset=shift)
        add_validation_columns(run, ["cjv", "efc"], offset=shift)
    return cohort


def test_every_metric_of_every_phase_reaches_the_figures_tables_and_data(phased, tmp_path):
    out = tmp_path / "all"
    res = generate_comparison(
        [phased], out, baseline="base", formats=("png",), dpi=60, metrics="all", html_report=False
    )
    manifest = json.loads((out / "report_manifest.json").read_text())
    assert not _undrawn(manifest), _undrawn(manifest)
    assert res["figures"]["cmp_phase_metrics"].exists()
    long = pd.read_csv(out / "data" / "metrics_long.csv")
    assert set(long["phase"]) == {"val", "train", "test"}
    assert {"cjv", "efc", "nmse"} <= set(long["metric"])
    val_tex = (out / "tables" / "metrics_val.tex").read_text()
    assert "cjv" in val_tex and "efc" in val_tex  # beyond psnr and ssim
    assert (out / "tables" / "metrics_test.tex").is_file() and res["tables"]


def test_a_metric_list_limits_the_phases_and_the_panels(phased, tmp_path):
    res = generate_comparison(
        [phased],
        tmp_path / "one",
        formats=("png",),
        dpi=60,
        metrics=["cjv"],
        phases=("val",),
        html_report=False,
    )
    long = pd.read_csv(tmp_path / "one" / "data" / "metrics_long.csv")
    assert set(long["phase"]) == {"val"}
    assert res["figures"]["cmp_metric_bars"].exists()


def test_a_metric_no_phase_logged_raises(phased, tmp_path):
    with pytest.raises(ValueError, match="no phase logged"):
        generate_comparison([phased], tmp_path / "x", metrics=["not_a_metric"], formats=("png",))


def test_more_metrics_than_panels_continue_in_numbered_figures(cohort, tmp_path):
    names = [f"m{i:02d}" for i in range(14)]
    for run in ("exp_base_run", "exp_better_run"):
        add_validation_columns(cohort / run, names)
    res = generate_comparison(
        [cohort],
        tmp_path / "many",
        formats=("png",),
        dpi=60,
        metrics=names,
        phases=("val",),
        max_panels=5,
        html_report=False,
    )
    bars = sorted(k for k in res["figures"] if k.startswith("cmp_metric_bars"))
    assert bars == ["cmp_metric_bars", "cmp_metric_bars_2", "cmp_metric_bars_3"]  # 5 + 5 + 4
    assert all(res["figures"][k].exists() for k in bars)


def test_the_acceleration_sweep_is_drawn_for_each_metric_with_accelerations(phased, tmp_path):
    res = generate_comparison(
        [phased], tmp_path / "sw", formats=("png",), dpi=60, metrics="all", html_report=False
    )
    assert res["figures"].get("acceleration_sweep_psnr") is not None


def test_learning_curves_for_many_metrics_continue_in_numbered_figures(cohort, tmp_path, recwarn):
    names = [f"m{i:02d}" for i in range(30)]
    for run in ("exp_base_run", "exp_better_run"):
        add_validation_columns(cohort / run, names)
    res = generate_comparison(
        [cohort],
        tmp_path / "curves",
        formats=("png",),
        dpi=50,
        metrics=names,
        phases=("val",),
        max_panels=12,
        html_report=False,
    )
    curves = sorted(k for k in res["figures"] if k.startswith("fig_1_2_learning_curves"))
    assert curves == [
        "fig_1_2_learning_curves",
        "fig_1_2_learning_curves_2",
        "fig_1_2_learning_curves_3",
    ]
    assert all(res["figures"][k].exists() for k in curves)  # 12 + 12 + 6 panels
    assert not [w for w in recwarn if "Glyph" in str(w.message)]  # no panel letter past z
