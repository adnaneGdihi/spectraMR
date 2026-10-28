"""``comparison/all_metric_figures.py``: the extra figures of an every-metric comparison."""

import matplotlib

matplotlib.use("Agg")

import pytest

from spectramr.infrastructure.reporting.comparison import load_arms
from spectramr.infrastructure.reporting.comparison.all_metric_figures import (
    cohort_figures,
    curve_names,
    more_curves,
    sweeps,
    table_metrics,
)
from spectramr.infrastructure.reporting.comparison.contrast_stats import (
    ContrastStats,
    contrast_stats,
)
from spectramr.infrastructure.reporting.comparison.contrasts import Ladder
from spectramr.infrastructure.reporting.comparison.phase_tables import phase_table

from ._runs import add_training_metrics, add_validation_columns, make_run


@pytest.fixture
def arms(tmp_path):
    runs = [make_run(tmp_path, "base"), make_run(tmp_path, "better", shift=1.0)]
    for run in runs:
        add_training_metrics(run, level=20.0)
        add_validation_columns(run, ["cjv", "efc"])
    return load_arms(runs, labels=["base", "better"], with_images=False)


def test_curve_names_follow_the_metric_order_with_training_first(arms):
    names = curve_names(arms, ["ssim", "psnr", "cjv"])
    assert names == ["val_ssim", "train_psnr", "val_psnr", "val_cjv"]


def test_a_metric_nobody_logged_has_no_curve(arms):
    assert curve_names(arms, ["not_a_metric"]) == []


def test_curves_beyond_the_first_chunk_are_numbered_from_two(arms, tmp_path):
    figures: dict = {}
    names = ["val_psnr", "val_ssim", "val_cjv", "val_efc", "train_psnr"]
    more_curves(arms, names, tmp_path, ("png",), 50, 2, None, figures)
    assert sorted(figures) == [
        "fig_1_2_learning_curves_2",
        "fig_1_2_learning_curves_3",
        "fig_1_2_learning_curves_4",
    ]
    assert all(p is not None and p.exists() for p in figures.values())


def test_a_sweep_needs_two_accelerations_and_is_drawn_per_metric(arms, tmp_path):
    figures, reasons = {}, {}
    sweeps(arms, ["psnr", "cjv"], tmp_path, ("png",), 50, "selected", None, figures, reasons)
    assert figures["acceleration_sweep_psnr"].exists()
    assert "acceleration_sweep_cjv" not in figures and reasons == {}  # one value per pass


def test_the_table_reads_the_metrics_of_the_headline_rows(arms):
    assert {"psnr", "cjv"} <= table_metrics(arms)


def _table(arms):
    return phase_table(arms, {"val": ["psnr", "ssim", "cjv"]})


def test_without_statistics_only_the_table_driven_figures_are_drawn(arms, tmp_path):
    figures, reasons = cohort_figures(
        arms, _table(arms), None, None, None, ("val",), tmp_path, ("png",), 40, 12, None
    )
    assert sorted(figures) == [
        "cmp_coverage",
        "cmp_group_bars",
        "cmp_phase_bars",
        "cmp_rank_reversal",
    ]
    assert reasons == {}


def test_with_statistics_the_forest_and_subject_lines_follow_and_the_ladder_if_declared(
    arms, tmp_path
):
    from ._runs import add_slice_rows

    for lab in arms.labels:
        add_slice_rows(arms.run_dirs[lab], offset=1.0 if lab == "better" else 0.0, seed=len(lab))
    stats = contrast_stats("c", arms, ("psnr",), "base", None)
    assert stats.rows, "the fixture must yield a paired test"
    figures, _ = cohort_figures(
        arms,
        _table(arms),
        stats,
        "base",
        Ladder(metric="psnr"),
        ("val",),
        tmp_path,
        ("png",),
        40,
        12,
        None,
    )
    assert {
        "cmp_effect_forest",
        "cmp_subject_lines",
        "cmp_metric_ladder",
        "cmp_effect_bars",
        "cmp_metric_group_bars",
    } <= set(figures)
    assert all(p is not None and p.exists() for p in figures.values())
    no_ladder, _ = cohort_figures(
        arms, _table(arms), stats, "base", None, ("val",), tmp_path / "n", ("png",), 40, 12, None
    )
    assert "cmp_metric_ladder" not in no_ladder


def test_an_empty_statistics_object_draws_nothing_it_cannot(arms, tmp_path):
    figures, reasons = cohort_figures(
        arms,
        _table(arms),
        ContrastStats(),
        "base",
        None,
        ("val",),
        tmp_path,
        ("png",),
        40,
        12,
        None,
    )
    assert figures["cmp_effect_forest"] is None and "no_data" in reasons["cmp_effect_forest"]
