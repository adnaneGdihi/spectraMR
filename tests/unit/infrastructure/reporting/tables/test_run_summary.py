"""The tables-only report floor must actually have a floor.

``generate_report`` on a plain training run used to emit NOTHING. Its ``default``
preset holds two publication tables — ``tab_2_1_main_results`` (rows = methods)
and ``tab_2_4_dataset_descriptor`` (needs a cohort descriptor) — and a single run
has neither, so both correctly returned ``None``. Verified on a real run dir
carrying 27 rows of perfectly good metrics: zero files written.

That made "fall back to tables" an empty promise, and an empty promise is worse
than the legacy generator it would have replaced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spectramr.infrastructure.reporting.tables import TABLES, run_summary


def _frame() -> pd.DataFrame:
    """Long-format, matching what ``aggregator.aggregate`` emits."""
    return pd.DataFrame(
        [
            {"split": "val", "metric": "val_psnr", "value": 20.0, "step": 10},
            {"split": "val", "metric": "val_psnr", "value": 25.0, "step": 20},
            {"split": "val", "metric": "val_psnr", "value": 22.0, "step": 30},
            {"split": "train", "metric": "loss", "value": 0.9, "step": 10},
            {"split": "train", "metric": "loss", "value": 0.3, "step": 20},
            {"split": "train", "metric": "loss", "value": 0.5, "step": 30},
        ]
    )


class TestRunSummaryTable:
    def test_it_is_registered(self):
        assert "tab_run_summary" in TABLES

    def test_emits_all_three_formats(self, tmp_path):
        out = run_summary.make(_frame(), tmp_path)
        assert out is not None
        assert set(out) == {"tex", "md", "csv"}
        for p in out.values():
            assert p.exists() and p.stat().st_size > 0

    def test_best_follows_the_direction_ssot_not_the_last_value(self, tmp_path):
        """The whole point. ``val_psnr`` peaks mid-run and ``loss`` bottoms
        mid-run; a naive "final value" summary would report neither."""
        run_summary.make(_frame(), tmp_path)
        df = pd.read_csv(tmp_path / "run_summary.csv").set_index("metric")

        psnr = df.loc["val_psnr"]
        assert psnr["final"] == 22.0 and psnr["best"] == 25.0
        assert psnr["best_step"] == 20 and psnr["direction"] == "higher"

        loss = df.loc["loss"]
        assert loss["final"] == 0.5 and loss["best"] == 0.3
        assert loss["best_step"] == 20 and loss["direction"] == "lower"

    def test_an_unknown_metric_gets_no_best_rather_than_a_guessed_one(self, tmp_path):
        """``resolve_direction`` returns None for an unrecognised key. Defaulting
        to max is exactly what made ``best_metric_name: lpips`` maximise LPIPS
        (#208), so an empty cell is the correct answer here."""
        df = pd.DataFrame(
            [
                {"split": "val", "metric": "zzz_not_a_metric", "value": 1.0, "step": 1},
                {"split": "val", "metric": "zzz_not_a_metric", "value": 9.0, "step": 2},
            ]
        )
        run_summary.make(df, tmp_path)
        row = pd.read_csv(tmp_path / "run_summary.csv").iloc[0]
        assert row["final"] == 9.0
        assert pd.isna(row["best"]) and pd.isna(row["best_step"])
        assert pd.isna(row["direction"]) or row["direction"] == ""

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame(),
            pd.DataFrame([{"unrelated": 1}]),
            pd.DataFrame([{"split": "val", "metric": "val_psnr", "value": None}]),
        ],
        ids=["empty", "wrong-columns", "all-null-values"],
    )
    def test_returns_none_rather_than_a_header_over_nothing(self, df, tmp_path):
        """A header with no rows reads as "measured, came back blank"."""
        assert run_summary.make(df, tmp_path) is None
        assert not list(tmp_path.glob("run_summary.*"))

    def test_works_without_a_step_column(self, tmp_path):
        """Hand-built frames (and older aggregator output) carry no ``step``."""
        df = _frame().drop(columns=["step"])
        assert run_summary.make(df, tmp_path) is not None


class TestThePublicationTablesCannotServeAsTheFloor:
    """The headline table reads one checkpoint; the floor reads every metric.

    ``tab_2_1`` used to be empty on every real run (it matched bare ``psnr``
    against logged ``val_psnr``), and this class pinned that emptiness as the
    reason the floor exists. The floor's reason is different: it carries every
    metric of every split with its best step, which one headline row cannot.
    """

    def test_main_results_reads_one_pass_and_drops_training_losses(self, tmp_path):
        from spectramr.infrastructure.reporting.tables import main_results

        out = main_results.make(_frame(), tmp_path)
        assert out is not None
        table = pd.read_csv(out["csv"])
        assert {"psnr_mean", "psnr_std"} <= set(table.columns)
        assert not any(c.startswith("loss") for c in table.columns)
        # no early-stopping record -> the final pass, never the trajectory mean
        assert table.loc[0, "psnr_mean"] == 22.0
        assert pd.isna(table.loc[0, "psnr_std"])

    def test_the_floor_keeps_what_the_headline_table_drops(self, tmp_path):
        run_summary.make(_frame(), tmp_path)
        assert "loss" in set(pd.read_csv(tmp_path / "run_summary.csv")["metric"])


def test_the_shared_zero_filled_column_is_never_ranked(tmp_path):
    from spectramr.infrastructure.reporting.tables import main_results

    df = pd.DataFrame(
        [
            {"run_id": m, "method": m, "split": "val", "step": 1, "metric": k, "value": v}
            for m, psnr in (("a", 20.0), ("b", 21.0))
            for k, v in (("val_psnr", psnr), ("val_zf_psnr", 23.0))
        ]
    )
    md = main_results.make(df, tmp_path)["markdown"].read_text()
    assert "**23.000**" not in md and "_23.000_" not in md
    assert "**21.000**" in md

def test_methods_without_a_test_split_stay_in_the_table(tmp_path):
    from spectramr.infrastructure.reporting.tables import main_results

    df = pd.DataFrame([
        {"run_id": "a", "method": "a", "split": "test", "step": -1, "metric": "psnr", "value": 30.0},
        {"run_id": "b", "method": "b", "split": "val", "step": 5, "metric": "val_psnr", "value": 29.0},
    ])
    table = pd.read_csv(main_results.make(df, tmp_path)["csv"]).set_index("method")
    assert set(table.index) == {"a", "b"}


def test_the_caption_names_subjects_and_claims_no_test_it_did_not_run(tmp_path):
    from spectramr.infrastructure.reporting.tables import main_results

    df = pd.DataFrame([
        {"run_id": m, "method": m, "split": "test", "step": -1, "metric": "psnr",
         "value": v + i, "subject_id": f"s{i}"}
        for m, v in (("a", 30.0), ("b", 31.0)) for i in range(3)
    ])
    tex = main_results.make(df, tmp_path)["latex"].read_text()
    assert "over subjects" in tex and "p<0.05" not in tex
    tex = main_results.make(df, tmp_path, baseline_method="a")["latex"].read_text()
    assert "p<0.05" in tex


def test_params_are_reported_from_the_run_split(tmp_path):
    from spectramr.infrastructure.reporting.tables import main_results

    df = pd.DataFrame([
        {"run_id": "a", "method": "a", "split": "val", "step": 5, "metric": "val_psnr", "value": 29.0},
        {"run_id": "a", "method": "a", "split": "run", "step": -1, "metric": "params_m", "value": 12.5},
    ])
    table = pd.read_csv(main_results.make(df, tmp_path)["csv"])
    assert table.loc[0, "params_m"] == 12.5

def test_a_subject_with_several_rows_is_still_tested(tmp_path):
    from spectramr.infrastructure.reporting.tables import main_results

    rng = np.random.default_rng(0)
    df = pd.DataFrame([
        {"run_id": m, "method": m, "split": "test", "step": -1, "metric": "psnr",
         "value": 30.0 + shift + rng.normal(0, 0.05), "subject_id": f"s{i}"}
        for m, shift in (("a", 0.0), ("b", 1.0)) for i in range(6) for _ in range(2)
    ])
    md = main_results.make(df, tmp_path, baseline_method="a")["markdown"].read_text()
    assert "*" in md.split("| b |")[1].split("\n")[0]


def test_test_split_and_validation_pass_are_marked_and_not_ranked(tmp_path):
    from spectramr.infrastructure.reporting.tables import main_results

    df = pd.DataFrame([
        {"run_id": "a", "method": "a", "split": "test", "step": -1, "metric": "psnr", "value": 31.0},
        {"run_id": "b", "method": "b", "split": "val", "step": 5, "metric": "val_psnr", "value": 42.0},
    ])
    out = main_results.make(df, tmp_path)
    md, tex = out["markdown"].read_text(), out["latex"].read_text()
    assert "b †" in md and "**" not in md
    assert "dagger" in tex and "not ranked" in tex
