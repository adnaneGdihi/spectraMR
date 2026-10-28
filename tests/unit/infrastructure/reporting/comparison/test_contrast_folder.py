"""``comparison/contrast_folder.py``: each contrast lands alone, with its data, in its own folder."""

import json

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.comparison.contrast_folder import write_contrasts
from spectramr.infrastructure.reporting.comparison.contrasts import ContrastSet

from ._runs import add_inference_results, add_slice_rows, add_training_metrics, make_run


def _set(**over):
    base = {
        "results_root": ".",
        "metrics": ["val_psnr"],
        "contrasts": [
            {
                "name": "better_vs_base",
                "arms": ["base", "better"],
                "baseline": "base",
                "family": "f",
            },
            {"name": "best_vs_base", "arms": ["base", "best"], "baseline": "base", "family": "f"},
        ],
    }
    return ContrastSet.model_validate({**base, **over})


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    for seed, (name, offset) in enumerate((("base", 0.0), ("better", 0.5), ("best", 1.0))):
        add_slice_rows(make_run(root, name, shift=offset), offset=offset, seed=seed)
    return root


def _write(runs, tmp_path, contrast_set=None, **kw):
    return write_contrasts(
        contrast_set or _set(),
        tmp_path / "out",
        results_root=runs,
        formats=("png",),
        dpi=60,
        html_report=False,
        **kw,
    )


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


def test_each_contrast_is_stored_alone_with_its_data(runs, tmp_path):
    results = _write(runs, tmp_path)
    out = tmp_path / "out"
    assert [r.name for r in results] == ["better_vs_base", "best_vs_base"]
    folder = out / "better_vs_base"
    names = {p.name for p in (folder / "data").iterdir()}
    assert {"arm_metrics.csv", "per_case.csv", "paired_stats.csv", "unavailable.json"} <= names
    assert (folder / "figures").is_dir() and (folder / "tables").is_dir()
    assert (folder / "comparison_summary.md").is_file() and (folder / "contrast.json").is_file()
    # alone: no other arm's name in this folder's data
    assert "best" not in set(pd.read_csv(folder / "data" / "arm_metrics.csv")["method"])
    paired = pd.read_csv(folder / "data" / "paired_subjects" / "val__psnr__better_minus_base.csv")
    assert paired["difference"].mean() == pytest.approx(0.5, abs=0.1)
    assert len(paired) == 6


def test_every_logged_metric_and_the_mosaic_are_stored(runs, tmp_path):
    every = _set(metrics=None)
    _write(runs, tmp_path, contrast_set=every)
    folder = tmp_path / "out" / "better_vs_base"
    wide = pd.read_csv(folder / "data" / "all_metrics.csv").set_index("arm")
    assert {"val_psnr", "val_ssim", "val_zf_psnr"} <= set(wide.columns)
    assert wide.loc["better", "step"] == 200
    tested = set(pd.read_csv(folder / "data" / "paired_stats.csv")["metric"])
    assert tested == {"psnr", "reference_noise_corrected_mse"}
    figures = json.loads((folder / "contrast.json").read_text())["figures"]
    assert figures["cmp_mosaic"] is not None and (folder / figures["cmp_mosaic"]).is_file()


def test_contrast_json_names_the_runs_and_the_pass_read(runs, tmp_path):
    _write(runs, tmp_path)
    record = json.loads((tmp_path / "out" / "better_vs_base" / "contrast.json").read_text())
    assert record["view"] == "selected" and record["metrics"] == ["val_psnr"]
    assert record["runs"]["better"]["pass_read"] == 200
    assert record["runs"]["better"]["run_dir"].endswith("runs/better")


def test_holm_spans_the_family_across_folders(runs, tmp_path):
    _write(runs, tmp_path)
    stats = [
        pd.read_csv(tmp_path / "out" / n / "data" / "paired_stats.csv").iloc[0]
        for n in ("better_vs_base", "best_vs_base")
    ]
    assert all(s["holm_m"] == 2 and s["holm_family"] == "f" for s in stats)
    smaller = min(stats, key=lambda s: s["p_ttest"])
    assert smaller["p_holm"] == pytest.approx(min(1.0, 2 * smaller["p_ttest"]))


def test_only_rewrites_one_folder_and_keeps_the_index(runs, tmp_path):
    _write(runs, tmp_path)
    _write(runs, tmp_path, only=["best_vs_base"])
    index = pd.read_csv(tmp_path / "out" / "INDEX.csv")
    assert sorted(index["contrast"]) == ["best_vs_base", "better_vs_base"]
    assert index.set_index("contrast").loc["better_vs_base", "tests"] == 1


def test_only_does_not_change_the_corrected_p_value(runs, tmp_path):
    _write(runs, tmp_path)
    full = pd.read_csv(tmp_path / "out" / "best_vs_base" / "data" / "paired_stats.csv")
    _write(runs, tmp_path / "again", only=["best_vs_base"])
    part = pd.read_csv(tmp_path / "again" / "out" / "best_vs_base" / "data" / "paired_stats.csv")
    assert part.loc[0, "holm_m"] == 2
    assert part.loc[0, "p_holm"] == pytest.approx(full.loc[0, "p_holm"])
    assert not (tmp_path / "again" / "out" / "better_vs_base").exists()


def test_an_unknown_only_name_and_a_missing_run_raise(runs, tmp_path):
    with pytest.raises(ValueError, match="unknown contrast"):
        _write(runs, tmp_path, only=["nope"])
    gone = _set(contrasts=[{"name": "x", "arms": ["base", "absent"]}])
    with pytest.raises(FileNotFoundError, match="absent"):
        _write(runs, tmp_path, contrast_set=gone)


def test_runs_without_slice_rows_still_produce_the_folder_and_say_why(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    make_run(root, "base")
    make_run(root, "better", shift=1.0)
    only = _set(contrasts=[{"name": "c", "arms": ["base", "better"], "baseline": "base"}])
    (result,) = _write(root, tmp_path, contrast_set=only)
    assert result.n_tests == 0 and "wrote no per-slice rows" in result.unavailable[0]["reason"]
    folder = tmp_path / "out" / "c" / "data"
    assert pd.read_csv(folder / "paired_stats.csv").empty
    assert json.loads((folder / "unavailable.json").read_text())[0]["metric"] == "psnr"


def test_all_three_phases_are_stored_and_the_test_phase_is_paired(runs, tmp_path):
    for name, offset in (("base", 0.0), ("better", 0.5), ("best", 1.0)):
        add_training_metrics(runs / name, level=20.0 + offset)
        add_inference_results(runs / name, offset=offset)
    every = _set(metrics=None)
    _write(runs, tmp_path, contrast_set=every)
    data = tmp_path / "out" / "better_vs_base" / "data"
    long = pd.read_csv(data / "metrics_long.csv")
    assert set(long["phase"]) == {"val", "train", "test"}
    assert {"psnr", "nmse"} <= set(long.loc[long["phase"] == "test", "metric"])
    assert (data / "metrics_train.csv").is_file() and (data / "metrics_test.csv").is_file()
    stats = pd.read_csv(data / "paired_stats.csv")
    test_psnr = stats[(stats["phase"] == "test") & (stats["metric"] == "psnr")].iloc[0]
    assert test_psnr["mean_diff"] == pytest.approx(0.5) and test_psnr["unit"] == "cases"
    assert (data / "paired_subjects" / "test__psnr__better_minus_base.csv").is_file()
    record = json.loads((tmp_path / "out" / "better_vs_base" / "contrast.json").read_text())
    assert "training writes no per-item rows" in record["phases"]["train"]["paired_tests"]


def test_phases_can_be_limited(runs, tmp_path):
    only_val = _set(phases=["val"])
    _write(runs, tmp_path, contrast_set=only_val)
    data = tmp_path / "out" / "better_vs_base" / "data"
    assert set(pd.read_csv(data / "metrics_long.csv")["phase"]) == {"val"}
    assert not (data / "metrics_train.csv").exists()


def test_a_metric_an_inference_run_skipped_is_listed_with_its_reason(runs, tmp_path):
    for name, offset in (("base", 0.0), ("better", 0.5), ("best", 1.0)):
        add_inference_results(runs / name, offset=offset)  # records ssim as skipped
    _write(runs, tmp_path, contrast_set=_set(metrics=None))
    listed = json.loads(
        (tmp_path / "out" / "better_vs_base" / "data" / "unavailable.json").read_text()
    )
    skipped = [u for u in listed if u["phase"] == "test" and u["metric"] == "ssim"]
    assert {u["variant"] for u in skipped} == {"base", "better"}
    assert all("no_reference_available_on_this_path" in u["reason"] for u in skipped)


def test_the_cohort_figures_are_drawn_into_the_folder(runs, tmp_path):
    spec = _set(
        metrics=None,
        contrasts=[
            {
                "name": "better_vs_base",
                "arms": ["base", "better"],
                "baseline": "base",
                "ladder": {"metric": "psnr", "references": {"reference": 30.0}},
            }
        ],
    )
    _write(runs, tmp_path, contrast_set=spec)
    folder = tmp_path / "out" / "better_vs_base"
    figures = json.loads((folder / "contrast.json").read_text())["figures"]
    for fid in ("cmp_coverage", "cmp_effect_forest", "cmp_subject_lines", "cmp_metric_ladder"):
        assert figures[fid] is not None and (folder / figures[fid]).is_file(), fid
    manifest = json.loads((folder / "report_manifest.json").read_text())
    assert not _undrawn(manifest), _undrawn(manifest)


def test_declared_groups_reach_the_group_bars_and_every_artifact_is_ok(runs, tmp_path):
    spec = _set(
        metrics=None,
        contrasts=[
            {
                "name": "grouped",
                "arms": [
                    {"run": "base", "group": "baseline"},
                    {"run": "better", "group": "tuned"},
                    {"run": "best", "group": "tuned"},
                ],
                "baseline": "base",
            }
        ],
    )
    _write(runs, tmp_path, contrast_set=spec)
    folder = tmp_path / "out" / "grouped"
    figures = json.loads((folder / "contrast.json").read_text())["figures"]
    for fid in ("cmp_group_bars", "cmp_phase_bars", "cmp_effect_bars", "cmp_metric_group_bars"):
        assert figures[fid] is not None and (folder / figures[fid]).is_file(), fid
    manifest = json.loads((folder / "report_manifest.json").read_text())
    assert not _undrawn(manifest), _undrawn(manifest)
