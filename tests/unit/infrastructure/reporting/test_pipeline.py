import importlib.util
from pathlib import Path

import pytest

from spectramr.infrastructure.reporting import pipeline as pipe
from tests.utils.corpus import tracked_yamls


def test_resolve_preset_unknown_task_raises():
    with pytest.raises(ValueError):
        pipe._resolve_preset("not_a_task")


def test_validate_ids_unknown_figure_raises():
    with pytest.raises(ValueError, match="unknown figure"):
        pipe._validate_figure_ids(["fig_1_2_learning_curves", "does_not_exist"])


def test_validate_ids_unknown_table_raises():
    with pytest.raises(ValueError, match="unknown table"):
        pipe._validate_table_ids(["tab_2_1_main_results", "ghost_table"])


def test_reconstruction_preset_includes_new_figures():
    preset = pipe.TASK_PRESETS["reconstruction"]
    for fid in (
        "mri_recon_panel",
        "kspace_error_spectrum",
        "metric_distribution",
        "acceleration_sweep",
        "contact_sheet",
    ):
        assert fid in preset["figures"]


def test_all_presets_end_with_contact_sheet():
    for task, preset in pipe.TASK_PRESETS.items():
        assert preset["figures"][-1] == "contact_sheet", task


def test_calibration_preset_resolves_and_uses_registered_ids():
    """The conformal-calibration arms report a CERTIFICATE, not a recon. The
    preset must exist, resolve, and reference only registered figures/tables
    (coverage reliability, significance, agreement) — not silently fall back to
    `default`. Pairs with the ReportTask.CALIBRATION enum member."""
    preset = pipe._resolve_preset("calibration")
    assert "calibration_coverage" in preset["figures"]
    assert "significance_matrix" in preset["figures"]
    # Must not raise — every id is a registered figure / table.
    pipe._validate_figure_ids(preset["figures"])
    pipe._validate_table_ids(preset["tables"])


def test_inprogress_conformal_arms_use_a_preset_backed_task():
    """Regression: the 5 ``inprogress/conformal`` arms set
    ``reporting.task: certification`` — neither a ``ReportTask`` member nor a
    ``TASK_PRESETS`` key — so they raised ``ValidationError`` at load AND
    ``_resolve_preset`` would have raised (they carry no explicit ``figures``
    list to fall back on). The ``calibration`` preset is the one *designed* to
    grade certificates, so the arms now use it. This pins every conformal arm's
    task to a preset-backed, enum-valid value."""
    import pathlib

    import yaml as _yaml

    from spectramr.config.schemas.enums import ReportTask

    conformal = (
        pathlib.Path(__file__).resolve().parents[4]
        / "experiments"
        / "inprogress"
        / "conformal"
    )
    valid = {t.value for t in ReportTask}
    arms = tracked_yamls(conformal, recursive=False)
    assert arms, "no conformal arms found — path drift?"
    for arm in arms:
        task = _yaml.safe_load(arm.read_text())["reporting"]["task"]
        assert task in valid, f"{arm.name}: reporting.task={task!r} not a ReportTask"
        # Raises if the task has no figure/table preset (these arms have no
        # explicit figures list, so the preset MUST exist).
        pipe._resolve_preset(task)


import pandas as pd

from spectramr.infrastructure.reporting import plotters


def test_dispatch_emits_multiple_formats(tmp_path):
    df = pd.DataFrame(
        {
            "method": ["m"] * 4,
            "metric": ["loss"] * 4,
            "split": ["train"] * 4,
            "step": [1, 2, 3, 4],
            "value": [1.0, 0.5, 0.3, 0.2],
        }
    )
    out = plotters.dispatch(
        df, ["fig_1_2_learning_curves"], tmp_path, formats=("pdf", "png")
    )
    base = out["fig_1_2_learning_curves"]
    assert base is not None
    assert base.with_suffix(".png").exists()
    assert base.with_suffix(".pdf").exists()


def test_dispatch_does_not_crash_on_unmigrated_plotter_kwargs(tmp_path):
    # passing an extra kwarg a plotter doesn't accept must be filtered, not crash
    df = pd.DataFrame(
        {
            "method": ["m"],
            "metric": ["loss"],
            "split": ["train"],
            "step": [1],
            "value": [0.5],
        }
    )
    out = plotters.dispatch(
        df,
        ["fig_1_2_learning_curves"],
        tmp_path,
        formats=("pdf",),
        some_unknown_kwarg=123,
    )
    assert "fig_1_2_learning_curves" in out


import numpy as np

from spectramr.infrastructure.reporting.cases.recorder import ReportCaseRecorder


def test_generate_report_autoloads_cases(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    rec = ReportCaseRecorder(
        n_cases=3,
        selection="best_median_worst",
        primary_metric="psnr",
        higher_is_better=True,
    )
    for i in range(5):
        rec.observe(
            case_id=f"s{i}",
            arrays={
                "input": np.zeros((8, 8), np.float32),
                "prediction": np.full((8, 8), 0.1 * i, np.float32),
                "target": np.ones((8, 8), np.float32),
            },
            metrics={"psnr": 20.0 + i},
            domain={"acceleration": 4},
        )
    rec.write(tmp_path)
    res = pipe.generate_report(
        tmp_path, task="reconstruction", figures=["mri_a7_kspace_recon"], tables_=[]
    )
    assert res["figures"].get("mri_a7_kspace_recon") is not None


def test_generate_report_emits_manifest(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    res = pipe.generate_report(
        tmp_path,
        task="default",
        figures=["fig_1_2_learning_curves"],
        tables_=[],
        emit_manifest=True,
    )
    assert (res["out_dir"] / "report_manifest.json").exists()


def test_generate_report_accepts_panel_labels(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    res = pipe.generate_report(
        tmp_path,
        task="default",
        figures=["fig_1_2_learning_curves"],
        tables_=[],
        panel_labels=False,
    )
    assert res["figures"].get("fig_1_2_learning_curves") is not None
    from spectramr.infrastructure.reporting import style

    style.set_panel_labels(True)  # restore


# ---------------------------------------------------------------------------
# 2026-07-01 upgrade: tikz emission + headline summary + new-figure presets
# ---------------------------------------------------------------------------


def test_generate_report_emits_tikz_bundle(tmp_path):
    import json

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n3,0.2\n")
    (tmp_path / "run_summary.json").write_text(json.dumps({"model_params": 5_000_000}))
    res = pipe.generate_report(
        tmp_path,
        task="default",
        figures=["fig_1_2_learning_curves"],
        tables_=[],
        tikz=True,
    )
    assert res["tikz"], "tikz bundle should contain at least one figure"
    for fid, path in res["tikz"].items():
        assert path.exists(), fid
        assert path.suffix == ".tex"
        assert path.with_suffix(".tikz").exists()
    # tikz artifacts ride in the manifest like any figure
    manifest = json.loads((res["out_dir"] / "report_manifest.json").read_text())
    ids = {a["id"] for a in manifest["artifacts"]}
    assert any(i.startswith("tikz_") for i in ids)


def test_generate_report_tikz_off_by_default(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    res = pipe.generate_report(
        tmp_path, task="default", figures=["fig_1_2_learning_curves"], tables_=[]
    )
    assert res["tikz"] == {}
    assert not (res["out_dir"] / "figures" / "tikz").exists()


def test_summary_embeds_pngs_and_headline_numbers(tmp_path):
    import json

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text(
        "step,val_loss,val_ssim\n1,0.5,0.80\n2,0.3,0.84\n"
    )
    # the per-metric best block is not what the summary quotes: it mixes checkpoints
    (tmp_path / "final_metrics.json").write_text(
        json.dumps({"best": {"ssim_best": 0.99}, "early_stopping_best_iteration": 2})
    )
    res = pipe.generate_report(
        tmp_path, task="default", figures=["fig_1_2_learning_curves"], tables_=[]
    )
    text = (res["out_dir"] / "report_summary.md").read_text()
    assert "## Headline numbers (selected pass, step 2)" in text
    assert "validation SSIM | 0.84" in text and "0.99" not in text
    assert "![fig_1_2_learning_curves]" in text  # inline PNG embed


def test_default_preset_includes_upgrade_figures():
    preset = pipe.TASK_PRESETS["default"]
    for fid in (
        "fig_1_16_run_summary_card",
        "fig_1_17_metric_correlation",
        "fig_1_18_train_val_gap",
    ):
        assert fid in preset["figures"]
    # and they resolve to registered plotters
    pipe._validate_figure_ids(preset["figures"])


# ---------------------------------------------------------------------------
# Default = every registered figure
# ---------------------------------------------------------------------------


def test_all_figure_ids_covers_registry_with_contact_sheet_last():
    ids = pipe._all_figure_ids()
    assert set(ids) == set(plotters.list_available())
    assert ids[-1] == "contact_sheet"  # composites the others → must run last


def test_default_report_attempts_the_task_preset(tmp_path):
    """No explicit ``figures`` → the DEFAULT TASK PRESET, not all 44 (#719).

    This asserted the full registry, which is what the code did and what neither
    the schema (``figures``: "None = preset default") nor the docs ever said.
    The preset is authoritative now, so `task:` finally selects something.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    res = pipe.generate_report(tmp_path, tables_=[], html_report=False)
    attempted = set(res["figures"])

    assert attempted == set(pipe.TASK_PRESETS["default"]["figures"])
    assert attempted < set(plotters.list_available()), "preset must be a subset"


def test_a_non_default_task_attempts_its_own_preset(tmp_path):
    """Anti-vacuity: the fix is worthless if every task resolves the same."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    res = pipe.generate_report(
        tmp_path, task="reconstruction", tables_=[], html_report=False
    )

    assert set(res["figures"]) == set(pipe.TASK_PRESETS["reconstruction"]["figures"])
    assert set(res["figures"]) != set(pipe.TASK_PRESETS["default"]["figures"])


def test_default_report_honours_qc_gate(tmp_path):
    """qc_figures=False strips the qc_ ids from the preset."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    res = pipe.generate_report(
        tmp_path, tables_=[], html_report=False, qc_figures=False
    )
    attempted = set(res["figures"])
    assert not any(k.startswith("qc_") for k in attempted)
    assert attempted == {
        f for f in pipe.TASK_PRESETS["default"]["figures"] if not f.startswith("qc_")
    }


def test_a_qc_less_preset_still_gets_qc_when_enabled(tmp_path):
    """The additive half, end to end: `gan` lists no ``qc_*`` in its preset."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    res = pipe.generate_report(tmp_path, task="gan", tables_=[], html_report=False)

    assert not any(f.startswith("qc_") for f in pipe.TASK_PRESETS["gan"]["figures"])
    assert [k for k in res["figures"] if k.startswith("qc_")]


# ---------------------------------------------------------------------------
# QC report wiring
# ---------------------------------------------------------------------------


def test_qc_figures_in_recon_and_default_presets():
    for task in ("reconstruction", "super_resolution", "synthesis", "default"):
        figs = pipe.TASK_PRESETS[task]["figures"]
        for fid in ("qc_group_strip", "qc_subject_mosaic", "qc_carpet"):
            assert fid in figs, f"{fid} missing from {task}"


def test_qc_figures_gate_strips_them(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    res = pipe.generate_report(
        tmp_path, task="reconstruction", tables_=[], qc_figures=False, html_report=False
    )
    assert not any(k.startswith("qc_") for k in res["figures"])


def test_generate_report_emits_qc_html(tmp_path):
    import numpy as np

    from spectramr.infrastructure.reporting.cases.recorder import ReportCaseRecorder

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    # per-case metrics → group strip has one point per case
    (tmp_path / "per_call_metrics.csv").write_text(
        "case_id,split,step,psnr,ssim\n"
        + "\n".join(f"c{i},val,2,{30 + i},{0.8 + 0.01 * i}" for i in range(8))
        + "\n"
    )
    rec = ReportCaseRecorder(
        n_cases=3,
        selection="best_median_worst",
        primary_metric="psnr",
        higher_is_better=True,
    )
    for i in range(5):
        rec.observe(
            case_id=f"s{i}",
            arrays={
                "input": np.zeros((8, 8), np.float32),
                "prediction": np.full((8, 8), 0.1 * i, np.float32),
                "target": np.ones((8, 8), np.float32),
            },
            metrics={"psnr": 20.0 + i},
            domain={},
        )
    rec.write(tmp_path)
    res = pipe.generate_report(
        tmp_path,
        task="reconstruction",
        tables_=[],
        figures=["qc_group_strip", "qc_subject_mosaic", "qc_carpet"],
        formats=("pdf", "png"),
        html_report=True,
    )
    assert res["figures"].get("qc_group_strip") is not None
    assert res["html"] is not None and res["html"].exists()
    doc = res["html"].read_text()
    assert "QC report" in doc
    assert "data:image/png;base64," in doc


def test_generate_report_html_off(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_loss\n1,0.5\n2,0.3\n")
    res = pipe.generate_report(
        tmp_path,
        task="default",
        figures=["fig_1_2_learning_curves"],
        tables_=[],
        html_report=False,
    )
    assert res["html"] is None
    assert not (res["out_dir"] / "qc_report.html").exists()


def _seed_qc_run(tmp_path, *, with_volume=False):
    import numpy as np

    from spectramr.infrastructure.reporting.cases.recorder import ReportCaseRecorder

    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "training_metrics.csv").write_text(
        "step,loss,train_psnr,val_psnr\n"
        + "\n".join(f"{i*10},{1-i/20},{25+i/4},{24+i/4}" for i in range(20))
        + "\n"
    )
    (tmp_path / "per_call_metrics.csv").write_text(
        "case_id,split,step,psnr,ssim\n"
        + "\n".join(f"c{i},val,2,{30+i},{0.8+0.01*i}" for i in range(8))
        + "\n"
    )
    rec = ReportCaseRecorder(
        n_cases=3,
        selection="best_median_worst",
        primary_metric="psnr",
        higher_is_better=True,
        record_volumes=with_volume,
    )
    for i in range(4):
        arrays: dict = {
            "input": np.zeros((16, 16), np.float32),
            "prediction": np.full((16, 16), 0.1 * i, np.float32),
            "target": np.ones((16, 16), np.float32),
        }
        if with_volume and i == 0:
            arrays["prediction_volume"] = np.ones((6, 16, 16), np.float32)
            arrays["target_volume"] = np.zeros((6, 16, 16), np.float32)
        rec.observe(
            case_id=f"s{i}", arrays=arrays, metrics={"psnr": 20.0 + i}, domain={}
        )
    rec.write(tmp_path)


@pytest.mark.skipif(
    importlib.util.find_spec("plotly") is None, reason="plotly not installed"
)
def test_generate_report_wires_interactive_viewers(tmp_path):
    _seed_qc_run(tmp_path, with_volume=True)
    res = pipe.generate_report(
        tmp_path,
        task="reconstruction",
        tables_=[],
        figures=["qc_group_strip", "qc_subject_mosaic"],
        html_report=True,
        interactive=True,
    )
    doc = res["html"].read_text()
    assert "subject-2d" in doc and "volume-3d" in doc  # both viewers wired
    assert "Plotly.newPlot" in doc
    assert '<script src="https://cdn' not in doc  # offline self-contained


def test_generate_report_interactive_false_strips_layer(tmp_path):
    _seed_qc_run(tmp_path)
    res = pipe.generate_report(
        tmp_path,
        task="reconstruction",
        tables_=[],
        figures=["qc_group_strip", "qc_subject_mosaic"],
        html_report=True,
        interactive=False,
    )
    doc = res["html"].read_text()
    assert "Plotly.newPlot" not in doc  # no interactive layer
    assert "data:image/png;base64," in doc  # static PNGs embedded


def test_load_domain_artifacts_routes_csv_and_kwargs(tmp_path):
    """report_artifacts/ feeds the physics/geometry plotters: a per-figure CSV
    becomes that plotter's df; active_acquisition/*.csv -> csv_paths;
    qmap_slices.npz -> slices."""
    import numpy as np

    root = tmp_path / "report_artifacts"
    (root / "active_acquisition").mkdir(parents=True)
    (root / "fig_b3_bloch_consistency_residual.csv").write_text(
        "tissue,step,bloch_eq_residual\nwm,0,0.1\nwm,1,0.05\n"
    )
    (root / "active_acquisition" / "policy_a.csv").write_text(
        "step,psnr_db\n0,20\n1,22\n"
    )
    np.savez(
        root / "qmap_slices.npz", euclidean=np.zeros((4, 4)), riemannian=np.ones((4, 4))
    )

    df_over, dom = pipe._load_domain_artifacts(tmp_path)
    assert "fig_b3_bloch_consistency_residual" in df_over
    assert "csv_paths" in dom["fig_2_15_active_acquisition_trajectory"]
    assert set(dom["fig_b7_qmap_riemannian_vs_euclidean"]["slices"]) == {
        "euclidean",
        "riemannian",
    }


def test_load_domain_artifacts_absent_is_empty(tmp_path):
    df_over, dom = pipe._load_domain_artifacts(tmp_path)  # no report_artifacts/ dir
    assert df_over == {} and dom == {}


# ---------------------------------------------------------------------------
# #719: `figures: null` means THE TASK PRESET.
#
# It used to mean every registered plotter, while `ReportingSettings.figures`
# ("None = preset default") and docs/reporting_pipeline.rst both documented the
# preset. Three parties, and only the code said "all 44". Honouring the preset
# is also what makes `task:` load-bearing -- selecting a task changed nothing
# about the figure set before this.
# ---------------------------------------------------------------------------


class TestFiguresNullResolvesToThePreset:
    @staticmethod
    def _resolve(task: str, *, qc: bool = True, explicit=None):
        """Mirror of the resolution in ``generate_report`` (pre-dispatch)."""
        preset = pipe._resolve_preset(task)
        explicit_given = explicit is not None
        ids = list(explicit) if explicit_given else list(preset["figures"])
        if qc:
            if not explicit_given:
                ids += [
                    f
                    for f in pipe._all_figure_ids()
                    if f.startswith("qc_") and f not in ids
                ]
        else:
            ids = [f for f in ids if not f.startswith("qc_")]
        return pipe._contact_sheet_last(ids)

    @pytest.mark.parametrize(
        ("task", "expected"),
        [("default", 11), ("reconstruction", 31), ("gan", 10)],
    )
    def test_the_preset_not_all_44(self, task, expected):
        assert len(pipe._all_figure_ids()) == 69, "registry size changed; update below"
        assert len(self._resolve(task)) == expected

    def test_different_tasks_now_give_different_figure_sets(self):
        """The point of the fix: `task:` was inert for figures before it."""
        assert self._resolve("gan") != self._resolve("reconstruction")

    def test_an_explicit_list_still_wins(self):
        ids = self._resolve("reconstruction", explicit=["fig_1_1_architecture"])
        assert ids == ["fig_1_1_architecture"]


class TestQcFiguresIsAdditiveNotAFilter:
    """`qc_figures` promises to *include* the QC plotters.

    4 of the 8 presets (gan, diffusion, vae, calibration) list no ``qc_*`` at
    all, so the moment the preset became authoritative a pure filter would have
    left the knob advertising an inclusion it never performed -- pitfall #15,
    introduced by the very change that fixed #719.
    """

    _NO_QC_PRESETS = ["gan", "diffusion", "vae", "calibration"]

    @pytest.mark.parametrize("task", _NO_QC_PRESETS)
    def test_a_preset_without_qc_still_gets_qc_when_enabled(self, task):
        ids = TestFiguresNullResolvesToThePreset._resolve(task, qc=True)
        assert [
            f for f in ids if f.startswith("qc_")
        ], f"{task}: qc_figures=True produced no QC plotters"

    @pytest.mark.parametrize("task", _NO_QC_PRESETS + ["default", "reconstruction"])
    def test_disabling_removes_them_everywhere(self, task):
        ids = TestFiguresNullResolvesToThePreset._resolve(task, qc=False)
        assert not [f for f in ids if f.startswith("qc_")]

    def test_the_union_does_not_duplicate(self):
        """`default` already lists all three; unioning must not double them."""
        ids = TestFiguresNullResolvesToThePreset._resolve("default", qc=True)
        assert len(ids) == len(set(ids))

    def test_an_explicit_list_is_not_topped_up(self):
        """Explicit means explicit -- the union applies only to the preset path."""
        ids = TestFiguresNullResolvesToThePreset._resolve(
            "gan", qc=True, explicit=["fig_1_1_architecture"]
        )
        assert ids == ["fig_1_1_architecture"]


class TestContactSheetStaysLast:
    """It composites the other emitted PNGs, so it must render after them."""

    @pytest.mark.parametrize("task", list(pipe.TASK_PRESETS))
    def test_after_the_qc_union(self, task):
        ids = TestFiguresNullResolvesToThePreset._resolve(task, qc=True)
        if "contact_sheet" in ids:
            assert ids[-1] == "contact_sheet"

    def test_the_helper_moves_it_from_any_position(self):
        assert pipe._contact_sheet_last(["contact_sheet", "a", "b"]) == [
            "a",
            "b",
            "contact_sheet",
        ]

    def test_the_helper_is_a_no_op_without_it(self):
        assert pipe._contact_sheet_last(["a", "b"]) == ["a", "b"]


class TestSkippedFiguresAreAccountedForNotDropped:
    """A report drawn after a prediction run lacks the training-only inputs.

    The *requested* set is identical either way -- `fig_ids` derives only from
    `task`/`figures`/`qc_figures`, never from the caller -- so the only thing
    that differs between a post-train and a post-predict report is which
    figures found data. Recording that as a reason turns "the two reports
    differ" into an auditable statement instead of a silent gap (pitfall #16),
    and it is what lets a reviewer check the union criterion:
    ``{figures after training} <= {figures after predict} | {skipped-with-reason}``.
    """

    @staticmethod
    def _figures(res):
        import json

        man = json.loads((Path(res["out_dir"]) / "report_manifest.json").read_text())
        return [a for a in man["artifacts"] if a["kind"] == "figure"]

    @staticmethod
    def _seed_predict_like(tmp_path):
        """What `infer` writes -- no training_metrics.csv, so most figures skip."""
        (tmp_path / "final_eval.json").write_text('{"mae": 0.1}')
        (tmp_path / "run_summary.json").write_text(
            '{"model_params": 100, "duration_sec": 1.0, "effective_batch": 1}'
        )

    def test_no_skipped_figure_is_reasonless(self, tmp_path):
        self._seed_predict_like(tmp_path)
        res = pipe.generate_report(tmp_path, tables_=[], html_report=False)
        skipped = [a for a in self._figures(res) if a["status"] == "skipped"]
        assert skipped, "anti-vacuity: a predict-like dir must skip something"
        assert [a["id"] for a in skipped if not a.get("reason")] == []

    def test_reasons_are_attached_selectively_not_blanket(self, tmp_path):
        """Both halves in one manifest, because either alone is satisfiable.

        "no emitted figure has a reason" is vacuously true when nothing has one
        (i.e. on the unfixed code this test guards), so it only means something
        alongside a sibling entry that DOES carry one.
        """
        self._seed_predict_like(tmp_path)
        res = pipe.generate_report(tmp_path, tables_=[], html_report=False)
        figs = self._figures(res)
        ok = [a for a in figs if a["status"] == "ok"]
        skipped = [a for a in figs if a["status"] == "skipped"]

        assert ok and skipped, "anti-vacuity: need both outcomes in one manifest"
        assert [a["id"] for a in ok if a.get("reason")] == []
        assert [a["id"] for a in skipped if a.get("reason")] != []

    def test_the_reason_names_the_cause_it_does_not_flatten_to_one_word(
        self, tmp_path, monkeypatch
    ):
        """A plotter that RAISED must not be reported as 'had no data'.

        Conflating the two is the failure this whole change exists to prevent:
        a crashing figure would read as an expected absence on the predict path.
        """
        from spectramr.infrastructure.reporting.plotters import (
            SKIP_NO_DATA,
            SKIP_RAISED,
            FigureOutcome,
        )

        def _boom(df, fig_ids, out_dir, **kwargs):
            return {f: FigureOutcome(f, None, SKIP_RAISED) for f in fig_ids}

        monkeypatch.setattr(pipe.plotters, "dispatch_detailed", _boom)
        self._seed_predict_like(tmp_path)
        res = pipe.generate_report(tmp_path, tables_=[], html_report=False)
        reasons = {a.get("reason") for a in self._figures(res)}

        assert reasons == {SKIP_RAISED}
        assert SKIP_NO_DATA not in reasons

def test_extra_runs_stay_out_of_the_main_runs_summary(tmp_path):
    def run(path, psnr):
        (path / "logs").mkdir(parents=True)
        (path / "logs" / "validation_metrics.csv").write_text(
            f"step,val_psnr\n100,{psnr - 1}\n200,{psnr}\n"
        )
        return path

    main = run(tmp_path / "main", 20.0)
    extra = run(tmp_path / "extra", 99.0)
    res = pipe.generate_report(main, figures=[], tables_=[], extra_runs=[extra],
                               html_report=False, qc_figures=False)
    summary = res["summary"].read_text()
    assert "| validation PSNR (dB) | 20 |" in summary and "| 99 |" not in summary


@pytest.mark.parametrize(
    ("summary", "status", "label"),
    [
        ({"success": False, "error_type": "RuntimeError"}, "failed", "crashed: RuntimeError"),
        (None, "unknown", "status unknown: no run_summary.json"),
        ({"success": True}, "ok", "ok"),
    ],
)
def test_a_per_run_report_states_whether_the_run_finished(tmp_path, summary, status, label):
    """#2525: a crashed run is still reported, with its status in summary and manifest."""
    import json

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "validation_metrics.csv").write_text("step,val_psnr\n1,20.0\n2,21.0\n")
    if summary is not None:
        (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    res = pipe.generate_report(
        tmp_path,
        figures=[],
        tables_=["tab_2_1_main_results"],
        html_report=False,
        qc_figures=False,
    )
    assert f"- Run status: `{label}`" in res["summary"].read_text()
    manifest = json.loads((res["out_dir"] / "report_manifest.json").read_text())
    assert [r["status"] for r in manifest["runs"]] == [status]
    table = res["tables"]["tab_2_1_main_results"]["markdown"].read_text()
    assert ("‡" in table) is (status != "ok")
