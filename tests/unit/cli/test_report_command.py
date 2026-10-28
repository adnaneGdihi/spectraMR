"""Tests for the ``spectramr report`` handler (``spectramr.cli.app.report``).

Covers the --config reporting-block parity path and CLI-override precedence
without loading a full TrainingSettings YAML (stubbed) or rendering figures
(generate_report stubbed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spectramr.cli import app
from spectramr.config.schemas.logging import LoggingConfigSchema


class _Rep:
    task = "reconstruction"
    style = "nature"
    formats = ["pdf", "png"]
    dpi = 300
    panel_labels = True
    method_name = "cfgmethod"
    figures = None
    tables = None
    metrics = None
    hyperparameters = None
    extra_runs = None
    out_subdir = "report"
    emit_manifest = True
    submission_bundle = False
    tikz = False
    qc_figures = False
    html_report = True
    cohort = None


class _Run:
    seed = 11


class _Settings:
    logging = LoggingConfigSchema()
    reporting = _Rep()
    run = _Run()
    # The RETIRED top-level spelling, kept as a decoy. `RENAMES` moved `seed`
    # to `run.seed` with a raise posture, so a real `TrainingSettings` cannot
    # carry this attribute at all -- which is why the old pin below (`== 7`,
    # "falls back to settings.seed") passed on the double while `report`
    # stamped `seed: null` on every real run. The double asserted a schema
    # that no longer exists.
    seed = 7


def _ns(exp_dir, **over):
    base = dict(exp_dir=exp_dir, config=None, task=None, method=None,
                out_subdir=None, seed=None, dataset_version=None,
                cohort_json=None, html=None, qc=None, recursive=False)
    base.update(over)
    return argparse.Namespace(**base)


def _patch(monkeypatch):
    captured: dict = {}

    def _fake_gen(exp_dir, **kw):
        captured.update(kw)
        captured["exp_dir"] = exp_dir
        return {"figures": {}, "tables": {}, "out_dir": Path(exp_dir),
                "summary": Path(exp_dir) / "s.md", "html": None}

    monkeypatch.setattr("spectramr.infrastructure.reporting.generate_report", _fake_gen)
    monkeypatch.setattr("spectramr.config.settings.TrainingSettings.from_yaml",
                        lambda p: _Settings())
    return captured


def test_missing_dir_returns_2(tmp_path):
    assert app.report(_ns(tmp_path / "nope")) == 2


def test_config_block_is_forwarded(tmp_path, monkeypatch):
    captured = _patch(monkeypatch)
    exp = tmp_path / "run"; exp.mkdir()
    rc = app.report(_ns(exp, config=tmp_path / "c.yaml"))
    assert rc == 0
    assert captured["task"] == "reconstruction"
    assert captured["method_name"] == "cfgmethod"
    assert captured["dpi"] == 300
    assert captured["qc_figures"] is False
    assert captured["html_report"] is True
    assert captured["seed"] == 11, "the seed comes from run.seed"
    assert captured["seed"] != 7, "the retired top-level spelling must not be read"


def test_cli_flags_override_config(tmp_path, monkeypatch):
    captured = _patch(monkeypatch)
    exp = tmp_path / "run"; exp.mkdir()
    rc = app.report(_ns(exp, config=tmp_path / "c.yaml", task="synthesis",
                        html=False, qc=True, method="cli_method"))
    assert rc == 0
    assert captured["task"] == "synthesis"
    assert captured["html_report"] is False
    assert captured["qc_figures"] is True
    assert captured["method_name"] == "cli_method"


def test_no_config_uses_cli_defaults(tmp_path, monkeypatch):
    captured = _patch(monkeypatch)
    exp = tmp_path / "run"; exp.mkdir()
    rc = app.report(_ns(exp))  # no --config
    assert rc == 0
    assert captured["task"] == "default"
    assert captured["method_name"] == "run"  # exp_dir.name fallback
    assert captured["out_subdir"] == "report"


def test_recursive_dispatches_to_batch(tmp_path, monkeypatch):
    exp = tmp_path / "cohort"; exp.mkdir()
    seen: dict = {}

    def _fake_batch(root, **kw):
        seen["root"] = root
        seen["kw"] = kw
        return {"runs": [], "index": None, "n_runs": 2, "n_ok": 2}

    monkeypatch.setattr("spectramr.infrastructure.reporting.generate_reports", _fake_batch)
    rc = app.report(_ns(exp, recursive=True))
    assert rc == 0
    assert seen["root"] == exp
    # method_name must NOT be forwarded (batch names each run itself)
    assert "method_name" not in seen["kw"]


def test_recursive_no_runs_found(tmp_path, monkeypatch):
    exp = tmp_path / "empty"; exp.mkdir()
    monkeypatch.setattr("spectramr.infrastructure.reporting.generate_reports",
                        lambda root, **kw: {"runs": [], "index": None,
                                            "n_runs": 0, "n_ok": 0})
    assert app.report(_ns(exp, recursive=True)) == 0


# ---- contrast folders and the all-metrics flags ------------------------------------


def _cohort(root):
    from tests.unit.infrastructure.reporting.comparison._runs import (
        add_inference_results,
        add_slice_rows,
        make_run,
    )

    for seed, (name, shift) in enumerate((("arm_base", 0.0), ("arm_better", 1.0))):
        run = make_run(root, name, shift=shift)
        add_slice_rows(run, offset=shift, seed=seed)
        add_inference_results(run, offset=shift)
    return root


def _contrast_ns(**over):
    base = {
        "exp_dir": None, "contrasts": None, "contrast_pairs": None, "out_root": None,
        "only": None, "results_root": None, "baseline": None, "view": "selected",
        "style": None, "html": False, "metrics": None, "phases": None,
        "max_panels": 12, "train_window": 50,
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_exp_dir_is_required_without_contrasts(capsys):
    from spectramr.cli.app import report

    assert report(_ns(None)) == 2
    assert "--exp-dir is required" in capsys.readouterr().out


def test_contrast_pairs_write_one_folder_per_arm_for_any_cohort(tmp_path, capsys):
    from spectramr.cli.app import report

    cohort = _cohort(tmp_path / "cohort")
    code = report(
        _contrast_ns(exp_dir=cohort, contrast_pairs="vs_baseline", baseline="arm_base",
                     phases="val,test")
    )
    assert code == 0
    folder = cohort / "contrasts" / "arm_better__vs__arm_base"
    assert (folder / "data" / "paired_stats.csv").is_file()
    assert (folder / "data" / "metrics_test.csv").is_file()
    assert not (folder / "data" / "metrics_train.csv").exists()
    assert "arm_better__vs__arm_base" in capsys.readouterr().out


def test_contrasts_and_contrast_pairs_together_are_refused(tmp_path, capsys):
    from spectramr.cli.app import report

    code = report(_contrast_ns(contrasts=tmp_path / "x.yaml", contrast_pairs="pairs"))
    assert code == 2 and "not both" in capsys.readouterr().out


def test_a_contrast_file_needs_an_out_root(tmp_path, capsys):
    import yaml

    from spectramr.cli.app import report

    spec = tmp_path / "set.yaml"
    spec.write_text(yaml.safe_dump({"contrasts": [{"name": "c", "arms": ["a", "b"]}]}))
    assert report(_contrast_ns(contrasts=spec)) == 2
    assert "--out-root" in capsys.readouterr().out


def test_metric_flags_become_keyword_arguments():
    from spectramr.cli.app import _metric_flags

    assert _metric_flags(_contrast_ns(metrics="all", phases="val, test")) == {
        "metrics": "all", "phases": ("val", "test"), "max_panels": 12, "train_window": 50,
    }
    assert _metric_flags(_contrast_ns(metrics="psnr, cjv"))["metrics"] == ["psnr", "cjv"]
    assert "metrics" not in _metric_flags(_contrast_ns())
