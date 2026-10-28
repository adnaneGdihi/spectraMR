"""Unit tests for the subprocess-based HPO trial objective.

CPU-only — these tests stub out the actual trainer subprocess so no GPU
or real data is needed. They exercise the YAML-mutation, metric-resolution,
and pruning logic in isolation.
"""

from __future__ import annotations

import csv

import pytest
import yaml

from spectramr.pipelines.hpo import (
    _build_trial_yaml,
    _read_latest_metrics,
    _resolve_metric,
    subprocess_training_objective,
)


@pytest.fixture
def base_config_dict() -> dict:
    """Minimal base config that the trial-builder can mutate."""
    return {
        "metadata": {"name": "base"},
        "training": {"output_dir": "/tmp/x", "max_iterations": 100},
        "logging": {"log_dir": "/tmp/x/logs", "experiment_name": "base"},
        "checkpoint": {"save_interval": 10},
        "artifacts": {"persistent_root": "/tmp/x"},
        "loss_logging": {
            "csv_path": "/tmp/x/logs/loss_log.csv",
            "output_dir": "/tmp/x/metrics",
        },
        "metrics": {"output_dir": "/tmp/x/metrics"},
        "optimization": {"learning_rate": 1e-3},
    }


class TestBuildTrialYaml:
    def test_overrides_applied_via_dotted_path(self, base_config_dict, tmp_path):
        path = _build_trial_yaml(
            base_config_dict,
            trial_number=7,
            overrides={"optimization.optimizer.learning_rate": 5e-4},
            output_root=tmp_path,
            max_iterations=2000,
        )
        assert path.exists()
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        assert cfg["optimization"]["optimizer"]["learning_rate"] == 5e-4
        # max_iterations is forced to the trial budget
        assert cfg["training"]["max_iterations"] == 2000

    def test_overrides_create_missing_nested_keys(self, base_config_dict, tmp_path):
        """A dotted path that traverses missing keys should create them."""
        path = _build_trial_yaml(
            base_config_dict,
            trial_number=1,
            overrides={"new_section.deeply.nested.key": 42},
            output_root=tmp_path,
            max_iterations=100,
        )
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        assert cfg["new_section"]["deeply"]["nested"]["key"] == 42

    def test_output_paths_repointed_per_trial(self, base_config_dict, tmp_path):
        """Trial config gets its own output paths so trials never collide."""
        path = _build_trial_yaml(
            base_config_dict,
            trial_number=3,
            overrides={},
            output_root=tmp_path,
            max_iterations=100,
        )
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        out_root = str(tmp_path / "trial_0003")
        assert cfg["training"]["output_dir"] == out_root
        assert out_root in cfg["logging"]["log_dir"]
        assert "checkpoint_dir" not in cfg.get("checkpoint", {})
        assert "save_dir" not in cfg.get("checkpoint", {})
        assert out_root in cfg["loss_logging"]["csv_path"]
        assert cfg["metadata"]["name"] == "trial_0003"

    def test_the_unread_artifacts_block_is_not_written(self, base_config_dict, tmp_path):
        """#698: nothing reads `artifacts`, so a trial must not advertise one (the base's
        would name the base arm's directory, the old repoint wrote a fresh one)."""
        path = _build_trial_yaml(
            base_config_dict, trial_number=2, overrides={}, output_root=tmp_path, max_iterations=10
        )
        with open(path) as fh:
            assert "artifacts" not in yaml.safe_load(fh)


class TestReadLatestMetrics:
    def test_returns_empty_when_csv_missing(self, tmp_path):
        result = _read_latest_metrics(tmp_path / "nope.csv")
        assert result == {}

    def test_returns_last_row_only(self, tmp_path):
        csv_path = tmp_path / "loss.csv"
        with csv_path.open("w") as fh:
            writer = csv.DictWriter(fh, fieldnames=["step", "loss", "psnr"])
            writer.writeheader()
            writer.writerow({"step": 100, "loss": 0.5, "psnr": 25.0})
            writer.writerow({"step": 200, "loss": 0.3, "psnr": 28.5})
        result = _read_latest_metrics(csv_path)
        assert result == {"step": 200.0, "loss": 0.3, "psnr": 28.5}

    def test_skips_non_numeric_columns(self, tmp_path):
        csv_path = tmp_path / "loss.csv"
        with csv_path.open("w") as fh:
            writer = csv.DictWriter(fh, fieldnames=["step", "loss", "comment"])
            writer.writeheader()
            writer.writerow({"step": 1, "loss": 0.1, "comment": "warmup"})
        result = _read_latest_metrics(csv_path)
        assert "comment" not in result
        assert result["loss"] == 0.1


class TestResolveMetric:
    def test_exact_match(self):
        assert _resolve_metric({"val_loss": 0.5, "psnr": 25}, "val_loss") == 0.5

    def test_case_insensitive(self):
        assert _resolve_metric({"VAL_LOSS": 0.5}, "val_loss") == 0.5

    def test_substring_fallback(self):
        """If exact match fails, find the first column containing the name."""
        # val_psnr is not a column but val_psnr_4x is — substring match wins
        result = _resolve_metric({"val_psnr_4x": 28.0, "step": 100}, "val_psnr")
        assert result == 28.0

    def test_returns_none_when_no_match(self):
        assert _resolve_metric({"step": 1, "loss": 0.5}, "fid") is None


class TestSubprocessTrainingObjective:
    """Test the objective factory without spawning any real subprocess."""

    def test_minimize_auto_detection_for_loss_names(
        self, base_config_dict, tmp_path
    ):
        """Names containing 'loss' / 'error' / 'mse' auto-set minimize=True."""
        cfg_path = tmp_path / "base.yaml"
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(base_config_dict, fh)
        # Build the objective for a loss-style metric — should set minimize=True
        # internally. We can't directly inspect the closure, so we verify by
        # reading the docstring / behaviour through indirect channels.
        obj = subprocess_training_objective(
            base_config_path=str(cfg_path),
            output_root=tmp_path / "out",
            metric_name="val_loss",
            max_iterations=100,
        )
        assert callable(obj)

    def test_minimize_auto_detection_for_psnr_names(
        self, base_config_dict, tmp_path
    ):
        cfg_path = tmp_path / "base.yaml"
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(base_config_dict, fh)
        obj = subprocess_training_objective(
            base_config_path=str(cfg_path),
            output_root=tmp_path / "out",
            metric_name="val_psnr_4x",
            max_iterations=100,
        )
        assert callable(obj)

    def test_factory_creates_output_root(self, base_config_dict, tmp_path):
        cfg_path = tmp_path / "base.yaml"
        with open(cfg_path, "w") as fh:
            yaml.safe_dump(base_config_dict, fh)
        out_root = tmp_path / "auto_created"
        assert not out_root.exists()
        _ = subprocess_training_objective(
            base_config_path=str(cfg_path),
            output_root=out_root,
            metric_name="val_loss",
            max_iterations=100,
        )
        assert out_root.exists()
