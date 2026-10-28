"""``resolve_inference_settings``: the run's resolved config beside the checkpoint wins (#1379)."""

import pytest

from spectramr.infrastructure.validation.inference_settings import (
    _describe_inference_source,
    resolve_inference_settings,
)


class TestResolveInferenceSettings:
    @staticmethod
    def _reference_yaml():
        import pathlib

        import spectramr.config.schemas as _schemas

        return pathlib.Path(_schemas.__file__).parent / "templates" / "v1.0_reference.yaml"

    @staticmethod
    def _run_dir_with_artifact(tmp_path, yaml_path):
        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.validation.resolved_config_artifact import (
            write_resolved_config,
        )

        run = tmp_path / "run"
        run.mkdir()
        write_resolved_config(run, TrainingSettings.from_yaml(str(yaml_path)), run_id="t")
        ckpt = run / "best.pt"
        ckpt.write_bytes(b"")
        return ckpt

    def test_artifact_beside_the_checkpoint_wins_over_the_yaml(self, tmp_path):
        yaml_path = self._reference_yaml()
        ckpt = self._run_dir_with_artifact(tmp_path, yaml_path)
        settings, source = resolve_inference_settings(yaml_path, ckpt)
        assert source["kind"] == "resolved_config" and source["path"].endswith(
            "resolved_config.json"
        )
        assert "diverging_blocks" not in source
        assert settings.model.model_type

    def test_from_yaml_forces_the_yaml(self, tmp_path):
        yaml_path = self._reference_yaml()
        ckpt = self._run_dir_with_artifact(tmp_path, yaml_path)
        _, source = resolve_inference_settings(yaml_path, ckpt, from_yaml=True)
        assert source["kind"] == "yaml" and source["resolved_config"] is not None

    def test_neither_source_raises_with_guidance(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"")
        with pytest.raises(FileNotFoundError, match=r"resolved_config\.json"):
            resolve_inference_settings(None, ckpt)
        with pytest.raises(FileNotFoundError, match="--from-yaml was set"):
            resolve_inference_settings(None, ckpt, from_yaml=True)

    def test_a_disagreeing_yaml_is_reported_not_used(self, tmp_path, caplog):
        import logging

        import yaml as _yaml

        yaml_path = self._reference_yaml()
        ckpt = self._run_dir_with_artifact(tmp_path, yaml_path)
        doc = _yaml.safe_load(yaml_path.read_text())
        doc["optimization"]["optimizer"]["learning_rate"] = 0.31337
        other = tmp_path / "other.yaml"
        other.write_text(_yaml.safe_dump(doc, sort_keys=False))
        with caplog.at_level(logging.WARNING):
            settings, source = resolve_inference_settings(other, ckpt)
        assert source["kind"] == "resolved_config"
        assert "optimization" in source["diverging_blocks"]
        assert settings.optimization.optimizer.learning_rate != 0.31337
        assert any("disagree" in r.message for r in caplog.records)

    def test_an_artifact_without_the_declared_block_yields_to_the_yaml(self, tmp_path, caplog):
        """Every run directory written before the block: the documented
        ``infer --config`` command keeps working, and the source says why."""
        import json
        import logging

        run = tmp_path / "run"
        run.mkdir()
        (run / "resolved_config.json").write_text(
            json.dumps({"model": {"model_type": "unet"}, "_ledger": {}})
        )
        ckpt = run / "best.pt"
        ckpt.write_bytes(b"")
        yaml_path = self._reference_yaml()
        with caplog.at_level(logging.WARNING):
            settings, source = resolve_inference_settings(yaml_path, ckpt)
        assert source["kind"] == "yaml" and source["resolved_config_predates_declared"] is True
        assert settings.model.model_type
        assert any("predates" in r.message for r in caplog.records)
        assert (
            _describe_inference_source(yaml_path, ckpt, False)["resolved_config_predates_declared"]
            is True
        )
        with pytest.raises(FileNotFoundError, match="predates"):
            resolve_inference_settings(None, ckpt)
