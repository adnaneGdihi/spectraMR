"""Reachability: the experiment_96 YAML drives the post-training compression export.

Production path: TrainingSettings (arm YAML) -> ``_maybe_run_compression_export`` ->
ModelCompressionExportService -> ModelCompressionPipeline + metric registry ->
``TrainingLoop.evaluate(record=False)`` -> ``_run_validation`` (patched to forward the live
generator on a tiny tensor).
"""

import ast
import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import spectramr.pipelines.train as train_mod
import spectramr.pipelines.training_loop as training_loop_mod
from spectramr.config.settings import TrainingSettings
from spectramr.pipelines.train import _maybe_run_compression_export, _refuse_compression_with_ema
from spectramr.pipelines.training_loop import TrainingLoop

ARM = (
    Path(__file__).resolve().parents[3]
    / "experiments/inprogress/efficiency/experiment_96_model_pruning_&_quantization.yaml"
)


def _setup(monkeypatch):
    torch.manual_seed(0)
    gen = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 3, padding=1))
    x = torch.randn(2, 1, 8, 8)
    target = gen(x).detach()
    seen: list[dict] = []

    def fake_validation(pipeline, strategy, iteration, epoch, logging_service, **kw):
        seen.append(kw)
        with torch.no_grad():
            mse = ((pipeline.generator(x) - target) ** 2).mean()
        return {"val_mse": float(mse)}

    monkeypatch.setattr(training_loop_mod, "_run_validation", fake_validation)
    pipeline = MagicMock()
    pipeline.generator = gen
    pipeline.ema = None
    loop = TrainingLoop("strat", pipeline, "cfg", "efficient_unet", metrics_service="met")
    return gen, pipeline, loop, seen


def test_arm_yaml_fires_the_compression_export(monkeypatch, tmp_path):
    config = TrainingSettings.from_yaml(str(ARM))
    gen, pipeline, loop, seen = _setup(monkeypatch)
    before = {k: v.clone() for k, v in gen.state_dict().items()}
    result = {"success": True}

    _maybe_run_compression_export(
        loop, pipeline, config, {"run_output_dir": str(tmp_path)}, result, is_sanity_check=False
    )

    report = json.loads((tmp_path / "compression_report.json").read_text())
    assert report["applied"]["weight_bits"] == 8
    assert report["weight_sparsity"] == pytest.approx(0.3, abs=0.02)
    assert report["val_metrics_fp32"]["val_mse"] == 0.0
    assert report["val_metrics_compressed"]["val_mse"] > 0.0  # the compressed weights were scored
    assert 3.0 < report["compression_ratio"] < 4.0
    assert result["compression"]["compression_ratio"] == report["compression_ratio"]
    assert [kw["metrics_service"] for kw in seen] == [None, None]  # never into the val curves
    for k, v in gen.state_dict().items():
        assert torch.equal(v, before[k])  # fp32 weights restored


def test_ema_is_refused(monkeypatch, tmp_path):
    config = TrainingSettings.from_yaml(str(ARM))
    _, pipeline, loop, _ = _setup(monkeypatch)
    pipeline.ema = object()
    with pytest.raises(ValueError, match="EMA"):
        _maybe_run_compression_export(
            loop,
            pipeline,
            config,
            {"run_output_dir": str(tmp_path)},
            {"success": True},
            is_sanity_check=False,
        )


def test_skipped_without_block_or_on_yield(monkeypatch, tmp_path):
    config = TrainingSettings.from_yaml(str(ARM))
    _, pipeline, loop, seen = _setup(monkeypatch)
    _maybe_run_compression_export(
        loop,
        pipeline,
        config,
        {"run_output_dir": str(tmp_path)},
        {"wall_clock_yield": True},
        is_sanity_check=False,
    )
    assert not seen and not (tmp_path / "compression_report.json").exists()


def test_launch_refuses_compression_with_ema():
    config = TrainingSettings.from_yaml(str(ARM))
    _refuse_compression_with_ema(config)  # ema off: accepted
    with_ema = config.model_copy(update={"ema": config.ema.model_copy(update={"enabled": True})})
    with pytest.raises(ValueError, match=r"ema\.enabled"):
        _refuse_compression_with_ema(with_ema)


def test_pipeline_calls_the_export_after_the_test_pass_and_before_the_summary():
    """The production call site: after the held-out test pass, before run_summary.json."""
    src = inspect.getsource(train_mod.run_training_pipeline)
    calls = [
        (node.lineno, node.func.id)
        for node in ast.walk(ast.parse(src.lstrip()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    first = {}
    for line, name in sorted(calls):
        first.setdefault(name, line)
    last_test_pass = max(line for line, name in calls if name == "_run_held_out_test_pass")
    assert "_refuse_compression_with_ema" in first
    assert last_test_pass < first["_maybe_run_compression_export"] < first["_emit_run_summary"]
