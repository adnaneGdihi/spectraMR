"""A multi-stage run's best restore, and the stage-checkpoint refusals (#2962, #2964)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.checkpointing import codecs, store
from spectramr.infrastructure.checkpointing.layout import CheckpointLayout
from spectramr.infrastructure.checkpointing.restore import best_weights
from tests.unit.infrastructure.checkpointing.conftest import make_pipeline
from tests.unit.infrastructure.training.strategies.pipeline_stage_fakes import (
    _config,
    _equal,
    _resume,
    _save,
    _strategy,
    _train_step,
    _trained,
)


def _best(tmp_path: Path, pipeline, s) -> Path:
    return store.save_best(
        pipeline,
        layout=CheckpointLayout.for_run(tmp_path),
        config=_config("pt"),
        strategy=s,
        parallel=None,
        early_stopping=None,
        epoch=3,
        step=12,
        metric_name="val_psnr",
        metric_value=30.0,
    )


def test_restore_best_puts_the_stage_weights_back(tmp_path: Path) -> None:
    pipeline, s = _trained()
    path = _best(tmp_path, pipeline, s)
    best = {n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in s.multi_stages.items()}
    _train_step(pipeline, s)
    best_weights(path, pipeline, parallel=None, strategy=s)
    for name, module in s.multi_stages.items():
        assert all(torch.equal(module.state_dict()[k], v) for k, v in best[name].items()), name


def test_a_refused_best_restore_moves_no_module(tmp_path: Path) -> None:
    pipeline, s = _trained()
    path = _best(tmp_path, pipeline, s)
    state = codecs.read(path)
    del state["models"]["stage:b"]
    codecs.write(path, state)
    _train_step(pipeline, s)
    before = {
        n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in s.multi_stages.items()
    }
    before["generator"] = {k: v.clone() for k, v in pipeline.generator.state_dict().items()}
    with pytest.raises(ValueError, match="#2962"):
        best_weights(path, pipeline, parallel=None, strategy=s)
    live = {n: m.state_dict() for n, m in s.multi_stages.items()}
    live["generator"] = pipeline.generator.state_dict()
    assert all(torch.equal(live[n][k], v) for n, sd in before.items() for k, v in sd.items())


def test_load_optimizer_state_off_leaves_the_stage_optimizers_fresh(tmp_path: Path) -> None:
    """The gate covers every optimizer, the stage ones included (#2964)."""
    pipeline, s = _trained()
    path = _save(tmp_path, pipeline, s)
    fresh_pipeline = make_pipeline(discriminator=False, ema=False)
    fresh = _strategy(fresh_pipeline, seed=2)
    _resume(path, fresh_pipeline, fresh, load_optimizer_state=False)
    assert fresh._stage_optimizers["b"].state_dict()["state"] == {}
    assert _equal(fresh.multi_stages["b"], s.multi_stages["b"])


def test_a_file_with_stages_but_no_strategy_part_is_refused(tmp_path: Path) -> None:
    """Without it the stage optimizers and early stopping would silently restart."""
    pipeline, s = _trained()
    path = _save(tmp_path, pipeline, s)
    state = codecs.read(path)
    del state["strategy"]
    codecs.write(path, state)
    fresh_pipeline = make_pipeline(discriminator=False, ema=False)
    with pytest.raises(RuntimeError, match=r"'pipeline'.*#2962"):
        _resume(path, fresh_pipeline, _strategy(fresh_pipeline, seed=2))


def _drop_pipeline_entry(state: dict) -> None:
    del state["strategy"]["pipeline"]


def _drop_stage_optimizer(state: dict) -> None:
    state["strategy"]["pipeline"]["stage_optimizers"] = {}


@pytest.mark.parametrize("corrupt", [_drop_pipeline_entry, _drop_stage_optimizer])
def test_a_refused_resume_moves_no_stage(tmp_path: Path, corrupt) -> None:
    pipeline, s = _trained()
    path = _save(tmp_path, pipeline, s)
    state = codecs.read(path)
    corrupt(state)
    codecs.write(path, state)
    fresh_pipeline = make_pipeline(discriminator=False, ema=False)
    fresh = _strategy(fresh_pipeline, seed=2)
    before = {
        n: {k: v.clone() for k, v in m.state_dict().items()} for n, m in fresh.multi_stages.items()
    }
    with pytest.raises(RuntimeError, match="#2962"):
        _resume(path, fresh_pipeline, fresh)
    for name, module in fresh.multi_stages.items():
        assert all(torch.equal(module.state_dict()[k], v) for k, v in before[name].items()), name
