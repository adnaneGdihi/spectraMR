"""A multi-stage run's stages, stage optimizers and stage early stopping round-trip (#2962).

The fakes live in ``pipeline_stage_fakes``; the best restore and the refusals are in
``test_pipeline_stage_checkpoint_guards``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.restore import weights
from tests.unit.infrastructure.checkpointing.conftest import make_pipeline
from tests.unit.infrastructure.training.strategies.pipeline_stage_fakes import (
    FORMATS,
    _equal,
    _resume,
    _save,
    _strategy,
    _train_step,
    _trained,
)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_step_file_carries_every_stage_and_the_pipeline_state(tmp_path: Path, fmt) -> None:
    pipeline, s = _trained()
    s.on_epoch_end(0, {"val_a_l1": 0.5})
    state = codecs.read(_save(tmp_path, pipeline, s, fmt=fmt))
    assert {"stage:a", "stage:b"} <= set(state["models"])
    entry = state["strategy"]["pipeline"]
    assert set(entry["stage_optimizers"]) == {"b"}
    assert entry["stage_optimizers"]["b"]["state"], "the stage optimizer's moments are missing"
    assert (entry["es_best"], entry["es_wait"]) == ({"a": 0.5}, {"a": 0})
    assert entry["frozen_by_early_stopping"] == []
    assert entry["all_unfrozen"] is False


@pytest.mark.parametrize("fmt", FORMATS)
def test_resume_restores_stage_weights_optimizer_and_early_stopping(tmp_path: Path, fmt) -> None:
    pipeline, s = _trained()
    s.on_epoch_end(0, {"val_a_l1": 0.5})
    s.on_epoch_end(1, {"val_a_l1": 0.25})
    path = _save(tmp_path, pipeline, s, fmt=fmt)

    fresh_pipeline = make_pipeline(discriminator=False, ema=False)
    fresh = _strategy(fresh_pipeline, seed=2)
    assert not _equal(fresh.multi_stages["a"], s.multi_stages["a"])
    _resume(path, fresh_pipeline, fresh, fmt=fmt)

    for name in ("a", "b"):
        assert _equal(fresh.multi_stages[name], s.multi_stages[name]), name
    saved_b = s._stage_optimizers["b"].state_dict()["state"]
    live_b = fresh._stage_optimizers["b"].state_dict()["state"]
    assert all(torch.equal(saved_b[i]["exp_avg"], live_b[i]["exp_avg"]) for i in saved_b)
    assert (fresh._es_best, fresh._es_wait) == ({"a": 0.25}, {"a": 0})


def test_resume_after_the_e2e_unfreeze_reproduces_opt_g_param_groups(tmp_path: Path) -> None:
    """Stage ``a`` frozen at init joins ``opt_g`` at the unfreeze; a fresh run must add
    that group before ``opt_g``'s state is loaded, or the load raises on a group count."""
    pipeline, s = _trained(frozen_at_init=("a",))
    s.on_epoch_start(2)
    _train_step(pipeline, s)
    groups = len(pipeline.optimizer_g.param_groups)
    path = _save(tmp_path, pipeline, s)

    fresh_pipeline = make_pipeline(discriminator=False, ema=False)
    fresh = _strategy(fresh_pipeline, seed=2, frozen_at_init=("a",))
    assert len(fresh_pipeline.optimizer_g.param_groups) < groups
    _resume(path, fresh_pipeline, fresh)

    assert len(fresh_pipeline.optimizer_g.param_groups) == groups
    assert fresh._all_unfrozen
    assert all(p.requires_grad for p in fresh.multi_stages["a"].parameters())
    # The unfreeze already happened; the next epoch start must not add a group again.
    fresh.on_epoch_start(3)
    assert len(fresh_pipeline.optimizer_g.param_groups) == groups


def test_resume_refreezes_a_stage_early_stopping_froze(tmp_path: Path) -> None:
    pipeline, s = _trained()
    s.on_epoch_end(0, {"val_a_l1": 0.5})
    s.on_epoch_end(1, {"val_a_l1": 0.75})
    assert not any(p.requires_grad for p in s.multi_stages["a"].parameters())
    path = _save(tmp_path, pipeline, s)

    fresh_pipeline = make_pipeline(discriminator=False, ema=False)
    fresh = _strategy(fresh_pipeline, seed=2)
    _resume(path, fresh_pipeline, fresh)
    assert not any(p.requires_grad for p in fresh.multi_stages["a"].parameters())
    assert all(p.requires_grad for p in fresh.multi_stages["b"].parameters())
    trainable = {id(p) for p in fresh._trainable_params}
    assert not trainable & {id(p) for p in fresh.multi_stages["a"].parameters()}


def test_the_e2e_unfreeze_releases_an_early_stopped_stage_on_resume(tmp_path: Path) -> None:
    """The unfreeze sets every stage trainable, the early-stopped one included."""
    pipeline, s = _trained()
    s.on_epoch_end(0, {"val_a_l1": 0.5})
    s.on_epoch_end(1, {"val_a_l1": 0.75})
    s.on_epoch_start(2)
    path = _save(tmp_path, pipeline, s)
    assert codecs.read(path)["strategy"]["pipeline"]["frozen_by_early_stopping"] == []

    fresh_pipeline = make_pipeline(discriminator=False, ema=False)
    fresh = _strategy(fresh_pipeline, seed=2)
    _resume(path, fresh_pipeline, fresh)
    assert all(p.requires_grad for p in fresh.multi_stages["a"].parameters())


@pytest.mark.parametrize("stages", [{"a"}, {"a", "b", "c"}])
def test_a_file_whose_stages_differ_from_the_live_ones_is_refused(tmp_path: Path, stages) -> None:
    pipeline, s = _trained()
    path = _save(tmp_path, pipeline, s)
    state = codecs.read(path)
    state["models"] = {k: v for k, v in state["models"].items() if not k.startswith("stage:")} | {
        f"stage:{n}": nn.Linear(4, 4).state_dict() for n in stages
    }
    codecs.write(path, state)
    fresh_pipeline = make_pipeline(discriminator=False, ema=False)
    with pytest.raises(RuntimeError, match="#2962"):
        _resume(path, fresh_pipeline, _strategy(fresh_pipeline, seed=2))


def test_a_file_with_no_stage_entries_names_the_issue(tmp_path: Path) -> None:
    pipeline, s = _trained()
    path = _save(tmp_path, pipeline, s)
    state = codecs.read(path)
    state["models"] = {"generator": state["models"]["generator"]}
    codecs.write(path, state)
    fresh_pipeline = make_pipeline(discriminator=False, ema=False)
    with pytest.raises(RuntimeError, match=r"no stage.*#2962"):
        _resume(path, fresh_pipeline, _strategy(fresh_pipeline, seed=2))


def test_weights_warm_starts_a_stage_from_a_runs_own_step_file(tmp_path: Path) -> None:
    pipeline, s = _trained()
    path = _save(tmp_path, pipeline, s)
    target = nn.Linear(4, 4)
    weights(path, "stage:a", target, policy="exact")
    assert _equal(target, s.multi_stages["a"])
