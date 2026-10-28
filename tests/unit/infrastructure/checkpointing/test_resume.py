"""resume(): every part back into the live run, in the order the director restored it."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from spectramr.infrastructure.checkpointing import codecs, store
from spectramr.infrastructure.checkpointing import resume as resume_mod
from spectramr.infrastructure.checkpointing.layout import CheckpointLayout
from spectramr.infrastructure.checkpointing.resume import resume
from tests.unit.infrastructure.checkpointing.conftest import (
    FakeEarlyStopping,
    FakeNativeAdapter,
    make_pipeline,
    native_runtime,
)


def _config(**ckpt):
    # The load gates and topology opt-outs are unset unless a test sets them: their
    # readers' defaults (load, refuse) are what a bare config means.
    fields = {"format": "pt", "keep_last_n": 5} | ckpt
    return SimpleNamespace(checkpoint=SimpleNamespace(**fields))


def _save(
    tmp_path: Path, pipeline=None, *, epoch=1, step=4, fmt="pt", strategy=None, es=True
) -> Path:
    return store.save_step(
        pipeline or make_pipeline(),
        layout=CheckpointLayout.for_run(tmp_path),
        config=_config(format=fmt),
        strategy=strategy,
        parallel=None,
        early_stopping=FakeEarlyStopping() if es else None,
        reason="periodic",
        epoch=epoch,
        step=step,
    )


def _resume(path: Path, pipeline=None, *, strategy=None, parallel=None, **ckpt):
    live = pipeline or make_pipeline()
    return resume(path, live, strategy=strategy, parallel=parallel, config=_config(**ckpt))


def _native_file(tmp_path: Path, *, strategy="deepspeed", tag="step_00000004") -> Path:
    """A step file as a native adapter writes it: it names its tag directory."""
    path = _save(tmp_path)
    state = codecs.read(path)
    state["meta"]["native_tag"] = tag
    state["meta"]["topology"]["parallel_strategy"] = strategy
    codecs.write(path, state)
    return path


def test_position_and_reason_come_from_meta(tmp_path: Path) -> None:
    result = _resume(_save(tmp_path, step=40))
    assert (result.epoch, result.step, result.reason) == (1, 40, "periodic")


def test_epoch_zero_step_zero_is_a_legitimate_position(tmp_path: Path) -> None:
    result = _resume(_save(tmp_path, epoch=0, step=0, es=False))
    assert (result.epoch, result.step) == (0, 0)


@pytest.mark.parametrize("fmt", ["safetensors", "pth"])
def test_resume_reads_a_step_file_of_another_format(tmp_path: Path, fmt: str) -> None:
    """The run now declares pt; the latest file was written as another format."""
    assert _resume(_save(tmp_path, fmt=fmt, step=9), format="pt").step == 9


def test_weights_and_optimizer_come_back(tmp_path: Path) -> None:
    saved = make_pipeline()
    path = _save(tmp_path, saved)
    fresh = make_pipeline()
    nn.init.zeros_(fresh.generator.weight)
    _resume(path, fresh)
    assert torch.equal(fresh.generator.weight, saved.generator.weight)
    assert fresh.optimizer_g.state_dict()["state"][0]["step"] == 1


def test_load_optimizer_state_false_leaves_the_optimizer_untouched(tmp_path: Path) -> None:
    path = _save(tmp_path)
    fresh = make_pipeline()
    fresh.optimizer_g.state.clear()
    _resume(path, fresh, load_optimizer_state=False)
    assert fresh.optimizer_g.state_dict()["state"] == {}


@pytest.mark.parametrize(("gate", "expected"), [(True, 3), (False, 0)])
def test_load_scheduler_state_gates_the_scheduler(tmp_path: Path, gate, expected) -> None:
    saved = make_pipeline()
    saved.scheduler_g.last_epoch = 3
    path = _save(tmp_path, saved)
    fresh = make_pipeline()
    _resume(path, fresh, load_scheduler_state=gate)
    assert fresh.scheduler_g.last_epoch == expected


def test_discriminator_optimizer_is_restored_independently(tmp_path: Path) -> None:
    saved = make_pipeline()
    path = _save(tmp_path, saved)
    fresh = make_pipeline()
    fresh.optimizer_d.state.clear()
    _resume(path, fresh)
    assert fresh.optimizer_d.state_dict()["state"][0]["step"] == 1


def test_scaler_state_restores_into_the_live_scaler(tmp_path: Path) -> None:
    saved = make_pipeline(scaler=True)
    saved.scaler.scale(torch.ones(1))  # the scale is created lazily by the first scale()
    saved.scaler.update(1024.0)
    path = _save(tmp_path, saved)
    live = torch.amp.GradScaler("cpu", enabled=True)
    strategy = SimpleNamespace(amp_helper=SimpleNamespace(scaler=live))
    _resume(path, make_pipeline(scaler=True), strategy=strategy)
    assert live.get_scale() == 1024.0


def test_ema_counter_survives_a_round_trip(tmp_path: Path) -> None:
    path = _save(tmp_path)
    fresh = make_pipeline()
    fresh.ema.num_updates = 0
    _resume(path, fresh)
    assert fresh.ema.num_updates == 1


@pytest.mark.parametrize(("es", "wait"), [(True, 1), (False, None)])
def test_early_stopping_state_is_returned_or_none(tmp_path: Path, es: bool, wait) -> None:
    state = _resume(_save(tmp_path, es=es)).early_stopping
    assert (state["wait_count"] if es else state) == wait


def test_strategy_state_is_restored_before_the_optimizer(tmp_path: Path) -> None:
    """#2796: a strategy that sizes a module lazily adds its param group on restore."""
    extra = nn.Parameter(torch.ones(1))
    saved = make_pipeline()
    saved.optimizer_g.add_param_group({"params": [extra]})
    strategy = SimpleNamespace(strategy_state_dict=lambda: {"params": {"extra": extra.detach()}})
    path = _save(tmp_path, saved, strategy=strategy)

    fresh = make_pipeline()

    def load_strategy_state_dict(state, **_gates):
        fresh.optimizer_g.add_param_group({"params": [nn.Parameter(state["params"]["extra"])]})

    restoring = SimpleNamespace(load_strategy_state_dict=load_strategy_state_dict)
    _resume(path, fresh, strategy=restoring)
    assert len(fresh.optimizer_g.param_groups) == 2


def test_a_strategy_owning_only_a_random_stream_resumes_it(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(3)
    torch.rand(2, generator=gen)
    saved_state = gen.get_state()
    owner = SimpleNamespace(strategy_state_dict=lambda: {"generators": {"_rng": saved_state}})
    path = _save(tmp_path, strategy=owner)
    seen: dict = {}
    load = SimpleNamespace(load_strategy_state_dict=lambda st, **_: seen.update(st))
    _resume(path, strategy=load)
    assert torch.equal(seen["generators"]["_rng"], saved_state)


def test_strategy_state_reaches_the_strategy_on_the_pipelines_device(tmp_path: Path) -> None:
    """As the director's map_location did: a lazily built module must not land on CPU (#2796)."""
    owned = {"params": {"s": torch.ones(2)}, "shapes": [(torch.ones(1), 3)]}
    owner = SimpleNamespace(strategy_state_dict=lambda: owned)
    path = _save(tmp_path, strategy=owner)
    fresh = make_pipeline()
    fresh.device = torch.device("meta")
    seen: dict = {}
    load = SimpleNamespace(load_strategy_state_dict=lambda st, **_: seen.update(st))
    _resume(path, fresh, strategy=load)
    assert seen["params"]["s"].device.type == "meta"
    assert seen["shapes"][0][0].device.type == "meta" and seen["shapes"][0][1] == 3


class TestRng:
    def test_resume_replays_the_interrupted_runs_stream(self, tmp_path: Path) -> None:
        torch.manual_seed(11)
        np.random.seed(11)
        random.seed(11)
        path = _save(tmp_path)
        expected = (torch.rand(2), np.random.rand(2), random.random())
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        _resume(path)
        got = (torch.rand(2), np.random.rand(2), random.random())
        assert torch.equal(got[0], expected[0]) and np.array_equal(got[1], expected[1])
        assert got[2] == expected[2]

    def test_a_non_main_rank_keeps_its_own_stream(self, tmp_path: Path) -> None:
        path = _save(tmp_path)
        fresh = make_pipeline()  # built first: make_pipeline seeds torch itself
        torch.manual_seed(5)
        expected = torch.rand(2)
        torch.manual_seed(5)
        with patch.object(resume_mod.RankUtility, "is_main_rank", staticmethod(lambda: False)):
            _resume(path, fresh)
        assert torch.equal(torch.rand(2), expected)

    def test_a_file_without_rng_still_loads(self, tmp_path: Path) -> None:
        path = _save(tmp_path)
        state = codecs.read(path)
        del state["rng"]
        codecs.write(path, state)
        assert _resume(path).step == 4


class TestNativeHandledKeys:
    """Keys the native artifact restored are skipped per model, never per run."""

    def test_a_zero_optimizer_is_not_reloaded_from_the_payload(self, tmp_path: Path) -> None:
        native = FakeNativeAdapter(frozenset({"generator"}))
        fresh = make_pipeline()
        fresh.optimizer_g.state.clear()
        _resume(_native_file(tmp_path), fresh, parallel=native_runtime(native))
        assert fresh.optimizer_g.state_dict()["state"] == {}  # #2482: the engine owns it
        assert native.seen["tag"] == "step_00000004"
        assert native.seen["checkpoint_dir"] == tmp_path / "checkpoints"

    def test_a_plain_discriminator_beside_an_engine_is_still_restored(self, tmp_path: Path) -> None:
        fresh = make_pipeline()
        fresh.optimizer_d.state.clear()
        _resume(
            _native_file(tmp_path),
            fresh,
            parallel=native_runtime(FakeNativeAdapter(frozenset({"generator"}))),
        )
        assert fresh.optimizer_d.state_dict()["state"][0]["step"] == 1  # #1309

    def test_the_optimizer_and_scheduler_gates_reach_the_native_load(self, tmp_path: Path) -> None:
        native = FakeNativeAdapter(frozenset({"generator"}))
        _resume(
            _native_file(tmp_path),
            parallel=native_runtime(native),
            load_optimizer_state=False,
            load_scheduler_state=False,
        )
        assert (native.seen["optimizer"], native.seen["scheduler"]) == (False, False)  # #2582

    def test_a_file_without_a_native_tag_is_not_read_natively(self, tmp_path: Path) -> None:
        native = FakeNativeAdapter(on_load=lambda **_: pytest.fail("no tag to load"))
        path = _native_file(tmp_path, tag=None)
        assert _resume(path, parallel=native_runtime(native)).step == 4
