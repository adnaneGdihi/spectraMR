"""resume() refusals: each one fires before any part of the run is changed."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.distributed.distributed_training import RankUtility
from tests.unit.infrastructure.checkpointing.conftest import (
    FakeNativeAdapter,
    make_pipeline,
    native_runtime,
)
from tests.unit.infrastructure.checkpointing.test_resume import (
    _native_file,
    _resume,
    _save,
)


def test_an_orphan_strategy_entry_fails_the_resume(tmp_path: Path) -> None:
    """#2893: the strategy's refusal surfaces through the load failure, never swallowed."""
    owner = SimpleNamespace(strategy_state_dict=lambda: {"params": {"gone": torch.ones(1)}})
    path = _save(tmp_path, strategy=owner)
    refuse = SimpleNamespace(load_strategy_state_dict=lambda _state, **_: {}["gone"])
    with pytest.raises(RuntimeError, match="KeyError"):
        _resume(path, strategy=refuse)


def test_resume_hands_the_optimizer_gate_to_the_strategy(tmp_path: Path) -> None:
    """A strategy owning optimizers obeys load_optimizer_state like the pipeline (#2964)."""
    owner = SimpleNamespace(strategy_state_dict=lambda: {"params": {"s": torch.ones(1)}})
    path = _save(tmp_path, strategy=owner)
    gates: list[bool] = []
    load = SimpleNamespace(
        load_strategy_state_dict=lambda _st, *, load_optimizer: gates.append(load_optimizer)
    )
    _resume(path, strategy=load, load_optimizer_state=False)
    assert gates == [False]


def test_a_strategys_refusal_runs_before_any_model_moves(tmp_path: Path) -> None:
    """check_strategy_state is called before the loads, without a strategy part too."""
    path = _save(tmp_path)
    live = make_pipeline()
    nn.init.zeros_(live.generator.weight)
    seen: list = []

    def refuse(part):
        seen.append(part)
        raise RuntimeError("refused")

    with pytest.raises(RuntimeError, match="refused"):
        _resume(path, live, strategy=SimpleNamespace(check_strategy_state=refuse))
    assert seen == [None]
    assert torch.count_nonzero(live.generator.weight) == 0


def test_topology_is_refused_before_the_native_load(tmp_path: Path, monkeypatch) -> None:
    """The file names a native tag, so a native load would run if the guard came second."""
    path = _native_file(tmp_path, strategy="none")
    monkeypatch.setattr(RankUtility, "get_world_size", staticmethod(lambda: 4))
    native = FakeNativeAdapter(on_load=lambda **_: pytest.fail("native load ran first"))
    with pytest.raises(RuntimeError, match="allow_world_size_change"):
        _resume(path, parallel=native_runtime(native, strategy="none"))


def test_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        _resume(tmp_path / "checkpoints" / "step_00000004.pt")


def test_a_failed_load_names_the_file_and_the_exception_type(tmp_path: Path, caplog) -> None:
    """#2902: the operator reads the message, which must say which file and what failed."""
    path = _save(tmp_path)
    fresh = make_pipeline()
    fresh.generator = nn.Linear(5, 5)
    expected = f"Failed to load checkpoint {path}: RuntimeError: "
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match=re.escape(expected)):
        _resume(path, fresh)
    assert expected in caplog.text


def test_a_partitioned_ema_shadow_is_refused(tmp_path: Path) -> None:
    """At ZeRO-3 the step file holds rank 0's partitioned shadow, not a whole one."""
    native = FakeNativeAdapter(frozenset({"generator"}), ema_whole=False)
    with pytest.raises(RuntimeError, match="EMA shadow"):
        _resume(_native_file(tmp_path), parallel=native_runtime(native))
    assert native.seen == {}  # refused before the collective native load


class TestMissingNativeState:
    """#2965: an engine optimizer the file does not hold is refused by name, before any load."""

    def _without_generator_optim(self, tmp_path: Path, tag: str | None) -> Path:
        path = _native_file(tmp_path, tag=tag)
        state = codecs.read(path)
        del state["optim"]["generator"]  # where capture leaves an engine-owned optimizer
        codecs.write(path, state)
        return path

    def test_a_file_without_the_generator_optimizer_names_its_tag(self, tmp_path: Path) -> None:
        live = make_pipeline()
        before = live.generator.weight.detach().clone()
        path = self._without_generator_optim(tmp_path, tag="step_00000004")
        with pytest.raises(RuntimeError, match=r"generator optimizer.*step_00000004"):
            _resume(path, live, allow_parallel_strategy_change=True)
        assert torch.equal(live.generator.weight, before)

    def test_a_missing_tag_directory_is_refused_before_the_native_load(self, tmp_path) -> None:
        native = FakeNativeAdapter(on_load=lambda **_: pytest.fail("native load ran"))
        root = tmp_path / "checkpoints" / "deepspeed" / "generator"
        native.native_tag_roots = lambda **_: (root,)
        path = self._without_generator_optim(tmp_path, tag="step_00000004")
        with pytest.raises(RuntimeError, match=r"tag 'step_00000004'.*missing"):
            _resume(path, parallel=native_runtime(native))
        (root / "step_00000004").mkdir(parents=True)
        native.on_load = None
        assert _resume(path, parallel=native_runtime(native)).step == 4

    def test_the_optimizer_gate_off_resumes_the_weights(self, tmp_path: Path) -> None:
        path = self._without_generator_optim(tmp_path, tag=None)
        result = _resume(path, load_optimizer_state=False, allow_parallel_strategy_change=True)
        assert result.step == 4
