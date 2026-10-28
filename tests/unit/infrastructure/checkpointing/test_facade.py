"""CheckpointStore: the DI face of the module, forwarding by module attribute."""

from __future__ import annotations

import inspect

from spectramr.domain.interfaces.checkpoint_service_interface import ICheckpointService
from spectramr.infrastructure.checkpointing import facade, restore, resume, store
from spectramr.infrastructure.checkpointing.facade import CheckpointStore


def test_it_implements_the_one_interface() -> None:
    assert isinstance(CheckpointStore(), ICheckpointService)
    assert ICheckpointService.__abstractmethods__ == {
        "save_step",
        "save_best",
        "resume",
        "best_weights",
    }


def test_each_method_forwards_to_its_module_function(monkeypatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(store, "save_step", lambda p, **k: seen.append("step") or "s")
    monkeypatch.setattr(store, "save_best", lambda p, **k: seen.append("best") or "b")
    monkeypatch.setattr(resume, "resume", lambda path, p, **k: seen.append("resume"))
    monkeypatch.setattr(restore, "best_weights", lambda path, p, **k: seen.append("weights"))
    s = CheckpointStore()
    s.save_step(object(), step=1)
    s.save_best(object(), step=1)
    s.resume("x", object())
    s.best_weights("x", object())
    assert seen == ["step", "best", "resume", "weights"]


def test_the_facade_holds_no_logic_of_its_own() -> None:
    assert inspect.getsource(facade).count("def ") == 4


def test_the_writer_keyword_reaches_the_store(monkeypatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(store, "save_step", lambda p, **k: seen.update(step=k.get("writer")))
    monkeypatch.setattr(store, "save_best", lambda p, **k: seen.update(best=k.get("writer")))
    writer = object()
    CheckpointStore().save_step(object(), step=1, writer=writer)
    CheckpointStore().save_best(object(), step=1, writer=writer)
    assert seen == {"step": writer, "best": writer}
