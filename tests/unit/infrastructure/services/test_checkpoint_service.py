"""Unit tests for :mod:`spectramr.infrastructure.services.checkpoint_service`.

Focus: the safetensors save/load round-trip must preserve ``counter_state``
(the global-step / iteration counter). safetensors stores only tensors, so any
scalar / nested non-tensor metadata must be persisted out-of-band (pickled into
a uint8 tensor, like ``rng_state`` and ``strategy_state`` already are) and
re-hydrated on load. Without that, a resumed run silently restarts at step 0.
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from spectramr.config.schemas.checkpoint import CheckpointConfigSchema
from spectramr.core.rng_state import RNG_STATE_SAFE_GLOBALS, capture_rng_state
from spectramr.infrastructure.services.checkpoint_service import (
    CheckpointService,
    discover_best_checkpoint,
)


class _TinyModel(nn.Module):
    """Minimal model so state_dict round-trips through safetensors."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.linear(x)


class _FakeCounter:
    """Stand-in for IterationCounterService with the same state contract.

    Mirrors ``IterationCounterService.get_state`` / ``restore_state`` for the
    plain-int counter fields (the resume-critical ones). Avoids needing a full
    training config to instantiate the real service.
    """

    def __init__(self, step: int = 0, epoch: int = 0) -> None:
        self.current_step = step
        self.current_epoch = epoch
        self.steps_in_current_epoch = 0
        self.total_steps_this_epoch = 0

    def get_state(self) -> dict:
        return {
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "steps_in_current_epoch": self.steps_in_current_epoch,
            "total_steps_this_epoch": self.total_steps_this_epoch,
        }

    def restore_state(self, state: dict) -> None:
        self.current_step = state.get("current_step", 0)
        self.current_epoch = state.get("current_epoch", 0)
        self.steps_in_current_epoch = state.get("steps_in_current_epoch", 0)
        self.total_steps_this_epoch = state.get("total_steps_this_epoch", 0)


def _make_service(tmp_path, fmt: str) -> CheckpointService:
    config = CheckpointConfigSchema(
        checkpoint_dir=str(tmp_path),
        format=fmt,
        save_interval=1,
    )
    return CheckpointService(config)


@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["safetensors", "pth"])
def test_counter_state_round_trips(tmp_path, fmt):
    """A non-zero ``counter_state`` saved via either format must come back.

    This is the resume-correctness invariant: after loading, the restored
    counter's ``current_step`` must equal the value saved (1234), NOT 0.
    """
    service = _make_service(tmp_path, fmt)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    save_counter = _FakeCounter(step=1234, epoch=7)
    path = service.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=7,
        loss=0.5,
        step=1234,
        counter=save_counter,
    )

    # 1. The saved state dict must carry counter_state back through the loader.
    state = service.load_checkpoint(
        model=_TinyModel(),
        optimizer=torch.optim.SGD(_TinyModel().parameters(), lr=0.1),
        file_path=path,
    )
    assert state is not None
    assert "counter_state" in state, (
        f"counter_state dropped by the {fmt} round-trip "
        "-- resume would restart at step 0"
    )
    assert state["counter_state"]["current_step"] == 1234
    assert state["counter_state"]["current_epoch"] == 7

    # 2. A fresh counter handed to load_checkpoint must be restored in place.
    restore_counter = _FakeCounter(step=0, epoch=0)
    service.load_checkpoint(
        model=_TinyModel(),
        optimizer=torch.optim.SGD(_TinyModel().parameters(), lr=0.1),
        file_path=path,
        counter=restore_counter,
    )
    assert restore_counter.current_step == 1234, (
        f"counter not restored from {fmt} checkpoint -- resume restarts at step 0"
    )
    assert restore_counter.current_epoch == 7


@pytest.mark.unit
@pytest.mark.parametrize("fmt", ["safetensors", "pth"])
def test_scheduler_state_round_trips_and_restores(tmp_path, fmt):
    """LR scheduler state must survive the round-trip; otherwise a resume
    restarts the warmup/decay schedule at step 0 and the LR jumps."""
    service = _make_service(tmp_path, fmt)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(3):  # advance the schedule: last_epoch -> 3
        scheduler.step()
    assert scheduler.last_epoch == 3

    path = service.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        loss=0.5,
        step=3,
        scheduler=scheduler,
    )

    state = service.load_checkpoint(
        model=_TinyModel(),
        optimizer=torch.optim.SGD(_TinyModel().parameters(), lr=0.1),
        file_path=path,
    )
    assert state is not None
    assert "scheduler_state_dict" in state, (
        f"scheduler_state_dict dropped by the {fmt} round-trip -- LR schedule resets"
    )

    # A fresh scheduler handed to load_checkpoint must be restored in place.
    fresh_opt = torch.optim.SGD(_TinyModel().parameters(), lr=0.1)
    fresh_sched = torch.optim.lr_scheduler.StepLR(fresh_opt, step_size=1, gamma=0.5)
    assert fresh_sched.last_epoch == 0
    service.load_checkpoint(
        model=_TinyModel(),
        optimizer=fresh_opt,
        file_path=path,
        scheduler=fresh_sched,
    )
    assert fresh_sched.last_epoch == 3, (
        f"scheduler not restored from {fmt} checkpoint -- schedule restarts at 0"
    )


@pytest.mark.unit
def test_load_scheduler_state_false_skips_restore(tmp_path):
    """The advertised ``load_scheduler_state`` knob must actually gate the
    restore (pitfall #15: no dead knobs)."""
    config = CheckpointConfigSchema(
        checkpoint_dir=str(tmp_path),
        format="pth",
        save_interval=1,
        load_scheduler_state=False,
    )
    service = CheckpointService(config)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    for _ in range(3):
        scheduler.step()
    path = service.save_checkpoint(
        model=model, optimizer=optimizer, epoch=1, loss=0.5, step=3, scheduler=scheduler
    )

    fresh_opt = torch.optim.SGD(_TinyModel().parameters(), lr=0.1)
    fresh_sched = torch.optim.lr_scheduler.StepLR(fresh_opt, step_size=1, gamma=0.5)
    service.load_checkpoint(
        model=_TinyModel(),
        optimizer=fresh_opt,
        file_path=path,
        scheduler=fresh_sched,
    )
    assert fresh_sched.last_epoch == 0, "load_scheduler_state=False must skip restore"


@pytest.mark.unit
def test_keep_best_n_protects_top_checkpoints_from_retention(tmp_path):
    """With ``keep_best_n`` set, the best-metric periodic checkpoints must
    survive retention even when they fall outside the ``keep_last_n`` window."""
    config = CheckpointConfigSchema(
        checkpoint_dir=str(tmp_path),
        format="pth",
        save_interval=1,
        keep_last_n=1,
        keep_best_n=2,
    )
    service = CheckpointService(config)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Save four periodic checkpoints with decreasing loss (lower is better).
    # Step 1 (loss 0.9) is worst; steps 2 & 3 (0.4, 0.2) are the two best;
    # step 4 (0.5) is newest so protected by keep_last_n=1.
    paths = {}
    for step, loss in [(1, 0.9), (2, 0.4), (3, 0.2), (4, 0.5)]:
        paths[step] = service.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=0,
            loss=loss,
            step=step,
            metric_name="loss",
            metric_value=loss,
        )

    from pathlib import Path

    # keep_last_n=1 -> step 4 kept; keep_best_n=2 -> steps 2,3 kept; step 1 deleted.
    assert Path(paths[4]).exists(), "newest (keep_last_n) checkpoint deleted"
    assert Path(paths[3]).exists(), "best checkpoint deleted despite keep_best_n"
    assert Path(paths[2]).exists(), "2nd-best checkpoint deleted despite keep_best_n"
    assert not Path(paths[1]).exists(), "worst checkpoint should have been pruned"


@pytest.mark.unit
def test_discover_best_checkpoint_prefers_alias(tmp_path):
    """The alias both writers publish takes priority over the canonical file."""
    (tmp_path / "checkpoint_best.pt").write_text("canonical")
    (tmp_path / "best.pt").write_text("alias")
    assert discover_best_checkpoint(tmp_path).name == "best.pt"


@pytest.mark.unit
def test_discover_best_checkpoint_finds_safetensors_alias(tmp_path):
    """A safetensors-service run publishes best.safetensors, not best.pt — the
    old ``best.pt``-only discovery missed it entirely."""
    (tmp_path / "best.safetensors").write_text("x")
    assert discover_best_checkpoint(tmp_path).name == "best.safetensors"


@pytest.mark.unit
def test_discover_best_checkpoint_falls_back_to_newest_step(tmp_path):
    """With no best/alias, fall back to the newest step checkpoint — NOT the
    dead ``model_iter_*.pt`` glob (which matched nothing either writer emits)."""
    import os
    import time

    old = tmp_path / "checkpoint_step_100.pt"
    new = tmp_path / "checkpoint_step_200.safetensors"
    old.write_text("old")
    new.write_text("new")
    # Make 'new' unambiguously newer by mtime.
    past = time.time() - 100
    os.utime(old, (past, past))
    assert discover_best_checkpoint(tmp_path).name == "checkpoint_step_200.safetensors"


@pytest.mark.unit
def test_discover_best_checkpoint_none_when_empty(tmp_path):
    assert discover_best_checkpoint(tmp_path) is None
    assert discover_best_checkpoint(tmp_path / "does_not_exist") is None
    # The dead legacy pattern must NOT be resurrected as a match.
    (tmp_path / "model_iter_500.pt").write_text("legacy-name-no-writer-emits")
    assert discover_best_checkpoint(tmp_path) is None


@pytest.mark.unit
def test_rng_state_needs_exactly_the_declared_safe_globals(tmp_path):
    """The allowlist is a claim about what ``capture_rng_state`` pickles.

    Stated as a bidirectional pin: without the four entries a plain
    ``torch.load`` refuses the payload, and *with* them the weights-only
    unpickler accepts it. A drift in either direction -- numpy changing what
    ``get_state`` returns, or the list growing an entry it does not need --
    breaks one half.
    """
    path = tmp_path / "rng.pth"
    torch.save({"rng_state": capture_rng_state()}, path)

    with pytest.raises(Exception, match=r"[Ww]eights only"):
        torch.load(path, map_location="cpu")

    with torch.serialization.safe_globals(RNG_STATE_SAFE_GLOBALS):
        restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored["rng_state"]["numpy"][1].shape == (624,)


@pytest.mark.unit
def test_saved_checkpoint_loads_without_the_unsafe_fallback(tmp_path, caplog):
    """The weights-only guard must be reachable on the service's own format.

    ``save_checkpoint`` stamps ``rng_state`` unconditionally, so before the
    allowlist the ``weights_only=True`` attempt failed on *every* checkpoint
    this service writes: the load silently completed through the full
    unpickler, and logged a fallback warning each time (pitfall #10). Assert
    on the warning's absence, which is what tells the two paths apart -- the
    load itself succeeds either way.
    """
    service = _make_service(tmp_path, "pth")
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters())
    file_path = service.save_checkpoint(
        model=model, optimizer=optimizer, epoch=1, step=7, loss=0.25
    )

    with caplog.at_level(logging.WARNING):
        state = service.load_checkpoint(
            model=_TinyModel(),
            optimizer=torch.optim.Adam(_TinyModel().parameters()),
            file_path=file_path,
        )

    assert state is not None
    assert not [r for r in caplog.records if "weights_only=True" in r.getMessage()]

    # ...and the allowlist stayed scoped: a bare torch.load is still refused,
    # so the service did not quietly widen what every other caller unpickles.
    with pytest.raises(Exception, match=r"[Ww]eights only"):
        torch.load(file_path, map_location="cpu")


# ---------------------------------------------------------------------------
# Auto-resume discovery: the finder must see what the WRITER produced.
#
# `find_latest_checkpoint` globbed `.{self.format}` only. The writer on the
# production training path is `CheckpointDirector.save()`, which spells `.pt`
# unconditionally, while no corpus arm declares `checkpoint.format` -- so the
# schema default `safetensors` made the glob miss every file a real run had
# just written, `--resume auto` raised "No checkpoint found", and the chained
# job died instead of resuming.
#
# The tests that were green over this planted `.safetensors` names by hand,
# which is the reader's assumption rather than either writer's output. These
# plant the writer's, and the last one re-derives the suffix from the director's
# own source so the pair cannot silently drift apart again (non-negotiable 15).
# ---------------------------------------------------------------------------


def _director_checkpoint_suffix() -> str:
    """The suffix ``CheckpointDirector.save()`` actually writes, read off it."""
    import inspect
    import re

    from spectramr.infrastructure.builders.directors.checkpoint_director import (
        CheckpointDirector,
    )

    src = inspect.getsource(CheckpointDirector.save)
    suffixes = set(re.findall(r'f"checkpoint_epoch_\{[^"]*\}\.(\w+)"', src))
    assert suffixes, "could not read the director's checkpoint filename"
    assert len(suffixes) == 1, f"director now writes several suffixes: {suffixes}"
    return suffixes.pop()


@pytest.mark.unit
def test_auto_resume_finds_a_director_written_checkpoint(tmp_path):
    """The default-format service must resolve the director's ``.pt`` files.

    This is the exact shape a 120 h production arm leaves behind: the config
    declares no ``format``, so the service defaults to safetensors, and every
    file in the directory came from the director.
    """
    service = _make_service(tmp_path, "safetensors")
    for name in (
        "checkpoint_epoch_0001_step_005000.pt",
        "checkpoint_epoch_0002_step_010000.pt",
    ):
        (tmp_path / name).write_bytes(b"x")

    latest = service.find_latest_checkpoint(str(tmp_path))

    assert latest is not None, (
        "auto-resume found nothing in a directory full of checkpoints -- the "
        "finder is globbing the declared format while the director writes .pt"
    )
    assert "step_010000" in latest


@pytest.mark.unit
def test_auto_resume_orders_by_step_across_mixed_suffixes(tmp_path):
    """Both writers may have run; the newest STATE wins, not the newest suffix."""
    service = _make_service(tmp_path, "safetensors")
    (tmp_path / "checkpoint_step_9000.safetensors").write_bytes(b"x")
    (tmp_path / "checkpoint_epoch_0002_step_010000.pt").write_bytes(b"x")

    assert "step_010000" in service.find_latest_checkpoint(str(tmp_path))


@pytest.mark.unit
@pytest.mark.parametrize("alias", ["checkpoint_last", "checkpoint_best"])
def test_auto_resume_falls_back_to_a_pt_alias(tmp_path, alias):
    """``checkpoint_best.pt`` is what the director publishes; it must be seen."""
    service = _make_service(tmp_path, "safetensors")
    (tmp_path / f"{alias}.pt").write_bytes(b"x")

    assert service.find_latest_checkpoint(str(tmp_path)) is not None


@pytest.mark.unit
def test_last_alias_outranks_best_alias(tmp_path):
    """``last`` is the resume point; ``best`` is an evaluation artifact."""
    service = _make_service(tmp_path, "safetensors")
    (tmp_path / "checkpoint_best.pt").write_bytes(b"x")
    (tmp_path / "checkpoint_last.pt").write_bytes(b"x")

    assert "checkpoint_last" in service.find_latest_checkpoint(str(tmp_path))


@pytest.mark.unit
def test_finder_covers_the_suffix_the_director_writes(tmp_path):
    """Anti-drift: re-derive the director's suffix rather than hardcoding it."""
    suffix = _director_checkpoint_suffix()
    service = _make_service(tmp_path, "safetensors")
    (tmp_path / f"checkpoint_epoch_0001_step_000100.{suffix}").write_bytes(b"x")

    assert service.find_latest_checkpoint(str(tmp_path)) is not None, (
        f"the director writes .{suffix} but find_latest_checkpoint does not "
        "glob it -- auto-resume is dead on every arm again"
    )


@pytest.mark.unit
def test_empty_directory_still_reports_nothing(tmp_path):
    """Widening the glob must not invent a checkpoint where there is none."""
    assert _make_service(tmp_path, "safetensors").find_latest_checkpoint(str(tmp_path)) is None
