"""async_writer: the file write leaves the training thread, its failure does not get lost."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, namedtuple
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from spectramr.infrastructure.checkpointing import async_writer, codecs
from spectramr.infrastructure.checkpointing.async_writer import (
    CheckpointWriteError,
    CheckpointWriter,
    snapshot_to_cpu,
)
from tests.unit.infrastructure.checkpointing.conftest import make_run_state

_REAL_WRITE = codecs.write


def _gated(gate: threading.Event):
    """A ``codecs.write`` that waits for ``gate``, so a test controls when the file lands."""

    def write(path, state):
        assert gate.wait(timeout=10)
        _REAL_WRITE(path, state)

    return write


def _path(tmp_path: Path, step: int) -> Path:
    tmp_path.joinpath("checkpoints").mkdir(exist_ok=True)
    return tmp_path / "checkpoints" / f"step_{step:08d}.pt"


def test_the_submitted_state_is_a_cpu_copy(tmp_path: Path) -> None:
    state = make_run_state("pt")
    live = state["models"]["generator"]["0.weight"]
    before = live.clone()
    gate = threading.Event()
    writer = CheckpointWriter()
    with patch.object(async_writer.codecs, "write", _gated(gate)):
        writer.submit(_path(tmp_path, 1), state, None)
        live.add_(1.0)  # the next optimizer step, while the write is still queued
        gate.set()
        writer.drain()
    stored = codecs.read(_path(tmp_path, 1))["models"]["generator"]["0.weight"]
    assert torch.equal(stored, before)


def test_a_write_error_surfaces_at_the_next_submit_and_at_drain(tmp_path: Path) -> None:
    writer = CheckpointWriter()
    with patch.object(async_writer.codecs, "write", side_effect=OSError("disk full")):
        writer.submit(_path(tmp_path, 1), make_run_state("pt"), None)
        with pytest.raises(RuntimeError, match=r"step_00000001\.pt.*disk full"):
            writer.submit(_path(tmp_path, 2), make_run_state("pt"), None)
    # Sticky: a caller that swallowed it (the fail-soft best save) does not end the story.
    with pytest.raises(RuntimeError, match="disk full"):
        writer.drain()
    with pytest.raises(RuntimeError, match="disk full"):
        writer.drain()
    assert not _path(tmp_path, 2).exists()


def test_drain_makes_the_file_exist(tmp_path: Path) -> None:
    gate = threading.Event()
    writer = CheckpointWriter()
    with patch.object(async_writer.codecs, "write", _gated(gate)):
        writer.submit(_path(tmp_path, 1), make_run_state("pt"), None)
        assert not _path(tmp_path, 1).exists()
        gate.set()
        writer.drain()
    assert _path(tmp_path, 1).exists()


def test_then_runs_after_the_file_landed_off_the_calling_thread(tmp_path: Path) -> None:
    seen: dict = {}
    path = _path(tmp_path, 1)

    def then() -> None:
        seen.update(exists=path.exists(), thread=threading.get_ident())

    writer = CheckpointWriter()
    writer.submit(path, make_run_state("pt"), then)
    writer.drain()
    assert seen["exists"] is True
    assert seen["thread"] != threading.get_ident()


def test_a_failing_then_is_a_failed_save(tmp_path: Path) -> None:
    def then() -> None:
        raise ValueError("bad keep_last_n")

    writer = CheckpointWriter()
    writer.submit(_path(tmp_path, 1), make_run_state("pt"), then)
    with pytest.raises(RuntimeError, match="bad keep_last_n"):
        writer.drain()


def test_submit_waits_for_the_previous_write(tmp_path: Path) -> None:
    writer = CheckpointWriter()
    writer.submit(_path(tmp_path, 1), make_run_state("pt"), None)
    writer.submit(_path(tmp_path, 2), make_run_state("pt"), None)
    assert _path(tmp_path, 1).exists()
    writer.close()
    assert _path(tmp_path, 2).exists()


def test_no_thread_starts_until_something_is_submitted() -> None:
    """A non-writer rank holds a writer it never submits to; it must cost nothing."""
    writer = CheckpointWriter()
    writer.drain()
    writer.close()
    assert writer._executor is None


def test_close_raises_a_fatal_failure_and_stops_the_thread(tmp_path: Path) -> None:
    writer = CheckpointWriter()
    with patch.object(async_writer.codecs, "write", side_effect=OSError("disk full")):
        writer.submit(_path(tmp_path, 1), make_run_state("pt"), None)
        with pytest.raises(CheckpointWriteError, match="disk full"):
            writer.close()
    assert writer._executor is None


def test_each_surfaced_error_is_a_fresh_one_naming_the_failed_file(tmp_path: Path) -> None:
    writer = CheckpointWriter()
    with patch.object(async_writer.codecs, "write", side_effect=OSError("disk full")):
        writer.submit(_path(tmp_path, 1), make_run_state("pt"), None)
    errors = []
    for _ in range(2):
        with pytest.raises(CheckpointWriteError) as caught:
            writer.drain()
        errors.append(caught.value)
    assert errors[0] is not errors[1]
    assert all(e.path == _path(tmp_path, 1) for e in errors)
    assert isinstance(errors[0].__cause__, OSError)


def test_a_non_fatal_failure_is_kept_not_raised(tmp_path: Path) -> None:
    best = tmp_path / "checkpoints" / "best.pt"
    _path(tmp_path, 1)
    writer = CheckpointWriter()
    with patch.object(async_writer.codecs, "write", side_effect=OSError("disk full")):
        writer.submit(best, make_run_state("pt", role="best"), None, fatal=False)
    writer.submit(_path(tmp_path, 2), make_run_state("pt"), None)
    writer.close()
    assert _path(tmp_path, 2).exists()
    [failure] = writer.soft_failures
    assert (failure.role, failure.path) == ("best", best)
    assert failure.message.startswith("iter 150: ")
    assert str(best) in failure.message and "disk full" in failure.message


def test_snapshot_copies_tensors_keeps_containers_and_skips_sync_on_cpu() -> None:
    state = make_run_state("pt")
    with patch.object(torch.cuda, "synchronize") as sync:
        copy = snapshot_to_cpu(state)
    sync.assert_not_called()
    live = state["models"]["generator"]["0.weight"]
    stored = copy["models"]["generator"]["0.weight"]
    assert torch.equal(stored, live)
    assert stored.data_ptr() != live.data_ptr()
    assert type(copy["models"]["generator"]) is type(state["models"]["generator"])
    assert copy["rng"]["python"] == state["rng"]["python"]
    live_np, stored_np = state["rng"]["numpy"][1], copy["rng"]["numpy"][1]
    assert isinstance(stored_np, np.ndarray)
    assert np.array_equal(stored_np, live_np)
    assert not np.shares_memory(stored_np, live_np)
    assert copy["meta"] == state["meta"]


def test_a_quiet_close_logs_a_fatal_failure_and_stops_the_thread(tmp_path: Path, caplog) -> None:
    """Teardown while another exception propagates must not replace it."""
    writer = CheckpointWriter()
    with patch.object(async_writer.codecs, "write", side_effect=OSError("disk full")):
        writer.submit(_path(tmp_path, 1), make_run_state("pt"), None)
        with caplog.at_level(logging.ERROR, logger=async_writer.logger.name):
            writer.close(raise_fatal=False)
    assert writer._executor is None
    assert any("disk full" in r.getMessage() for r in caplog.records)


def test_a_second_close_neither_reraises_nor_relogs(tmp_path: Path, caplog) -> None:
    """The loop's exit-stack close runs after the normal-path close; it must do nothing."""
    writer = CheckpointWriter()
    with patch.object(async_writer.codecs, "write", side_effect=OSError("disk full")):
        writer.submit(_path(tmp_path, 1), make_run_state("pt"), None)
        with pytest.raises(CheckpointWriteError):
            writer.close()
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger=async_writer.logger.name):
        writer.close(raise_fatal=False)
        writer.close()
    assert not caplog.records
    assert writer._executor is None


def test_a_submit_after_close_is_closed_again(tmp_path: Path) -> None:
    writer = CheckpointWriter()
    writer.close()
    writer.submit(_path(tmp_path, 1), make_run_state("pt"), None)
    writer.close()
    assert _path(tmp_path, 1).exists() and writer._executor is None


def test_a_state_without_meta_is_a_classified_failure_not_a_stuck_one(tmp_path: Path) -> None:
    writer = CheckpointWriter()
    writer.submit(_path(tmp_path, 1), {"models": {}}, None, fatal=False)
    writer.submit(_path(tmp_path, 2), {"models": {}}, None)
    with pytest.raises(CheckpointWriteError, match="meta"):
        writer.close()
    assert len(writer.soft_failures) == 1


def test_an_ordered_dict_keeps_its_metadata() -> None:
    sd = OrderedDict(w=torch.ones(2))
    sd._metadata = OrderedDict({"": {"version": 1}})
    copy = snapshot_to_cpu({"models": {"g": sd}})["models"]["g"]
    assert copy._metadata == sd._metadata
    assert copy["w"].data_ptr() != sd["w"].data_ptr()


def test_a_namedtuple_holding_a_tensor_is_copied() -> None:
    point = namedtuple("point", "x y")
    live = point(torch.ones(2), 3)
    copy = snapshot_to_cpu({"p": live})["p"]
    assert type(copy) is point and copy.y == 3
    assert copy.x.data_ptr() != live.x.data_ptr()


def test_an_unrecognised_container_holding_a_tensor_raises() -> None:
    """Planted violation of rule 3: a Mapping subclass would be written while it is mutated."""

    class Holder(dict):
        pass

    with pytest.raises(TypeError, match="Holder"):
        snapshot_to_cpu({"state": Holder(w=torch.ones(2))})
    with pytest.raises(TypeError, match="set"):
        snapshot_to_cpu({"state": {torch.ones(1)}})
    assert snapshot_to_cpu({"state": Holder(w=1)})["state"]["w"] == 1
