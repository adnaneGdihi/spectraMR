"""A conversion interrupted while publishing finishes on a rerun, from its journal (I1, I2).

The refusals a rerun can meet are in ``test_convert_checkpoints_v1_journal_refusals.py``.
"""

from __future__ import annotations

import os
import sys

import pytest

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.layout import CheckpointLayout
from spectramr.infrastructure.checkpointing.select import ConversionPendingError, select
from tests.unit.scripts.convert_v1_helpers import converter, copy_run, snapshot

JOURNAL = ".convert_v1.journal.json"
STAGING = ".convert_v1_staging"


class SimulatedCrashError(Exception):
    pass


def _crash_on(monkeypatch, op: str, n: int) -> None:
    """Raise at the ``n``-th journal step of kind ``op``, the way a killed process stops."""
    journal = sys.modules["_checkpoint_v1_journal"]
    real, seen = journal.apply_step, {"n": 0}

    def apply_step(step, ck):
        if step["op"] == op:
            seen["n"] += 1
            if seen["n"] == n:
                raise SimulatedCrashError(f"{op} #{n}")
        return real(step, ck)

    monkeypatch.setattr(journal, "apply_step", apply_step)


def _crash(tmp_path, monkeypatch, name, fmt, op, n):
    conv = converter()
    run_dir = copy_run(tmp_path / "crashed", name)
    _crash_on(monkeypatch, op, n)
    assert conv.main([str(run_dir), "--format", fmt]) == 1
    monkeypatch.undo()
    assert (run_dir / "checkpoints" / JOURNAL).is_file()
    return conv, run_dir


@pytest.mark.parametrize(
    ("name", "fmt", "op", "n"),
    [
        ("director_run", "safetensors", "publish", 2),  # during the step renames
        ("director_run", "pt", "publish", 3),  # at best, which replaces the best.pt alias
        ("deepspeed_run", "safetensors", "tag", 1),  # before the tag rename
        ("deepspeed_run", "safetensors", "latest", 1),  # tags renamed, latest not rewritten
        ("director_run", "safetensors", "unlink", 2),  # during the sweep
        ("deepspeed_run", "pt", "unlink", 3),  # during the sweep, sidecar still present
    ],
)
def test_a_rerun_finishes_an_interrupted_publish(tmp_path, monkeypatch, capsys, name, fmt, op, n):
    conv, run_dir = _crash(tmp_path, monkeypatch, name, fmt, op, n)
    if op != "unlink":  # nothing original is removed before the sweep
        legacy = {k for k in snapshot(copy_run(tmp_path / "orig", name)) if "checkpoint_" in k}
        assert legacy <= set(snapshot(run_dir))
    capsys.readouterr()
    assert conv.main([str(run_dir), "--format", fmt]) == 0
    assert "resumed a partial conversion" in capsys.readouterr().out
    reference = copy_run(tmp_path / "reference", name)
    assert conv.main([str(reference), "--format", fmt]) == 0
    _assert_same_final_state(run_dir, reference)


def _assert_same_final_state(run_dir, reference) -> None:
    """Same entries; converted files equal part by part (pt bytes embed a temp name)."""
    got, want = snapshot(run_dir), snapshot(reference)
    assert set(got) == set(want)
    commit = sys.modules["_checkpoint_v1_commit"]
    for rel, (kind, _digest) in want.items():
        if kind == "file" and rel.startswith(("step_", "best.")):
            ck, ref = run_dir / "checkpoints" / rel, reference / "checkpoints" / rel
            commit.assert_same(codecs.read(ref), codecs.read(ck), rel)
        else:
            assert got[rel] == want[rel], rel


@pytest.mark.parametrize("n", [1, 2, 3])
def test_best_is_published_after_every_step_file(tmp_path, monkeypatch, n) -> None:
    """I2: a partial publish must never leave best as the newest resumable file."""
    _conv, run_dir = _crash(tmp_path, monkeypatch, "director_run", "safetensors", "publish", n)
    published = CheckpointLayout.for_run(run_dir).entries()
    assert all(e.role == "step" for e in published)  # no best before every step
    with pytest.raises(ConversionPendingError):  # and no reader picks from a partial set
        select(run_dir, "latest")


def test_staged_files_and_their_directories_are_synced_before_the_journal(tmp_path, monkeypatch):
    conv = converter()
    run_dir = copy_run(tmp_path, "director_run")
    synced, seen = [], {}
    real_fsync, real_write = os.fsync, conv.write_journal

    def fsync(fd):
        synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        return real_fsync(fd)

    def write_journal(ck, steps, **kwargs):
        seen["synced"] = list(synced)
        return real_write(ck, steps, **kwargs)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(conv, "write_journal", write_journal)
    assert conv.main([str(run_dir)]) == 0
    staging = run_dir / "checkpoints" / STAGING
    names = ["step_00000002", "step_00000004", "best"]
    want = {str(staging / f"{n}.converting.safetensors") for n in names}
    want |= {str(staging), str(run_dir / "checkpoints")}
    assert want <= set(seen["synced"])
