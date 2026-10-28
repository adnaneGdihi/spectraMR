"""A rerun that cannot finish a journal refuses, changes nothing, and prints an inventory.

The roll-forward reaches every state a crash alone can leave (the crash tests in
``test_convert_checkpoints_v1_recovery.py``), so each state here needs another writer:
a training run resumed mid-conversion, or a user editing ``checkpoints/``. The refusal
prescribes no recovery; it lists what is on disk. Each test compares a recursive listing
with mtimes before and after, and pins the inventory.
"""

from __future__ import annotations

import os
import re
import shutil

import pytest

from spectramr.infrastructure.checkpointing import codecs
from tests.unit.scripts.convert_v1_helpers import converter, copy_run, listing
from tests.unit.scripts.test_convert_checkpoints_v1_recovery import (
    JOURNAL,
    STAGING,
    _crash,
    _crash_on,
)

LEGACY = (
    "checkpoint_best.pt, checkpoint_epoch_0000_step_000002.pt, checkpoint_epoch_0001_step_000004.pt"
)
#: The pre-1.0 ``best.pt`` symlink is listed too while it is still in place.
WITH_ALIAS = f"best.pt, {LEGACY}"


def _refused(conv, run_dir, capsys, *args: str) -> str:
    before = listing(run_dir)
    capsys.readouterr()
    assert conv.main([str(run_dir), *args]) == 1
    err = " ".join(capsys.readouterr().err.split())
    assert listing(run_dir) == before
    return err


def _assert_inventory(err, run_dir, reason, *, fmt, differ, staged, legacy=LEGACY) -> None:
    ck = run_dir / "checkpoints"
    assert reason in err
    assert "Nothing was changed and nothing was deleted. Do not remove the journal" in err
    assert f"journal: {ck / JOURNAL}; format recorded: {fmt};" in err
    assert f"staging files in {ck / STAGING}: {staged};" in err
    assert f"targets that differ from the journal: {differ};" in err
    assert f"legacy sources remaining: {legacy}." in err
    assert "--recover" not in err and not re.search(r"\bmove\b", err)  # not a recipe


def _foreign(path, source, *, bump: int = 7) -> None:
    """What a resumed training run writes at ``path``: a later, different schema-1 file."""
    state = codecs.read(source)
    state["models"]["generator"]["weight"] = state["models"]["generator"]["weight"] + bump
    state["meta"]["step"] = 100
    if path.is_symlink() or path.exists():
        path.unlink()
    codecs.write(path, state)


def test_a_staged_file_that_vanished(tmp_path, monkeypatch, capsys):
    conv, run_dir = _crash(tmp_path, monkeypatch, "director_run", "safetensors", "publish", 2)
    (run_dir / "checkpoints" / STAGING / "step_00000004.converting.safetensors").unlink()
    err = _refused(conv, run_dir, capsys)
    _assert_inventory(
        err,
        run_dir,
        "cannot be completed safely",
        fmt="safetensors",
        differ="step_00000004.safetensors (missing; no staged copy); "
        "best.safetensors (missing; a staged copy is present)",
        staged="best.converting.safetensors",
        legacy=WITH_ALIAS,
    )


def test_a_newer_best_before_best_was_published(tmp_path, monkeypatch, capsys):
    conv, run_dir = _crash(tmp_path, monkeypatch, "director_run", "pt", "publish", 1)
    ck = run_dir / "checkpoints"
    _foreign(ck / "best.pt", ck / STAGING / "best.converting.pt")
    err = _refused(conv, run_dir, capsys, "--format", "pt")
    _assert_inventory(
        err,
        run_dir,
        "best.pt exists and is not the alias",
        fmt="pt",
        differ="step_00000002.pt (missing; a staged copy is present); "
        "step_00000004.pt (missing; a staged copy is present); "
        "best.pt (sha256 differs from the journal; a staged copy is present)",
        staged="best.converting.pt, step_00000002.converting.pt, step_00000004.converting.pt",
    )


def test_a_newer_best_after_best_was_published(tmp_path, monkeypatch, capsys):
    conv, run_dir = _crash(tmp_path, monkeypatch, "director_run", "pt", "unlink", 1)
    best = run_dir / "checkpoints" / "best.pt"
    _foreign(best, best)
    err = _refused(conv, run_dir, capsys, "--format", "pt")
    _assert_inventory(
        err,
        run_dir,
        "cannot be completed safely",
        fmt="pt",
        differ="best.pt (sha256 differs from the journal; no staged copy)",
        staged="none",
    )


@pytest.mark.parametrize("link", [shutil.copy2, os.link])
def test_a_deleted_staged_best_beside_a_regular_file_alias(tmp_path, monkeypatch, capsys, link):
    conv = converter()
    run_dir = copy_run(tmp_path / "crashed", "director_run")
    ck = run_dir / "checkpoints"
    (ck / "best.pt").unlink()
    link(ck / "checkpoint_best.pt", ck / "best.pt")
    _crash_on(monkeypatch, "publish", 1)
    assert conv.main([str(run_dir), "--format", "pt"]) == 1
    monkeypatch.undo()
    (ck / STAGING / "best.converting.pt").unlink()
    err = _refused(conv, run_dir, capsys, "--format", "pt")
    _assert_inventory(
        err,
        run_dir,
        "cannot be completed safely",
        fmt="pt",
        differ="step_00000002.pt (missing; a staged copy is present); "
        "step_00000004.pt (missing; a staged copy is present); "
        "best.pt (sha256 differs from the journal; no staged copy)",
        staged="step_00000002.converting.pt, step_00000004.converting.pt",
    )


def test_a_step_a_resumed_run_wrote(tmp_path, monkeypatch, capsys):
    conv, run_dir = _crash(tmp_path, monkeypatch, "director_run", "safetensors", "publish", 2)
    ck = run_dir / "checkpoints"
    staged = ck / STAGING / "step_00000004.converting.safetensors"
    _foreign(ck / "step_00000004.safetensors", staged, bump=3)
    staged.unlink()
    err = _refused(conv, run_dir, capsys)
    _assert_inventory(
        err,
        run_dir,
        "step_00000004.safetensors: its staged file is missing",
        fmt="safetensors",
        differ="step_00000004.safetensors (sha256 differs from the journal; no staged copy); "
        "best.safetensors (missing; a staged copy is present)",
        staged="best.converting.safetensors",
        legacy=WITH_ALIAS,
    )


def test_a_staged_file_changed_after_it_was_verified(tmp_path, monkeypatch, capsys):
    conv, run_dir = _crash(tmp_path, monkeypatch, "director_run", "safetensors", "publish", 2)
    staged = run_dir / "checkpoints" / STAGING / "best.converting.safetensors"
    _foreign(staged, staged, bump=1)
    err = _refused(conv, run_dir, capsys)
    _assert_inventory(
        err,
        run_dir,
        "best.converting.safetensors changed since it was verified",
        fmt="safetensors",
        differ="step_00000004.safetensors (missing; a staged copy is present); "
        "best.safetensors (missing; a staged copy is present)",
        staged="best.converting.safetensors, step_00000004.converting.safetensors",
        legacy=WITH_ALIAS,
    )


def test_a_tag_that_exists_under_both_names(tmp_path, monkeypatch, capsys):
    conv, run_dir = _crash(tmp_path, monkeypatch, "deepspeed_run", "safetensors", "tag", 1)
    roots = sorted((run_dir / "checkpoints" / "deepspeed").iterdir())
    old = next(t for t in sorted(roots[0].iterdir()) if t.name.startswith("epoch_"))
    shutil.copytree(old, old.parent / f"step_{int(old.name.rsplit('_', 1)[1]):08d}")
    err = _refused(conv, run_dir, capsys)
    assert "both or neither exist" in err
    assert (
        "format recorded: safetensors;" in err
        and "targets that differ from the journal: none;" in err
    )


@pytest.mark.parametrize("content", ["", '{"version": 1, "ste', '{"version": 1}'])
def test_an_unreadable_journal(tmp_path, monkeypatch, capsys, content):
    conv, run_dir = _crash(tmp_path, monkeypatch, "director_run", "safetensors", "publish", 2)
    (run_dir / "checkpoints" / JOURNAL).write_text(content)
    err = _refused(conv, run_dir, capsys)
    unknown = "unknown (the journal is unreadable)"
    _assert_inventory(
        err,
        run_dir,
        "is unreadable or incomplete",
        fmt=unknown,
        differ=unknown,
        staged="best.converting.safetensors, step_00000004.converting.safetensors",
        legacy=WITH_ALIAS,
    )


def test_a_pending_journal_in_another_format(tmp_path, monkeypatch, capsys):
    conv, run_dir = _crash(tmp_path, monkeypatch, "director_run", "pt", "publish", 2)
    err = _refused(conv, run_dir, capsys)  # the default, safetensors
    _assert_inventory(
        err,
        run_dir,
        "rerun with --format pt to finish it",
        fmt="pt",
        differ="step_00000004.pt (missing; a staged copy is present); "
        "best.pt (a symlink, not the converted file; a staged copy is present)",
        staged="best.converting.pt, step_00000004.converting.pt",
        legacy=WITH_ALIAS,
    )
    assert conv.main([str(run_dir), "--format", "pt"]) == 0
