"""python -m spectramr.cli.resolve_checkpoint: best, else the latest step, of a run."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from spectramr.cli.resolve_checkpoint import main
from tests.fixtures.run_state_files import write_run_state


def _v1(path: Path, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    role = "best" if path.stem == "best" else "step"
    write_run_state(path, {"generator": {"w": torch.zeros(1)}}, role=role)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _leftover(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_prefers_the_best_checkpoint(tmp_path, capsys) -> None:
    _v1(tmp_path / "checkpoints" / "step_00002000.pt")
    best = _v1(tmp_path / "checkpoints" / "best.safetensors")
    assert main([str(tmp_path / "checkpoints")]) == 0
    assert capsys.readouterr().out.strip() == str(best)


def test_without_a_best_the_latest_step_is_printed(tmp_path, capsys) -> None:
    _v1(tmp_path / "checkpoints" / "step_00002000.pt", mtime=1_000)
    latest = _v1(tmp_path / "checkpoints" / "step_00003000.safetensors")
    assert main([str(tmp_path / "checkpoints")]) == 0
    assert capsys.readouterr().out.strip() == str(latest)


@pytest.mark.parametrize("leftover", [None, "checkpoint_epoch_0002_step_002000.pt"])
def test_exits_non_zero_when_nothing_qualifies(tmp_path, capsys, leftover) -> None:
    (tmp_path / "checkpoints").mkdir()
    if leftover is not None:
        _leftover(tmp_path / "checkpoints" / leftover)
    assert main([str(tmp_path / "checkpoints")]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no checkpoint" in captured.err
    if leftover is not None:
        assert leftover in captured.err
        assert "convert_checkpoints_v1.py" in captured.err
    else:
        assert "convert_checkpoints_v1.py" not in captured.err


def test_a_long_pre_v1_list_is_truncated_like_resume(tmp_path, capsys) -> None:
    for step in range(7):
        _leftover(tmp_path / "checkpoints" / f"checkpoint_step_{step}.pt")
    assert main([str(tmp_path / "checkpoints")]) == 1
    assert "and 2 more" in capsys.readouterr().err


def test_a_pre_v1_file_beside_a_v1_step_is_ignored(tmp_path, capsys) -> None:
    step = _v1(tmp_path / "checkpoints" / "step_00000010.pt", mtime=1_000)
    _leftover(tmp_path / "checkpoints" / "checkpoint_step_9000.pt")
    assert main([str(tmp_path / "checkpoints")]) == 0
    assert capsys.readouterr().out.strip() == str(step)


def test_a_missing_directory_exits_non_zero(tmp_path, capsys) -> None:
    assert main([str(tmp_path / "absent" / "checkpoints")]) == 1
    assert "no checkpoint" in capsys.readouterr().err


def test_an_argument_that_is_not_a_checkpoints_dir_is_refused(tmp_path, capsys) -> None:
    """Planted: the run dir itself; reading its parent would select a sibling run."""
    _v1(tmp_path / "checkpoints" / "best.pt")
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "run")])
    assert exc.value.code == 2
    assert "checkpoints/ directory" in capsys.readouterr().err


def test_an_interrupted_conversion_exits_non_zero_naming_the_converter(tmp_path, capsys) -> None:
    ck = tmp_path / "checkpoints"
    _v1(ck / "step_00000002.pt")
    _leftover(ck / "checkpoint_best.pt")
    (ck / ".convert_v1.journal.json").write_text("{}")
    assert main([str(ck)]) == 1
    err = capsys.readouterr().err
    assert "partial checkpoint conversion" in err
    assert f"scripts/migrations/convert_checkpoints_v1.py {tmp_path}" in err
