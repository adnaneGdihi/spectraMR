"""Paired tests for ``scripts/verification/inspect_checkpoint.py``.

The script's job is to say, of a real artifact, which groups a resume would
restore. Its own failure mode is the one that matters: a report that prints
PRESENT for a key the file does not carry is worse than no report, because it
is the thing being consulted precisely when the checkpoint is in doubt. So each
case below plants an absence and asserts the report names it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tests.utils.repo_scripts import require_repo_file

_SCRIPT = require_repo_file("scripts/verification/inspect_checkpoint.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("inspect_checkpoint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inspect_checkpoint = _load_module()


def _complete_blob() -> dict:
    """The key set CheckpointDirector.save writes for a GAN arm."""
    return {
        "epoch": 3,
        "global_step": 40_000,
        "generator": {"w": torch.zeros(4, 4), "b": torch.zeros(4)},
        "optimizer_g": {"param_groups": [{"lr": 2e-4}], "state": {0: {}}},
        "scheduler_g": {"last_epoch": 3, "_step_count": 4},
        "counter_state": {"current_step": 40_000, "current_epoch": 3},
        "rng_state": {"torch": torch.zeros(8, dtype=torch.uint8), "numpy": (), "python": ()},
        "metrics": {},
    }


def _write(tmp_path: Path, blob: dict) -> Path:
    path = tmp_path / "checkpoint_epoch_0003_step_040000.pt"
    torch.save(blob, path)
    return path


@pytest.mark.unit
def test_complete_checkpoint_reports_ok(tmp_path, capsys):
    exit_code = inspect_checkpoint.report(_write(tmp_path, _complete_blob()))
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "global_step=40000" in out
    assert "OK: every group a resume needs" in out


@pytest.mark.unit
@pytest.mark.parametrize("dropped", ["generator", "optimizer_g", "global_step"])
def test_a_missing_required_group_fails_loudly(tmp_path, capsys, dropped):
    """The planted violation: a checkpoint that cannot continue a run."""
    blob = _complete_blob()
    del blob[dropped]

    exit_code = inspect_checkpoint.report(_write(tmp_path, blob))
    out = capsys.readouterr().out

    assert exit_code == 1
    assert f"{dropped:<20} MISSING" in out
    assert "cannot resume" in out


@pytest.mark.unit
def test_absent_rng_state_is_reported_not_assumed(tmp_path, capsys):
    """The exact regression this script was written after.

    ``rng_state`` is optional -- older checkpoints predate it -- so its absence
    must not fail the run, but it must be visible, because a resume without it
    silently draws a different timestep/mask sequence.
    """
    blob = _complete_blob()
    del blob["rng_state"]

    exit_code = inspect_checkpoint.report(_write(tmp_path, blob))
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "rng_state            absent" in out


@pytest.mark.unit
def test_rng_summary_names_the_streams(tmp_path, capsys):
    blob = _complete_blob()
    blob["rng_state"]["cuda"] = [torch.zeros(8, dtype=torch.uint8)] * 2

    inspect_checkpoint.report(_write(tmp_path, blob))
    out = capsys.readouterr().out

    assert "torch, numpy, python, cuda[2]" in out


@pytest.mark.unit
def test_the_uncarried_half_is_always_stated(tmp_path, capsys):
    """A key list cannot show what no key covers, so the report says it."""
    inspect_checkpoint.report(_write(tmp_path, _complete_blob()))
    out = capsys.readouterr().out

    assert "NOT CARRIED BY ANY CHECKPOINT" in out
    assert "dataloader position" in out
    assert "the config" in out


@pytest.mark.unit
def test_a_bare_state_dict_is_refused(tmp_path):
    """A consolidated DeepSpeed export is not an envelope and must not be read
    as one -- reporting every group MISSING would misdiagnose it."""
    path = tmp_path / "consolidated.pt"
    torch.save([torch.zeros(2)], path)

    with pytest.raises(TypeError, match="not a checkpoint envelope"):
        inspect_checkpoint.report(path)


@pytest.mark.unit
def test_missing_file_is_an_argparse_error(tmp_path):
    with pytest.raises(SystemExit):
        inspect_checkpoint.main(["--checkpoint", str(tmp_path / "nope.pt")])
