"""Paired tests for ``scripts/verification/inspect_checkpoint.py``.

The script's job is to say, of a real artifact, which RunState parts a resume would
restore. Its own failure mode is the one that matters: a report that prints PRESENT for
a part the file does not carry is worse than no report, because it is the thing being
consulted precisely when the checkpoint is in doubt. So each case below plants an
absence and asserts the report names it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.payload import build_meta
from tests.fixtures.run_state_files import write_run_state
from tests.utils.repo_scripts import require_repo_file

_SCRIPT = require_repo_file("scripts/verification/inspect_checkpoint.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("inspect_checkpoint", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inspect_checkpoint = _load_module()

_GENERATOR = {"w": torch.zeros(4, 4), "b": torch.zeros(4)}  # 2 tensors, 80 bytes


def _parts(**drop: bool) -> dict:
    """The optional parts a GAN step file carries; ``drop`` names the ones to leave out."""
    parts = {
        "optim": {"generator": {"param_groups": [{"lr": 2e-4}], "state": {}}},
        "sched": {"generator": {"last_epoch": 3, "_step_count": 4}},
        "rng": {"torch": torch.zeros(8, dtype=torch.uint8)},
    }
    return {k: v for k, v in parts.items() if not drop.get(k)}


def _write(tmp_path: Path, *, models: dict | None = None, ema: bool = True, **drop: bool) -> Path:
    return write_run_state(
        tmp_path / "step_00000010.pt",
        models if models is not None else {"generator": _GENERATOR, "discriminator": {"d": torch.zeros(2)}},
        ema={"w": torch.zeros(4, 4), "b": torch.zeros(4)} if ema else None,
        role="step",
        extra_parts=_parts(**drop),
    )


@pytest.mark.unit
def test_complete_checkpoint_reports_ok(tmp_path, capsys):
    exit_code = inspect_checkpoint.report(_write(tmp_path))
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "role=step" in out and "step=10" in out
    assert "OK: every part a resume needs" in out


@pytest.mark.unit
def test_each_part_reports_its_tensor_count_and_bytes(tmp_path, capsys):
    inspect_checkpoint.report(_write(tmp_path))
    lines = {line.strip() for line in capsys.readouterr().out.splitlines()}

    # ema holds the shadow's two tensors plus its integer update counter.
    assert f"{'models.generator':<22} PRESENT  2 tensors, 80 bytes" in lines
    assert f"{'models.discriminator':<22} PRESENT  1 tensors, 8 bytes" in lines
    assert f"{'ema':<22} PRESENT  2 tensors, 80 bytes" in lines


@pytest.mark.unit
def test_a_missing_optimizer_fails_loudly(tmp_path, capsys):
    """The planted violation: a checkpoint that cannot continue a run."""
    exit_code = inspect_checkpoint.report(_write(tmp_path, optim=True))
    out = capsys.readouterr().out

    assert exit_code == 1
    assert f"{'optim':<22} MISSING" in out
    assert "cannot resume" in out


@pytest.mark.unit
def test_a_missing_generator_fails_loudly(tmp_path, capsys):
    exit_code = inspect_checkpoint.report(
        _write(tmp_path, models={"discriminator": {"d": torch.zeros(2)}})
    )
    out = capsys.readouterr().out

    assert exit_code == 1
    assert f"{'models.generator':<22} MISSING" in out


@pytest.mark.unit
def test_an_engine_held_optimizer_is_not_missing(tmp_path, capsys):
    """Under DeepSpeed the optimizer lives in the native tag directory, so the
    file's ``optim`` part is empty by design."""
    path = tmp_path / "step_00000010.pt"
    meta = build_meta(
        role="step",
        reason="periodic",
        epoch=1,
        step=10,
        fmt="pt",
        topology={"world_size": 2, "parallel_strategy": "deepspeed"},
        native_tag="step_00000010",
    )
    codecs.write(path, {"meta": meta, "models": {"generator": _GENERATOR}, "optim": {}})

    exit_code = inspect_checkpoint.report(path)
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "native tag step_00000010" in out


@pytest.mark.unit
def test_absent_rng_is_reported_not_assumed(tmp_path, capsys):
    """``rng`` is optional, so its absence must not fail the run, but it must be
    visible: a resume without it draws a different timestep/mask sequence."""
    exit_code = inspect_checkpoint.report(_write(tmp_path, rng=True))
    out = capsys.readouterr().out

    assert exit_code == 0
    assert f"{'rng':<22} absent" in out


@pytest.mark.unit
def test_the_uncarried_half_is_always_stated(tmp_path, capsys):
    """A part list cannot show what no part covers, so the report says it."""
    inspect_checkpoint.report(_write(tmp_path))
    out = capsys.readouterr().out

    assert "NOT CARRIED BY ANY CHECKPOINT" in out
    assert "dataloader position" in out
    assert "the config" in out


@pytest.mark.unit
def test_a_pre_v1_file_exits_2_naming_the_converter(tmp_path, capsys):
    path = tmp_path / "checkpoint_epoch_0003_step_040000.pt"
    torch.save({"generator": _GENERATOR, "global_step": 40_000}, path)

    assert inspect_checkpoint.main(["--checkpoint", str(path)]) == 2
    assert "convert_checkpoints_v1" in capsys.readouterr().err


@pytest.mark.unit
def test_missing_file_is_an_argparse_error(tmp_path):
    with pytest.raises(SystemExit):
        inspect_checkpoint.main(["--checkpoint", str(tmp_path / "nope.pt")])
