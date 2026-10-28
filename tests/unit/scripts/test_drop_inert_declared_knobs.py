"""The inert-knob drainer edits text; what it must not touch is the interesting half."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "migrations" / "drop_inert_declared_knobs.py"
)


@pytest.fixture(scope="module")
def mig():
    spec = importlib.util.spec_from_file_location("_drop_inert_mig", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARM = """\
config_version: '1.0'
training:
  # why the workers are pinned
  num_workers: 8
  diffusion:
    timesteps: 29
    num_workers: 3
  batch_size: 2
data:
  num_workers: 1
"""


def _run(mig, tmp_path, text, targets, apply=True):
    path = tmp_path / "arm.yaml"
    path.write_text(text)
    return mig.process(path, targets, apply=apply), path.read_text()


def test_only_the_named_block_loses_the_key(mig, tmp_path) -> None:
    status, out = _run(mig, tmp_path, ARM, ("training.num_workers",))
    assert status.startswith("DROPPED")
    assert "# why the workers are pinned" not in out
    assert "    num_workers: 3" in out  # training.diffusion keeps its own
    assert "  num_workers: 1" in out  # data keeps its own
    assert "  batch_size: 2" in out


def test_a_block_valued_key_goes_with_its_children(mig, tmp_path) -> None:
    text = "config_version: '1.0'\nartifacts:\n  persistent_root: x\n  # tail\nrun:\n  seed: 1\n"
    status, out = _run(mig, tmp_path, text, ("artifacts",))
    assert status.startswith("DROPPED")
    assert out == "config_version: '1.0'\n  # tail\nrun:\n  seed: 1\n"


def test_a_flow_style_declaration_is_a_miss_not_a_silent_ok(mig, tmp_path) -> None:
    text = "config_version: '1.0'\ntraining: {num_workers: 8, batch_size: 2}\n"
    status, out = _run(mig, tmp_path, text, ("training.num_workers",))
    assert status.startswith("MISS")
    assert out == text


def test_an_undeclared_key_is_nothing_to_drop(mig, tmp_path) -> None:
    status, out = _run(mig, tmp_path, ARM, ("checkpoint.save_on_epoch",))
    assert status.startswith("OK") and out == ARM


def test_a_dry_run_writes_nothing(mig, tmp_path) -> None:
    status, out = _run(mig, tmp_path, ARM, ("training.num_workers",), apply=False)
    assert status.startswith("WOULD") and out == ARM


def test_the_cli_fails_on_a_miss(mig, tmp_path) -> None:
    (tmp_path / "a.yaml").write_text("training: {num_workers: 8}\n")
    assert mig.main([str(tmp_path), "--set", "issue_557"]) == 1


@pytest.mark.parametrize(
    "name", ["issue_568", "issue_563", "issue_557", "issue_306", "issue_1972"]
)
def test_every_set_names_dotted_paths(mig, name) -> None:
    assert all("." in p for p in mig.TARGET_SETS[name])


def test_the_live_checkpointing_key_is_never_touched(mig, tmp_path) -> None:
    text = (
        "training:\n  batch_size: 2\n  enable_gradient_checkpointing: true\n"
        "optimization:\n  gradient:\n    enable_checkpointing: false\n"
    )
    status, out = _run(mig, tmp_path, text, mig.TARGET_SETS["issue_1135"])
    assert status.startswith("DROPPED")
    assert out == (
        "training:\n  batch_size: 2\n"
        "optimization:\n  gradient:\n    enable_checkpointing: false\n"
    )


def test_a_file_at_another_config_version_is_left_alone(mig, tmp_path) -> None:
    text = "config_version: '5.0'\nartifacts:\n  persistent_root: x\n"
    path = tmp_path / "arm.yaml"
    path.write_text(text)
    status = mig.process(path, mig.TARGET_SETS["issue_698"], apply=True, version="1.0")
    assert status.startswith("OK") and path.read_text() == text
    v1 = "config_version: '1.0'\nartifacts:\n  persistent_root: x\n"
    path.write_text(v1)
    assert mig.process(path, ("artifacts",), apply=True, version="1.0").startswith("DROPPED")
    assert path.read_text() == "config_version: '1.0'\n"


def test_keep_best_n_goes_and_keep_last_n_stays(mig, tmp_path) -> None:
    """#1972: the retention key that is read sits next to the one that is not."""
    text = (
        "config_version: '1.0'\ncheckpoint:\n  save_interval: 500\n  keep_last_n: 3\n"
        "  # ranked best files\n  keep_best_n: 2\n"
    )
    status, out = _run(mig, tmp_path, text, mig.TARGET_SETS["issue_1972"])
    assert status.startswith("DROPPED")
    assert out == "config_version: '1.0'\ncheckpoint:\n  save_interval: 500\n  keep_last_n: 3\n"


def test_checkpoint_dir_goes_with_its_note_and_other_save_dirs_stay(mig, tmp_path) -> None:
    text = (
        "config_version: '1.0'\n"
        "validation:\n  save_dir: ./val\n"
        "checkpoint:\n  save_interval: 2000\n"
        "  # The key only sets the directory the pre-1.0 service creates at startup.\n"
        "  checkpoint_dir: experiments/results/arm/checkpoints\n"
        "  keep_last_n: 5\n"
    )
    status, out = _run(mig, tmp_path, text, mig.TARGET_SETS["issue_2842"])
    assert status.startswith("DROPPED")
    assert out == (
        "config_version: '1.0'\n"
        "validation:\n  save_dir: ./val\n"
        "checkpoint:\n  save_interval: 2000\n  keep_last_n: 5\n"
    )


def test_the_alias_spelling_is_drained_too(mig, tmp_path) -> None:
    text = "config_version: '1.0'\ncheckpoint:\n  save_dir: ./checkpoints\n  enabled: true\n"
    status, out = _run(mig, tmp_path, text, mig.TARGET_SETS["issue_2842"])
    assert status.startswith("DROPPED")
    assert out == "config_version: '1.0'\ncheckpoint:\n  enabled: true\n"
