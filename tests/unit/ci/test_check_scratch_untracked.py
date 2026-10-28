"""The scratch gate must turn red on each shape of tracked scratch file, and only on those."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess

import pytest

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3] / "scripts" / "ci" / "check_scratch_untracked.py"
)


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("check_scratch_untracked", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "path", ["scratch/probe.py", "scratch/sub/dir/out.png", "scratch/.gitkeep"]
)
def test_flags_every_tracked_shape(gate, path):
    assert gate.tracked_under_scratch([path, "src/spectramr/x.py"]) == [path]


@pytest.mark.parametrize(
    "path", ["tools/scratch/x.py", "scratch_notes.md", "scripts/scratchpad.py", "src/scratch.py"]
)
def test_ignores_lookalikes(gate, path):
    assert gate.tracked_under_scratch([path]) == []


def _repo(tmp_path, tracked):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel in tracked:
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
        subprocess.run(["git", "-C", str(tmp_path), "add", "-f", rel], check=True)


def test_main_exits_1_on_planted_tracked_file(gate, tmp_path):
    _repo(tmp_path, ["scratch/leftover.py"])
    assert gate.main(tmp_path) == 1


def test_main_exits_0_on_clean_tree(gate, tmp_path):
    _repo(tmp_path, ["tools/probe.py"])
    assert gate.main(tmp_path) == 0
