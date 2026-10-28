"""Shared fixtures for the convert_checkpoints_v1 tests: frozen pre-1.0 runs, copied to tmp."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "checkpoints_pre_v1"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())
SCRIPT = REPO / "scripts" / "migrations" / "convert_checkpoints_v1.py"


def converter():
    if str(SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("convert_checkpoints_v1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_run(tmp_path: Path, name: str) -> Path:
    """Copy a frozen fixture run under ``tmp_path``; the converter never sees the original."""
    target = tmp_path / name
    shutil.copytree(FIXTURES / name, target, symlinks=True)
    return target


def ck_names(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / "checkpoints").iterdir())


def resume_cfg():
    return SimpleNamespace(
        checkpoint=SimpleNamespace(
            load_optimizer_state=True,
            load_scheduler_state=True,
            allow_world_size_change=False,
            allow_parallel_strategy_change=True,
        )
    )


def snapshot(run_dir: Path) -> dict[str, tuple[str, str]]:
    """Every entry under ``checkpoints/``: a symlink's target, a file's sha256, or a dir."""
    out = {}
    root = run_dir / "checkpoints"
    for path in sorted(root.rglob("*")):
        rel = str(path.relative_to(root))
        if path.is_symlink():
            out[rel] = ("link", os.readlink(path))
        elif path.is_file():
            out[rel] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            out[rel] = ("dir", "")
    return out


def listing(run_dir: Path) -> dict[str, tuple]:
    """Every entry under ``run_dir``, recursively, with its kind, size, mtime and content.

    A refusal must leave this unchanged: ``snapshot`` alone would miss a file rewritten
    with the same bytes. A directory's own mtime is left out: a round-trip refusal must
    write its staging directory before it can know the result, then removes it, which
    moves the parent's mtime without touching any entry.
    """
    out = {}
    for path in sorted([run_dir, *run_dir.rglob("*")]):
        st = path.lstat()
        rel = str(path.relative_to(run_dir))
        if path.is_symlink():
            out[rel] = ("link", os.readlink(path), st.st_mtime_ns)
        elif path.is_dir():
            out[rel] = ("dir",)
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out[rel] = ("file", st.st_size, st.st_mtime_ns, digest)
    return out
