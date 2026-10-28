"""One discovery rule for latest, best and a given step."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from spectramr.infrastructure.checkpointing.select import select


def _touch(directory: Path, name: str, mtime_ns: int | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"x")
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_unknown_selector_raises_even_on_an_empty_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selector"):
        select(tmp_path, "newest")


def test_empty_or_missing_run_selects_nothing(tmp_path: Path) -> None:
    assert select(tmp_path, "latest") is None
    assert select(tmp_path, "best") is None


def test_latest_is_the_highest_step_across_formats(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    _touch(ckpt, "step_00000100.pt")
    _touch(ckpt, "step_00000200.pt")
    newest = _touch(ckpt, "step_00000300.safetensors")  # format changed on resume
    _touch(ckpt, "best.pt")
    assert select(tmp_path, "latest") == newest


def test_latest_falls_back_to_best(tmp_path: Path) -> None:
    best = _touch(tmp_path / "checkpoints", "best.pth")
    assert select(tmp_path, "latest") == best


def test_best_prefers_the_newest_when_two_formats_coexist(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    _touch(ckpt, "best.pt", mtime_ns=1_000_000_000)
    fresh = _touch(ckpt, "best.safetensors", mtime_ns=2_000_000_000)
    assert select(tmp_path, "best") == fresh


def test_step_selector_is_exact(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    target = _touch(ckpt, "step_00000200.pt")
    _touch(ckpt, "step_00000300.pt")
    assert select(tmp_path, "step:200") == target
    assert select(tmp_path, "step:250") is None


def test_pre_1_0_names_are_never_selected(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    _touch(ckpt, "checkpoint_best.pt")
    _touch(ckpt, "checkpoint_epoch_0001_step_000100.pt")
    assert select(tmp_path, "latest") is None


def test_discovery_imports_no_torch() -> None:
    code = (
        "import sys\n"
        "import spectramr.infrastructure.checkpointing.select\n"
        "import spectramr.infrastructure.checkpointing.ref\n"
        "import spectramr.infrastructure.checkpointing.layout\n"
        "assert 'torch' not in sys.modules, 'discovery pulled torch'\n"
    )
    env = {**os.environ, "SPECTRAMR_SUPPRESS_CLINICAL_WARNING": "1"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_pre_v1_names_lists_only_pre_v1_files(tmp_path: Path) -> None:
    from spectramr.infrastructure.checkpointing.select import pre_v1_names

    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for name in (
        "checkpoint_epoch_0001_step_000004.pt",
        "checkpoint_step_4.safetensors",
        "step_00000004.pt",
        "notes.txt",
    ):
        (ck / name).write_bytes(b"x")
    (ck / "best.pt").symlink_to("checkpoint_epoch_0001_step_000004.pt")
    assert pre_v1_names(tmp_path) == [
        "best.pt",
        "checkpoint_epoch_0001_step_000004.pt",
        "checkpoint_step_4.safetensors",
    ]
    assert select(tmp_path, "latest") == ck / "step_00000004.pt"


def test_pre_v1_names_of_a_missing_dir_is_empty(tmp_path: Path) -> None:
    from spectramr.infrastructure.checkpointing.select import pre_v1_names

    assert pre_v1_names(tmp_path / "nowhere") == []


def test_conversion_pending_names_the_journal_of_an_interrupted_conversion(tmp_path: Path) -> None:
    from spectramr.infrastructure.checkpointing.select import (
        CONVERSION_JOURNAL,
        conversion_pending,
    )

    assert conversion_pending(tmp_path) is None
    (tmp_path / "checkpoints").mkdir()
    assert conversion_pending(tmp_path) is None
    journal = tmp_path / "checkpoints" / CONVERSION_JOURNAL
    journal.write_text("{}")
    assert conversion_pending(tmp_path) == journal


@pytest.mark.parametrize("which", ["latest", "best", "step:2"])
def test_select_refuses_a_half_published_conversion(tmp_path: Path, which: str) -> None:
    """One owner (NN17): every reader that selects inherits the refusal."""
    from spectramr.infrastructure.checkpointing.select import (
        CONVERSION_JOURNAL,
        ConversionPendingError,
        select,
    )

    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "step_00000002.pt").write_bytes(b"x")
    (ck / CONVERSION_JOURNAL).write_text("{}")
    with pytest.raises(ConversionPendingError, match="partial checkpoint conversion") as info:
        select(tmp_path, which)
    assert isinstance(info.value, RuntimeError)
    assert f"scripts/migrations/convert_checkpoints_v1.py {tmp_path}" in str(info.value)
