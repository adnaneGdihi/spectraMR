"""The run's checkpoint directory and the two names written into it."""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.infrastructure.checkpointing.layout import (
    FORMATS,
    CheckpointLayout,
    native_tag,
    parse,
    run_dir_of,
    validate_format,
)


def test_formats_are_the_three_the_spec_names() -> None:
    assert FORMATS == ("safetensors", "pt", "pth")


@pytest.mark.parametrize("bad", ["", "ckpt", "PT", ".pt", "bin"])
def test_unknown_format_raises(bad: str) -> None:
    with pytest.raises(ValueError, match="not one of"):
        validate_format(bad)


@pytest.mark.parametrize("run_dir", [None, "", "   "])
def test_layout_needs_a_run_dir_and_has_no_default(run_dir) -> None:
    with pytest.raises(ValueError, match="no default"):
        CheckpointLayout.for_run(run_dir)


def test_paths_are_under_run_dir_checkpoints(tmp_path: Path) -> None:
    layout = CheckpointLayout.for_run(tmp_path)
    assert layout.directory == tmp_path / "checkpoints"
    assert layout.step_path(1200, "pt") == tmp_path / "checkpoints" / "step_00001200.pt"
    assert layout.best_path("safetensors") == tmp_path / "checkpoints" / "best.safetensors"
    assert native_tag(1200) == "step_00001200"


@pytest.mark.parametrize("step", [-1, 1.0, True, "3"])
def test_step_must_be_a_non_negative_int(tmp_path: Path, step) -> None:
    with pytest.raises(ValueError, match="step"):
        CheckpointLayout.for_run(tmp_path).step_path(step, "pt")


def test_ensure_creates_only_the_checkpoint_dir(tmp_path: Path) -> None:
    run = tmp_path / "run"
    created = CheckpointLayout.for_run(run).ensure()
    assert created == run / "checkpoints" and created.is_dir()
    assert sorted(p.name for p in run.iterdir()) == ["checkpoints"]


def test_construction_touches_no_filesystem(tmp_path: Path) -> None:
    CheckpointLayout.for_run(tmp_path / "nowhere")
    assert not (tmp_path / "nowhere").exists()


def test_parse_reads_back_role_step_and_format(tmp_path: Path) -> None:
    (tmp_path / "step_00000100.pth").write_bytes(b"x")
    (tmp_path / "step_123456789.pt").write_bytes(b"x")  # wider than 8 digits
    (tmp_path / "best.safetensors").write_bytes(b"x")
    a = parse(tmp_path / "step_00000100.pth")
    b = parse(tmp_path / "step_123456789.pt")
    c = parse(tmp_path / "best.safetensors")
    assert (a.role, a.step, a.fmt) == ("step", 100, "pth")
    assert (b.role, b.step) == ("step", 123456789)
    assert (c.role, c.step, c.fmt) == ("best", None, "safetensors")


@pytest.mark.parametrize(
    "name",
    [
        "checkpoint_best.pt",  # pre-1.0 spelling
        "best.pt.tmp",  # an in-flight atomic write
        "step_100.pt",  # not zero-padded
        "step_00000100.bin",
        "manifest.json",
    ],
)
def test_parse_ignores_every_other_name(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"x")
    assert parse(tmp_path / name) is None


def test_parse_ignores_symlinks_and_directories(tmp_path: Path) -> None:
    (tmp_path / "step_00000001.pt").mkdir()
    (tmp_path / "target.pt").write_bytes(b"x")
    (tmp_path / "best.pt").symlink_to(tmp_path / "target.pt")
    assert parse(tmp_path / "step_00000001.pt") is None
    assert parse(tmp_path / "best.pt") is None


def test_entries_lists_every_format_and_nothing_else(tmp_path: Path) -> None:
    layout = CheckpointLayout.for_run(tmp_path)
    assert layout.entries() == []  # missing dir is empty, not an error
    layout.ensure()
    for name in ["step_00000100.pt", "step_00000200.safetensors", "best.pth", "notes.txt"]:
        (layout.directory / name).write_bytes(b"x")
    assert sorted(e.path.name for e in layout.entries()) == [
        "best.pth",
        "step_00000100.pt",
        "step_00000200.safetensors",
    ]


@pytest.mark.parametrize(
    ("path", "run_dir"),
    [
        ("res/s1/checkpoints/best.safetensors", Path("res/s1")),
        ("res/s1/checkpoints/step_00000050.pt", Path("res/s1")),
        ("res/s1/best.safetensors", None),
        ("res/s1/checkpoints/vae.pt", None),
    ],
)
def test_run_dir_of_reads_the_layout_from_the_path_alone(path: str, run_dir) -> None:
    assert run_dir_of(path) == run_dir
