"""keep_last_n over step files of every format, and their native tag directories."""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.infrastructure.checkpointing.layout import CheckpointLayout
from spectramr.infrastructure.checkpointing.retention import prune, validate_keep_last_n


def _layout(tmp_path: Path, *names: str) -> CheckpointLayout:
    layout = CheckpointLayout.for_run(tmp_path)
    layout.ensure()
    for name in names:
        (layout.directory / name).write_bytes(b"x")
    return layout


def _names(layout: CheckpointLayout) -> list[str]:
    return sorted(p.name for p in layout.directory.iterdir())


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2", None])
def test_keep_last_n_must_be_an_int_of_at_least_one(bad) -> None:
    with pytest.raises(ValueError, match="keep_last_n"):
        validate_keep_last_n(bad)


def test_keeps_the_newest_n_steps_and_never_best(tmp_path) -> None:
    layout = _layout(
        tmp_path, "step_00000100.pt", "step_00000200.pt", "step_00000300.pt", "best.pt"
    )
    removed = prune(layout, layout.directory / "step_00000300.pt", keep_last_n=2)
    assert [p.name for p in removed] == ["step_00000100.pt"]
    assert _names(layout) == ["best.pt", "step_00000200.pt", "step_00000300.pt"]


def test_a_format_change_between_resumes_loses_nothing_it_should_keep(tmp_path) -> None:
    layout = _layout(tmp_path, "step_00000100.pt", "step_00000200.pt", "step_00000300.safetensors")
    assert prune(layout, layout.directory / "step_00000300.safetensors", keep_last_n=3) == []
    removed = prune(layout, layout.directory / "step_00000300.safetensors", keep_last_n=2)
    assert [p.name for p in removed] == ["step_00000100.pt"]


def test_two_formats_of_one_old_step_go_together(tmp_path) -> None:
    layout = _layout(tmp_path, "step_00000100.pt", "step_00000100.safetensors", "step_00000200.pt")
    removed = prune(layout, layout.directory / "step_00000200.pt", keep_last_n=1)
    assert sorted(p.name for p in removed) == ["step_00000100.pt", "step_00000100.safetensors"]


def test_steps_above_the_one_just_written_are_never_ranked(tmp_path) -> None:
    layout = _layout(tmp_path, "step_00000100.pt", "step_00000500.pt", "step_00000200.pt")
    removed = prune(layout, layout.directory / "step_00000200.pt", keep_last_n=1)
    assert [p.name for p in removed] == ["step_00000100.pt"]
    assert "step_00000500.pt" in _names(layout)


def test_just_written_must_be_a_step_file(tmp_path) -> None:
    layout = _layout(tmp_path, "best.pt")
    with pytest.raises(ValueError, match="step checkpoint"):
        prune(layout, layout.directory / "best.pt", keep_last_n=1)


def test_native_tag_dirs_follow_their_step_and_latest_is_protected(tmp_path) -> None:
    layout = _layout(tmp_path, "step_00000100.pt", "step_00000200.pt", "step_00000300.pt")
    root = layout.directory / "deepspeed" / "generator"
    for tag in ("step_00000100", "step_00000200", "step_00000300"):
        (root / tag).mkdir(parents=True)
    (root / "latest").write_text("step_00000200")
    removed = prune(
        layout, layout.directory / "step_00000300.pt", keep_last_n=1, native_roots=[root]
    )
    assert sorted(p.name for p in removed) == ["step_00000100", "step_00000100.pt"]
    assert sorted(p.name for p in root.iterdir()) == ["latest", "step_00000200", "step_00000300"]


def test_nothing_is_deleted_when_this_rank_cannot_see_the_new_tag(tmp_path) -> None:
    layout = _layout(tmp_path, "step_00000100.pt", "step_00000200.pt")
    root = layout.directory / "deepspeed" / "generator"
    root.mkdir(parents=True)
    assert (
        prune(layout, layout.directory / "step_00000200.pt", keep_last_n=1, native_roots=[root])
        == []
    )
