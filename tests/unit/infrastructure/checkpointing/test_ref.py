"""A checkpoint reference is a file path or a run plus a selector."""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.infrastructure.checkpointing.ref import resolve


def _best(run: Path) -> Path:
    (run / "checkpoints").mkdir(parents=True)
    path = run / "checkpoints" / "best.pt"
    path.write_bytes(b"x")
    return path


def test_path_ref_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve(tmp_path / "nope.pt")


def test_path_ref_resolves_to_itself(tmp_path: Path) -> None:
    best = _best(tmp_path)
    assert resolve(str(best)) == best
    assert resolve(best) == best


def test_run_ref_defaults_to_best(tmp_path: Path) -> None:
    best = _best(tmp_path)
    assert resolve({"run_dir": str(tmp_path)}) == best


def test_run_ref_with_nothing_to_select_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no 'latest' checkpoint"):
        resolve({"run_dir": str(tmp_path), "select": "latest"})


@pytest.mark.parametrize(
    "ref, error",
    [
        ({"select": "best"}, ValueError),
        ({"run_dir": "x", "pick": "best"}, ValueError),
        (42, TypeError),
    ],
)
def test_malformed_refs_raise(ref, error) -> None:
    with pytest.raises(error):
        resolve(ref)


def test_a_run_ref_refuses_a_half_published_conversion(tmp_path: Path) -> None:
    from spectramr.infrastructure.checkpointing.select import (
        CONVERSION_JOURNAL,
        ConversionPendingError,
    )

    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "step_00000002.pt").write_bytes(b"x")
    (ck / CONVERSION_JOURNAL).write_text("{}")
    with pytest.raises(ConversionPendingError):
        resolve({"run_dir": tmp_path, "select": "latest"})
