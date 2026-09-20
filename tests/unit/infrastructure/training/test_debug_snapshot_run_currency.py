"""``RUN.json``: which snapshots belong to the run that just finished.

A run directory is reused across runs, and a snapshot directory is named
``<tag>_step_<n>`` with no run in it. A shorter run therefore rewrites the early
steps IN PLACE and leaves the later ones untouched, and the listing reads as one
coherent set. ``experiment_11_attention_none`` came back from the 2026-09-16
smoke run (capped at ``training.max_iterations=2``) holding ``first_steps``
steps 1-2 from that run beside steps 3-8 and ``model_output_dc`` steps 500-4000
from the full run of 2026-09-14 -- and the mtimes agree with neither, because
rewriting a file in place does not touch its directory's mtime.

That blend is what made the snapshots look inconsistent across arms when nothing
about the snapshot writer had changed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from spectramr.infrastructure.training.debug_snapshot import (
    _MARKED_ROOTS,
    _WRITTEN_DIRS,
    _call_counts,
    save_debug_snapshot,
)

_WRITER_LOGGER = "spectramr.infrastructure.training.debug_snapshot"


def _logging_config():
    from tests.utils.block_config_stub import LoggingConfigStub

    return LoggingConfigStub(snapshots={"save_images": False})


def _snapshot(run_dir: Path, step: int, tag: str = "first_steps") -> Path:
    return save_debug_snapshot(
        run_dir=run_dir,
        step=step,
        tag=tag,
        tensors={"input": torch.zeros(1, 2, 8, 8)},
        paradigm="TestStrategy",
        config_section=_logging_config(),
    )


def _fresh_process_state() -> None:
    """A new run is a new process: budgets and the once-per-root scan reset."""
    _call_counts.clear()
    _MARKED_ROOTS.clear()
    _WRITTEN_DIRS.clear()


def test_the_marker_names_only_the_directories_this_run_wrote(tmp_path: Path) -> None:
    _fresh_process_state()
    _snapshot(tmp_path, 1)
    _snapshot(tmp_path, 2)

    marker = json.loads((tmp_path / "debug_snapshots" / "RUN.json").read_text())
    assert marker["directories"] == ["first_steps_step_000001", "first_steps_step_000002"]
    assert marker["run_id"], "a snapshot with no run id is the ambiguity this removes"


def test_a_shorter_second_run_does_not_claim_the_first_runs_leftovers(
    tmp_path: Path, caplog
) -> None:
    """The planted violation: 8 steps, then a 2-step rerun over the same directory."""
    _fresh_process_state()
    for step in range(1, 9):
        _snapshot(tmp_path, step)
    first = json.loads((tmp_path / "debug_snapshots" / "RUN.json").read_text())

    # A second run: new process, and a different published identity.
    _fresh_process_state()
    from spectramr.infrastructure.training import snapshot_provenance

    snapshot_provenance.set_run_identity(
        {"run_id": "rerun-20260916_192903-009d696372e9", "run_name": "rerun"}
    )
    try:
        with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
            _snapshot(tmp_path, 1)
            _snapshot(tmp_path, 2)
    finally:
        snapshot_provenance.reset_run_identity()

    second = json.loads((tmp_path / "debug_snapshots" / "RUN.json").read_text())
    assert second["run_id"] != first["run_id"]
    assert second["directories"] == [
        "first_steps_step_000001",
        "first_steps_step_000002",
    ], "steps 3-8 are the previous run's and must not be claimed by this one"

    # Still on disk -- the marker states ownership, it does not delete evidence.
    roots = {d.name for d in (tmp_path / "debug_snapshots").iterdir() if d.is_dir()}
    assert "first_steps_step_000008" in roots

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("from an earlier run" in w for w in warnings), warnings
    assert sum("from an earlier run" in w for w in warnings) == 1, (
        "the scan has one answer to give; it must not warn once per snapshot"
    )


def test_a_first_run_into_an_empty_directory_says_nothing(tmp_path: Path, caplog) -> None:
    _fresh_process_state()
    with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
        _snapshot(tmp_path, 1)
    assert not [r for r in caplog.records if "earlier run" in r.getMessage()]
