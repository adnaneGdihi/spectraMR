"""`MetricsTracker.log_cascading_validation` -- the tall CSV writer (issue #697).

The sweep used to be 45 column NAMES built from a hardcoded 15-metric list
crossed with the severity levels. They were assembled into a local and never
extended onto the header, so the row writer's ``extrasaction="ignore"``
discarded every value: the R-sweep was absent from every
`training_metrics.csv` ever written, while the per-level evaluation ran at full
GPU cost.

It is now tall -- one row per (iteration, severity point), with
`acceleration_level` and `timestep` as values -- in its own file, because
`training_metrics.csv` is one row per ITERATION and making that tall would
repeat every training loss once per level.

Routing through `_append_to_csv` is the point of the change ("the logging part
uses the ssot"): that method owns header caching, schema evolution and
backup-on-rewrite. These tests exercise it end to end on a real file rather
than asserting the call, so a regression in the SSOT surfaces here too.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from spectramr.core.cascading_validation import CASCADING_LEVELS, build_cascade_row
from spectramr.infrastructure.services.metrics_tracker import MetricsTracker


@pytest.fixture
def tracker(tmp_path: Path) -> MetricsTracker:
    return MetricsTracker(output_dir=str(tmp_path), model_type="diffusion")


def _sweep(**metrics: float) -> list[dict]:
    return [
        build_cascade_row(
            acceleration_level=lvl,
            heldout=False,
            timestep=lvl * 10,
            metrics=metrics or {"val_psnr": 30.0},
        )
        for lvl in CASCADING_LEVELS
    ]


def _read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_a_sweep_becomes_one_row_per_level(tracker: MetricsTracker) -> None:
    written = tracker.log_cascading_validation(_sweep(), iteration=100, epoch=1)
    assert written == len(CASCADING_LEVELS)

    rows = _read(tracker.cascading_csv)
    assert len(rows) == len(CASCADING_LEVELS)
    assert [float(r["acceleration_level"]) for r in rows] == [
        float(x) for x in CASCADING_LEVELS
    ]


def test_the_level_and_timestep_are_columns(tracker: MetricsTracker) -> None:
    """The whole point of (B): both are groupable values, not name fragments."""
    tracker.log_cascading_validation(_sweep(), iteration=100, epoch=1)
    header = _read(tracker.cascading_csv)[0].keys()
    for col in ("iteration", "epoch", "acceleration_level", "heldout", "timestep"):
        assert col in header
    assert not [h for h in header if h.endswith("x") and "_" in h and h[-2].isdigit()]


def test_the_writer_stamps_iteration_and_epoch(tracker: MetricsTracker) -> None:
    """The strategy that measures a sweep does not know its global step."""
    tracker.log_cascading_validation(_sweep(), iteration=250, epoch=3)
    rows = _read(tracker.cascading_csv)
    assert {r["iteration"] for r in rows} == {"250"}
    assert {r["epoch"] for r in rows} == {"3"}


def test_successive_sweeps_append_rather_than_replace(tracker: MetricsTracker) -> None:
    tracker.log_cascading_validation(_sweep(), iteration=100, epoch=1)
    tracker.log_cascading_validation(_sweep(), iteration=200, epoch=2)
    rows = _read(tracker.cascading_csv)
    assert len(rows) == 2 * len(CASCADING_LEVELS)
    assert {r["iteration"] for r in rows} == {"100", "200"}


def test_one_sweep_writes_the_header_once(tracker: MetricsTracker) -> None:
    """`_append_to_csv` REWRITES the file (and backs it up) whenever a row
    introduces a new key. Rows of one sweep carry identical keys, so that path
    must fire once for the file -- not once per row."""
    tracker.log_cascading_validation(_sweep(), iteration=100, epoch=1)
    text = tracker.cascading_csv.read_text()
    assert text.count("acceleration_level") == 1
    backups = list(tracker.cascading_csv.parent.glob("*backup*")) + list(
        tracker.cascading_csv.parent.glob("*.bak*")
    )
    assert not backups, f"header rewritten mid-sweep: {backups}"


def test_a_later_sweep_with_new_metrics_evolves_the_schema(
    tracker: MetricsTracker,
) -> None:
    """The case advisor flagged: a held-out point computing something the
    in-distribution points did not. Schema evolution must absorb it, and the
    earlier rows must stay readable rather than becoming ragged."""
    tracker.log_cascading_validation(_sweep(val_psnr=30.0), iteration=100, epoch=1)
    tracker.log_cascading_validation(
        _sweep(val_psnr=31.0, val_hfen=0.4), iteration=200, epoch=2
    )
    rows = _read(tracker.cascading_csv)
    assert len(rows) == 2 * len(CASCADING_LEVELS)
    assert "val_hfen" in rows[0]
    assert rows[0]["val_hfen"] == ""  # absent for the first sweep, not ragged
    assert float(rows[-1]["val_hfen"]) == pytest.approx(0.4)


def test_an_empty_sweep_writes_nothing_and_says_so(tracker: MetricsTracker) -> None:
    """Distinguishing "wrote nothing" from "was never called" is the thing the
    retired implementation could not do."""
    assert tracker.log_cascading_validation([], iteration=100, epoch=1) == 0
    assert not tracker.cascading_csv.exists()


def test_the_cascade_file_is_separate_from_training_metrics(
    tracker: MetricsTracker,
) -> None:
    """Making `training_metrics.csv` itself tall would repeat every training
    loss once per level in a file the report aggregator reads per pass."""
    assert tracker.cascading_csv != tracker.training_csv
    tracker.log_cascading_validation(_sweep(), iteration=100, epoch=1)
    assert tracker.cascading_csv.exists()
    assert not tracker.training_csv.exists()


def test_only_the_image_dirs_that_get_written_are_created(tmp_path: Path):
    """An empty artifact directory is a false report, not a harmless one.

    A third ``images/`` dir was created beside these two and never written by any
    code in src/, scripts/ or tests/. Every run shipped it empty, which reads to
    anyone inspecting a run directory as "image saving broke" -- the same class
    of misleading artifact as a stale config. Asserting the exact set, not just
    that the two live ones exist, is what makes this a gate: a re-added
    write-dead directory turns it red.
    """
    MetricsTracker(output_dir=str(tmp_path), model_type="diffusion", save_images=True)

    created = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
    assert created == ["fake_images", "real_images"]
