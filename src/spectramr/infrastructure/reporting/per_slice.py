r"""Per-slice validation rows at one checkpoint, keyed by slice identity.

``validation_metrics.csv`` carries one aggregate row per validation pass, so it
cannot pair two arms on the same data (#2478). Under
``validation.per_batch_rows: true`` each rank also writes one row per slice to
``validation_batches.csv`` (``validation_batches_rank<N>.csv`` off rank 0), with
the identifiers the loader put on the batch. This module reads those rows at
one pass and returns one value per slice, indexed by :data:`SLICE_KEY`.

Three rules make the rows comparable across arms:

* the pass is the run's early-stopping choice
  (``early_stopping_best_iteration`` in ``final_metrics.json``), or the last
  pass when the run recorded none, and the choice is logged;
* a slice scored twice in that pass (a pass replayed after a resume, or a
  sample a distributed sampler padded onto a second rank) keeps its last row;
* a row stands for exactly one slice: a batch of several samples is a mean over
  a set the loader drew, which differs between arms, so it is refused.

``batch_index`` is never part of the key: the loaders shuffle, so batch ``i``
of one arm is a different slice from batch ``i`` of another.

The NEX-equivalent is averaged as ``1/N_eq``, which is the unbiased quantity
(PAPER § 5.3); :func:`summarise` reports ``1 / mean(1/N_eq)``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.core.metrics.reciprocal import RECIPROCAL, is_reciprocal

from .per_case import nearest_step
from .views import SELECTED_KEY

logger = logging.getLogger(__name__)

#: The identity two arms are paired on. ``file_id`` is left out: it names the
#: file, which a slice-level loader may share across slices.
SLICE_KEY: tuple[str, ...] = ("subject_id", "contrast", "slice_index", "repetition_group")
ROWS_FILE = "validation_batches.csv"
#: Rank files written beside ``ROWS_FILE``; the writer's rule is in ``_run_validation``.
_RANK_FILE = re.compile(r"^validation_batches_rank(\d+)\.csv$")
_NOT_METRICS = frozenset({"iteration", "epoch", "batch_index", "n_samples", "file_id", *SLICE_KEY})


def rows_files(directory: str | Path) -> list[Path]:
    """The rank-0 rows file then every rank file in rank order; empty if none."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    ranks = sorted(
        (int(m.group(1)), p)
        for p in directory.iterdir()
        if (m := _RANK_FILE.match(p.name)) is not None
    )
    head = [directory / ROWS_FILE] if (directory / ROWS_FILE).is_file() else []
    return head + [p for _, p in ranks]


def find_rows_files(output_dir: str | Path) -> list[Path]:
    """Rows files of one run, under ``logs/`` or at the run root."""
    output_dir = Path(output_dir)
    for directory in (output_dir / "logs", output_dir):
        found = rows_files(directory)
        if found:
            return found
    return []


def run_root(rows_file: str | Path) -> Path:
    """The run directory a rows file belongs to (``final_metrics.json`` lives there)."""
    parent = Path(rows_file).parent
    return parent.parent if parent.name == "logs" else parent


def early_stopping_iteration(run_dir: str | Path) -> int | None:
    """The run's own early-stopping pass, or ``None`` when it recorded none."""
    path = Path(run_dir) / "final_metrics.json"
    try:
        value = json.loads(path.read_text()).get(SELECTED_KEY)
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


@dataclass(frozen=True)
class SlicePass:
    """One validation pass of one run, one row per slice."""

    frame: pd.DataFrame
    iteration: int
    selection: str
    replayed: int
    files: tuple[str, ...]

    @property
    def metric_columns(self) -> list[str]:
        return [str(c) for c in self.frame.columns]


def _read(files: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise ValueError(f"per-slice rows are empty: {[str(p) for p in files]}")
    frame = pd.concat(frames, ignore_index=True)
    missing = [c for c in ("iteration", "n_samples", *SLICE_KEY) if c not in frame.columns]
    if missing:
        raise ValueError(f"per-slice rows {files[0]} lack the columns {missing}")
    return frame


def _refuse_unpairable(frame: pd.DataFrame, source: Path) -> None:
    n_samples = pd.to_numeric(frame["n_samples"], errors="coerce")
    if (n_samples != 1).any():
        raise ValueError(
            f"{source}: {int((n_samples != 1).sum())} rows score a batch of other than one "
            "sample; a batch mean cannot be paired across arms whose loaders shuffle. "
            "Re-run validation with validation.loader.batch_size: 1 (#2478)."
        )
    for key in ("subject_id", "slice_index"):
        empty = frame[key].str.strip() == ""
        if empty.any():
            raise ValueError(
                f"{source}: {int(empty.sum())} rows carry no {key}; the loader did not "
                "put it on the batch, so the rows cannot be paired by slice (#2478)."
            )


def _canonical_index(frame: pd.DataFrame) -> pd.MultiIndex:
    """String identity columns, with ``3.0`` and ``3`` read as the same slice."""
    slice_index = pd.to_numeric(frame["slice_index"], errors="coerce")
    as_int = slice_index.dropna().astype(np.int64).astype(str)
    cells = frame[list(SLICE_KEY)].apply(lambda c: c.str.strip())
    cells.loc[as_int.index, "slice_index"] = as_int
    return pd.MultiIndex.from_frame(cells)


def read_pass(files: list[Path], best_iteration: int | None = None) -> SlicePass:
    """Rows of the pass at ``best_iteration`` (at or before it), else the last pass."""
    if not files:
        raise ValueError("no per-slice rows files given")
    frame = _read(files)
    _refuse_unpairable(frame, files[0])
    frame["step"] = pd.to_numeric(frame["iteration"], errors="coerce")
    iteration = nearest_step(frame, best_iteration)
    if best_iteration is None:
        selection = "last pass (no early-stopping record)"
        logger.warning("%s: no %s recorded; reading the last pass", files[0], SELECTED_KEY)
    elif iteration is None:
        raise ValueError(
            f"{files[0]}: no validation pass at or before the early-stopping "
            f"iteration {best_iteration}"
        )
    else:
        selection = f"{SELECTED_KEY}={best_iteration}"
    if iteration is None:
        raise ValueError(f"{files[0]}: no row carries a numeric iteration")
    one = frame[frame["step"] == iteration].copy()
    one.index = _canonical_index(one)
    kept = one[~one.index.duplicated(keep="last")]
    metrics = [c for c in kept.columns if c not in _NOT_METRICS and c != "step"]
    return SlicePass(
        frame=kept[metrics],
        iteration=int(iteration),
        selection=selection,
        replayed=int(len(one) - len(kept)),
        files=tuple(str(p) for p in files),
    )


def load_run(output_dir: str | Path) -> SlicePass | None:
    """The run's rows at its early-stopping pass; ``None`` when it wrote none."""
    files = find_rows_files(output_dir)
    if not files:
        return None
    return read_pass(files, early_stopping_iteration(run_root(files[0])))


def slice_values(rows: SlicePass, column: str) -> tuple[pd.Series, int]:
    """One finite value per slice on the analysis scale, and how many were dropped.

    A reciprocal metric is inverted before the finiteness test: ``N_eq = inf``
    is ``1/N_eq = 0``, a legitimate value, while ``N_eq = 0`` has no reciprocal.
    """
    if column not in rows.frame.columns:
        raise KeyError(f"{column!r} is not a column of the per-slice rows {rows.files[0]}")
    values = pd.to_numeric(rows.frame[column], errors="coerce").astype(np.float64)
    if is_reciprocal(column):
        with np.errstate(divide="ignore"):
            values = 1.0 / values
    finite = np.isfinite(values.to_numpy())
    return values[finite], int((~finite).sum())


def subject_means(values: pd.Series) -> pd.Series:
    """Slice values averaged within each subject."""
    return values.groupby(level="subject_id").mean()


def summarise(subject_values: pd.Series, column: str) -> dict[str, float | int | str]:
    """Mean, spread and ``n`` (subjects), back on the metric's own scale."""
    v = subject_values.to_numpy(dtype=np.float64)
    if not is_reciprocal(column):
        return {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "median": float(np.median(v)),
            "n": int(v.size),
            "unit": "subjects",
        }
    # A subject whose mean 1/N_eq is 0 has N_eq = inf, and the spread is then NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        per_subject = 1.0 / v
        return {
            "mean": float(1.0 / np.mean(v)),
            "std": float(np.std(per_subject)),
            "median": float(1.0 / np.median(v)),
            "n": int(v.size),
            "unit": "subjects (mean of 1/N_eq over slices, then subjects)",
        }


__all__ = [
    "RECIPROCAL",
    "ROWS_FILE",
    "SLICE_KEY",
    "SlicePass",
    "early_stopping_iteration",
    "find_rows_files",
    "is_reciprocal",
    "load_run",
    "read_pass",
    "rows_files",
    "run_root",
    "slice_values",
    "subject_means",
    "summarise",
]
