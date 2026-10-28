"""Node-local staging of the input files a run reads (``data.staging``).

The copy happens once, in the main process, before any DataLoader worker exists, so a
worker only ever opens a finished file. Every stager on a node serialises on one
``flock``; a waiter re-checks the files after it acquires the lock and finds them done.
A file is copied to ``<dest>.partial.<pid>`` and renamed into place, so a name that
exists is a complete file, and a job killed mid-copy leaves only a partial that the next
stager (the lock holder, hence the only live one) removes.
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "StageReport",
    "StagingNoSpaceError",
    "stage_files",
    "stage_index_records",
    "stage_path_lists",
    "stage_subtree",
]

_STAGE_DIR = "spectramr_stage"
_GB = 1e9


class StagingNoSpaceError(RuntimeError):
    """The files do not fit on the node-local root with ``min_free_gb`` to spare."""


@dataclass(frozen=True)
class StageReport:
    """What a staging call did; stamped into the run log."""

    root: str
    files: int
    copied: int
    reused: int
    bytes_copied: int
    seconds: float
    in_place: bool = False

    def summary(self) -> str:
        if self.in_place:
            return f"{self.files} file(s) read in place (staging skipped: no space)"
        return (
            f"{self.files} file(s) under {self.root}: copied {self.copied} "
            f"({self.bytes_copied / _GB:.2f} GB), reused {self.reused}, {self.seconds:.1f}s"
        )


def _dest(root: Path, source: Path) -> Path:
    """The copy of ``source``: its absolute path mirrored under the stage directory."""
    return root / _STAGE_DIR / str(source.resolve()).lstrip(os.sep)


def _current(source_stat: os.stat_result, dest: Path) -> bool:
    try:
        d = dest.stat()
    except FileNotFoundError:
        return False
    return d.st_size == source_stat.st_size and d.st_mtime_ns == source_stat.st_mtime_ns


def _copy_one(source: Path, dest: Path, mtime_ns: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f"{dest.name}.partial.{os.getpid()}")
    try:
        shutil.copyfile(source, partial)
        os.utime(partial, ns=(mtime_ns, mtime_ns))
        os.replace(partial, dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return dest.stat().st_size


def _remove_stale_partials(dest: Path) -> None:
    """Partials left by a killed stager; the caller holds the lock, so none is live."""
    for stale in dest.parent.glob(f"{dest.name}.partial.*") if dest.parent.is_dir() else ():
        stale.unlink(missing_ok=True)


def stage_files(sources: Iterable[Path | str], cfg: Any) -> tuple[dict[Path, Path], StageReport]:
    """Copy ``sources`` under ``cfg.root`` and return ``{source: staged}`` plus a report.

    ``cfg`` is a :class:`DataStagingConfigSchema`. A source that is missing raises
    ``FileNotFoundError``; a set that does not fit raises :class:`StagingNoSpaceError`
    unless ``cfg.when_full`` is ``read_in_place``, which warns and maps each file to itself.
    """
    started = time.perf_counter()
    unique = sorted({Path(s) for s in sources})
    missing = [s for s in unique if not s.is_file()]
    if missing:
        raise FileNotFoundError(
            f"data.staging: {len(missing)} input file(s) do not exist, e.g. {missing[0]}"
        )
    root = Path(cfg.root)
    stage_dir = root / _STAGE_DIR
    stage_dir.mkdir(parents=True, exist_ok=True)
    with open(stage_dir / ".lock", "a+") as lock:
        # A waiter blocks here while another stager copies, then finds the files current.
        fcntl.flock(lock, fcntl.LOCK_EX)
        stats = {s: s.stat() for s in unique}
        todo = [s for s in unique if not _current(stats[s], _dest(root, s))]
        needed = sum(stats[s].st_size for s in todo)
        free = shutil.disk_usage(stage_dir).free
        if free - needed < cfg.min_free_gb * _GB:
            message = (
                f"data.staging: {len(todo)} file(s) need {needed / _GB:.1f} GB but {root} has "
                f"{free / _GB:.1f} GB free with min_free_gb={cfg.min_free_gb}."
            )
            if cfg.when_full == "raise":
                raise StagingNoSpaceError(message)
            logger.warning(
                "%s Reading the shared filesystem in place (when_full=read_in_place).", message
            )
            report = StageReport(str(root), len(unique), 0, 0, 0, 0.0, in_place=True)
            return {s: s for s in unique}, report
        for s in todo:
            _remove_stale_partials(_dest(root, s))
        with ThreadPoolExecutor(max_workers=cfg.threads) as pool:
            sizes = list(
                pool.map(lambda s: _copy_one(s, _dest(root, s), stats[s].st_mtime_ns), todo)
            )
    report = StageReport(
        str(root),
        len(unique),
        len(todo),
        len(unique) - len(todo),
        sum(sizes),
        time.perf_counter() - started,
    )
    logger.info("[STAGING] %s", report.summary())
    return {s: _dest(root, s) for s in unique}, report


def stage_path_lists(
    cfg: Any, *lists: Sequence[Path]
) -> tuple[list[list[Path]], StageReport | None]:
    """``lists`` with every path replaced by its staged copy; unchanged when ``cfg.mode`` is off."""
    if cfg.mode == "off":
        return [list(entries) for entries in lists], None
    mapping, report = stage_files([p for entries in lists for p in entries], cfg)
    return [[mapping[Path(p)] for p in entries] for entries in lists], report


def stage_subtree(cfg: Any, root: str | Path, names: Iterable[str]) -> str:
    """The directory to read ``names`` from: ``root``, or its node-local copy of just those files.

    A name absent from ``root`` stays absent in the copy, so the caller's missing-file
    policy behaves as it did in place.
    """
    root = Path(root)
    if cfg.mode == "off":
        return str(root)
    files = [root / n for n in sorted(set(names)) if (root / n).is_file()]
    if not files:
        return str(root)
    mapping, _ = stage_files(files, cfg)
    return str(mapping[files[0]].parent)


_H5_SUFFIXES = frozenset({".h5", ".hdf5"})


def stage_index_records(
    cfg: Any, *indices: Sequence[dict[str, Any]], key: str = "primary_path"
) -> list[list[dict[str, Any]]]:
    """Index records with ``record[key]`` pointing at the staged copy; HDF5 files only.

    Records are copied, never edited in place. A record whose file is not HDF5 keeps its
    path: those are read whole and sequentially, which a copy does not speed up.
    """
    if cfg.mode == "off":
        return [list(records) for records in indices]
    wanted = [
        Path(r[key]) for records in indices for r in records if Path(r[key]).suffix in _H5_SUFFIXES
    ]
    mapping, _ = stage_files(wanted, cfg) if wanted else ({}, None)
    return [
        [{**r, key: str(mapping.get(Path(r[key]), r[key]))} for r in records] for records in indices
    ]
