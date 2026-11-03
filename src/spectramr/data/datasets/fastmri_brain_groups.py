"""fastMRI-brain contrast grid discovery (data SSOT).

fastMRI brain stores **one fully-sampled acquisition per file**, tagged with a
contrast (``AXT1`` / ``AXT2`` / ``AXT2FLAIR`` / ``AXFLAIR`` / ``AXT1POST`` /
``AXT1PRE``), and does not link contrasts across exams. The grid is therefore
``linked=False``: each content's ``subject_id`` is its exam id, and a grid row is a
positional ``index`` across contrasts, never a patient. Stem-grouping these names
(the M4Raw law) would NEX-average two patients into one pseudo-GT (#344).

Contrast parsing is owned by :mod:`spectramr.data.datasets.contrast_grid`; this
module only reads the header attr and builds the balanced grid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py

from spectramr.data.datasets.contrast_grid import (
    ContentKey,
    ContrastGrid,
    choose_contrasts,
    h5_files,
    resolve_brain_contrast,
)

logger = logging.getLogger(__name__)

__all__ = ["brain_contrast_of", "discover_brain_grid"]


def brain_contrast_of(path: Path) -> str:
    """:func:`resolve_brain_contrast` with the ``acquisition`` attr read from the file.

    Raises when the attr and the ``AX*`` filename token disagree, and when neither
    yields a contrast.
    """
    acquisition = None
    try:
        with h5py.File(str(path), "r") as f:
            acquisition = f.attrs.get("acquisition")
    except OSError as exc:
        logger.debug("No readable acquisition attr in %s (%s); using filename.", path, exc)
    return resolve_brain_contrast(path, acquisition)


def _exam_id(path: Path, contrast: str) -> str:
    tokens = path.name.split(".")[0].split("_")
    upper = [t.upper() for t in tokens]
    tail = tokens[upper.index(contrast) + 1 :] if contrast in upper else []
    return "_".join(tail) or path.name.split(".")[0]


def discover_brain_grid(
    data_dir: Path,
    contrasts: list[str] | None = None,
    max_subjects: int = 5,
    max_contrasts: int = 3,
) -> ContrastGrid:
    """Unlinked grid: one file per content, ``subject_id`` = exam id.

    Buckets are balanced to the smallest chosen contrast (capped by
    ``max_subjects``) so rows stay rectangular; the drop count is logged.
    """
    buckets: dict[str, list[Path]] = {}
    for f in h5_files(data_dir):
        buckets.setdefault(brain_contrast_of(f), []).append(f)
    present = {c: len(v) for c, v in buckets.items()}
    chosen = choose_contrasts(present, contrasts, max_contrasts, data_dir)
    per_contrast = min(min(len(buckets[c]) for c in chosen), max_subjects)
    total = sum(len(buckets[c]) for c in chosen)
    if per_contrast * len(chosen) < total:
        logger.info(
            "fastMRI-brain grid balanced to %d per contrast x %d; dropped %d volumes.",
            per_contrast,
            len(chosen),
            total - per_contrast * len(chosen),
        )
    keys = tuple(
        ContentKey(_exam_id(buckets[c][i], c), c, None, (str(buckets[c][i]),))
        for i in range(per_contrast)
        for c in chosen
    )
    subjects = tuple(dict.fromkeys(k.subject_id for k in keys))
    logger.info(
        "fastMRI-brain grid: %d index rows x %d contrasts %s", per_contrast, len(chosen), chosen
    )
    return ContrastGrid(keys, subjects, tuple(chosen), False)
