"""The one ``dataset_kind`` -> :class:`ContrastGrid` dispatch for real-data references.

Shared by ``scripts/sim2rank`` and ``spectramr meta-evaluate`` so neither keeps its
own grouping branch: the M4Raw stem law must never reach fastMRI-brain files, where
it NEX-averages two patients into one pseudo-GT (#344).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from spectramr.data.datasets.contrast_grid import ContrastGrid, discover_m4raw_grid
from spectramr.data.datasets.fastmri_brain_groups import discover_brain_grid

__all__ = ["DATASET_KINDS", "NATIVE_ACQUISITION", "discover_native_grid"]


def _m4raw(data_dir: Path, contrasts, max_subjects, max_contrasts, allow_incomplete):
    return discover_m4raw_grid(
        data_dir,
        contrasts=contrasts,
        max_subjects=max_subjects,
        max_contrasts=max_contrasts,
        allow_incomplete=allow_incomplete,
    )


def _brain(data_dir: Path, contrasts, max_subjects, max_contrasts, allow_incomplete):
    if allow_incomplete:
        raise ValueError(
            "--allow-incomplete-subjects does not apply to fastmri_brain: its exams are "
            "unlinked, so no subject can lack a contrast."
        )
    return discover_brain_grid(
        data_dir, contrasts=contrasts, max_subjects=max_subjects, max_contrasts=max_contrasts
    )


_DISCOVERERS: dict[str, Callable[..., ContrastGrid]] = {
    "m4raw": _m4raw,
    "fastmri_brain": _brain,
}

DATASET_KINDS: frozenset[str] = frozenset(_DISCOVERERS)

#: How each kind's files were acquired (an ``AcquisitionMode`` value). Both are 2-D
#: multislice stacks, so a ``volume_3d`` encoding of them is a simulation.
NATIVE_ACQUISITION: dict[str, str] = {
    "m4raw": "multislice_2d",
    "fastmri_brain": "multislice_2d",
}


def discover_native_grid(
    dataset_kind: str,
    data_dir: Path,
    *,
    contrasts: list[str] | None = None,
    max_subjects: int = 5,
    max_contrasts: int = 3,
    allow_incomplete: bool = False,
) -> ContrastGrid:
    """Grid for ``dataset_kind``; an unregistered kind raises (no default grouper)."""
    try:
        discover = _DISCOVERERS[dataset_kind]
    except KeyError:
        raise ValueError(
            f"Unknown dataset_kind {dataset_kind!r}. Registered kinds: {sorted(DATASET_KINDS)}."
        ) from None
    return discover(Path(data_dir), contrasts, max_subjects, max_contrasts, allow_incomplete)
