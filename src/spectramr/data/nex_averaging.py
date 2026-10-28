"""Average each M4Raw repetition group into one volume and write it back as k-space.

M4Raw acquires every contrast two or three times (``<subject>_<contrast><NN>.h5``).
The repetitions of one contrast are independent noise draws of the same anatomy, and
the head moves between them by a pixel at most (the ``m4raw-reps`` floor of
``contrast_registration_check.py``), so the motion that matters for cross-contrast
work is between the averaged volumes of two contrasts, not between single excitations.

Each ``(subject, contrast)`` group is combined by the loader's own NEX target,
``_average_reps(reps, "phase_aligned_mean")``: every repetition's global phase per
``(slice, coil)`` is aligned to the first repetition, then all are averaged. A plain
complex mean would cancel signal under the receiver phase drift between repetitions.
The result is written as ``<subject>_<contrast>00.h5`` into a mirror of the split:
``00`` keeps the name parseable by ``parse_m4raw_file_id`` and cannot collide with an
acquisition, which M4Raw numbers from ``01``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py

from spectramr.data.contrast_registration import read_kspace, write_kspace_copy
from spectramr.data.datasets.m4raw_dataset import _average_reps
from spectramr.data.datasets.m4raw_identity import NEX_AVERAGE_REPETITION, parse_m4raw_file_id

AVERAGE_MODE = "phase_aligned_mean"


def average_group(paths: list[Path], dest: Path) -> dict[str, Any]:
    """Write the phase-aligned mean of ``paths`` (one group, in repetition order) to ``dest``."""
    reps = []
    for path in paths:
        with h5py.File(path, "r") as f:
            reps.append(read_kspace(f["kspace"]))
    shapes = {tuple(r.shape) for r in reps}
    if len(shapes) > 1:
        return {"status": f"refused: repetitions on different grids {sorted(shapes)}"}
    write_kspace_copy(
        paths[0],
        dest,
        _average_reps(reps, AVERAGE_MODE),
        {
            "nex_average_mode": AVERAGE_MODE,
            "nex_average_count": len(paths),
            "nex_average_sources": [p.name for p in paths],
            "nex_phase_anchor": paths[0].name,
        },
    )
    return {"status": "averaged"}


def group_split(src_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """``(subject, contrast)`` -> repetitions in order, reading names only; writes nothing.

    A file whose name does not parse, or that is itself an average, raises: averaging
    it would mix trees or drop a subject silently.
    """
    files = sorted(src_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files directly in {src_dir}")
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in files:
        ident = parse_m4raw_file_id(path.stem)
        if not ident.contrast:
            raise ValueError(f"{path.name} is not <subject>_<contrast><NN>.h5")
        if ident.repetition == NEX_AVERAGE_REPETITION:
            raise ValueError(f"{path.name} is already an average; pass the acquired split")
        groups[(ident.subject, ident.contrast)].append(path)
    return dict(groups)


def average_split(src_dir: Path, dst_dir: Path) -> list[dict[str, Any]]:
    """Mirror ``src_dir`` into ``dst_dir`` with one averaged file per ``(subject, contrast)``.

    Returns one record per group; ``group_split`` decides what may be averaged.
    """
    groups = group_split(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if any(dst_dir.iterdir()):
        raise FileExistsError(f"{dst_dir} is not empty; refusing to mix two averagings")

    records = []
    for (subject, contrast), paths in sorted(groups.items()):
        name = f"{subject}_{contrast}{NEX_AVERAGE_REPETITION}.h5"
        record = {
            "subject": subject,
            "contrast": contrast,
            "file": name,
            "sources": [p.name for p in paths],
        }
        record.update(average_group(paths, dst_dir / name))
        records.append(record)
    return records
