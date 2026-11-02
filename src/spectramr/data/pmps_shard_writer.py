"""Shards and manifest of the PMPS Stage-C synthetic paired corpus: the files' one owner.

Stage C writes each rendered pair ``(ulf, hf, theta)`` to
``<output_root>/<accepted|rejected>/shard_<k>/pair_<index>.<ext>`` and one ``manifest.json`` at
``<output_root>`` listing every pair with its gate verdict and residuals. Stage D's
``synthetic_paired_manifest`` names that manifest. Arrays are ``(C, H, W)`` float32.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import nibabel as nib

    HAVE_NIBABEL = True
except ImportError:
    nib = None
    HAVE_NIBABEL = False

try:
    import h5py

    HAVE_H5PY = True
except ImportError:
    h5py = None
    HAVE_H5PY = False

__all__ = ["MANIFEST_NAME", "OUTPUT_FORMATS", "PAIR_KEYS", "write_manifest", "write_pair"]

OUTPUT_FORMATS: tuple[str, ...] = ("nifti", "h5", "npz")
PAIR_KEYS: tuple[str, ...] = ("ulf", "hf", "theta")
MANIFEST_NAME = "manifest.json"


def _require_backend(fmt: str) -> None:
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"output_format {fmt!r} is not one of {OUTPUT_FORMATS}")
    if fmt == "nifti" and not HAVE_NIBABEL:
        raise ImportError("output_format 'nifti' needs nibabel, which is not installed")
    if fmt == "h5" and not HAVE_H5PY:
        raise ImportError("output_format 'h5' needs h5py, which is not installed")


def write_pair(
    root: Path,
    *,
    shard: int,
    index: int,
    accepted: bool,
    arrays: dict[str, np.ndarray],
    fmt: str,
) -> list[str]:
    """Write one pair and return its file paths relative to ``root``."""
    _require_backend(fmt)
    missing = set(PAIR_KEYS) - set(arrays)
    if missing:
        raise ValueError(f"pair {index} lacks {sorted(missing)}")
    folder = root / ("accepted" if accepted else "rejected") / f"shard_{shard:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"pair_{index:06d}"
    data = {k: np.asarray(arrays[k], dtype=np.float32) for k in PAIR_KEYS}
    if fmt == "npz":
        paths = [folder / f"{stem}.npz"]
        np.savez_compressed(paths[0], **data)
    elif fmt == "h5":
        paths = [folder / f"{stem}.h5"]
        with h5py.File(paths[0], "w") as handle:
            for key, value in data.items():
                handle.create_dataset(key, data=value)
    else:
        paths = []
        for key, value in data.items():
            # NIfTI is (X, Y, C): spatial axes first, channels last.
            path = folder / f"{stem}_{key}.nii.gz"
            nib.save(nib.Nifti1Image(np.moveaxis(value, 0, -1), np.eye(4)), path)
            paths.append(path)
    return [str(p.relative_to(root)) for p in paths]


def write_manifest(root: Path, records: list[dict[str, Any]], provenance: dict[str, Any]) -> Path:
    """Write the corpus manifest: every pair's record and the run's resolved knobs."""
    root.mkdir(parents=True, exist_ok=True)
    accepted = sum(1 for r in records if r["accepted"])
    payload = {
        "n_pairs": len(records),
        "n_accepted": accepted,
        "provenance": provenance,
        "pairs": records,
    }
    path = root / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("PMPS manifest: %d pairs (%d accepted) at %s", len(records), accepted, path)
    return path
