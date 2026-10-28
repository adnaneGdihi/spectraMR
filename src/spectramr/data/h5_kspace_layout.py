"""The HDF5 layout for k-space that training reads one slice at a time.

M4Raw ships ``kspace`` as ``(slices, coils, H, W)`` complex64 with chunks of
``(slices, coils, H, 1)`` under gzip level 4. One chunk is one k-space column across
every slice, so reading a single slice decompresses all of the file: measured on
carya, 124 ms for one slice against 602 ms for the whole volume, and the same from
node-local disk, so the cost is decoding and not storage.

One chunk per slice makes a slice read touch only its own chunk (0.3 ms uncompressed,
0.4 ms under gzip level 1 with byte shuffle), and the whole-volume read is still
2.7x faster than the released layout. Gzip level 1 with shuffle keeps files at about
three times the released size, against 4.6x for no compression.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SLICE_FILTERS: dict[str, Any] = {"compression": "gzip", "compression_opts": 1, "shuffle": True}


def slice_chunked_kwargs(shape: Sequence[int]) -> dict[str, Any]:
    """``create_dataset`` keywords that store ``shape`` as one chunk per leading-axis slice."""
    if len(shape) < 3:
        raise ValueError(f"k-space of shape {tuple(shape)} has no slice axis to chunk along")
    return {"chunks": (1, *shape[1:]), **SLICE_FILTERS}


def rechunk_file(source: Path, dest: Path) -> None:
    """Copy ``source`` to ``dest`` with ``kspace`` stored one chunk per slice.

    Everything else is copied untouched, ``reconstruction_*`` images included: unlike a
    derived tree, the values here are identical to the source's.
    """
    with h5py.File(source, "r") as src, h5py.File(dest, "w") as dst:
        if "kspace" not in src:
            raise KeyError(f"No 'kspace' dataset in {source}. Available keys: {list(src.keys())}")
        for key in src:
            if key != "kspace":
                src.copy(key, dst)
        original = src["kspace"]
        out = dst.create_dataset(
            "kspace", data=original[()], **slice_chunked_kwargs(original.shape)
        )
        out.attrs.update(original.attrs)
        dst.attrs.update(src.attrs)


def _mismatch(source: Path, dest: Path) -> str | None:
    """The first way ``dest`` differs from ``source`` in content, or ``None`` if it does not."""
    with h5py.File(source, "r") as src, h5py.File(dest, "r") as dst:
        if set(src) != set(dst):
            return f"datasets {sorted(set(src) ^ set(dst))} differ"
        if set(src.attrs) != set(dst.attrs):
            return f"attributes {sorted(set(src.attrs) ^ set(dst.attrs))} differ"
        for key in sorted(src):
            if not np.array_equal(src[key][()], dst[key][()]):
                return f"{key} values differ"
    return None


def rechunk_split(src_dir: Path, dst_dir: Path) -> list[dict[str, Any]]:
    """Mirror ``src_dir`` into ``dst_dir``, rewriting each ``.h5`` and symlinking the rest.

    Every rewritten file is read back and compared with its source; a difference is
    recorded as ``mismatch: <what>`` rather than raised, so one bad file does not hide
    the others.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in sorted(src_dir.iterdir()):
        dest = dst_dir / path.name
        if path.suffix != ".h5":
            dest.symlink_to(path.resolve())
            continue
        rechunk_file(path, dest)
        problem = _mismatch(path, dest)
        records.append(
            {
                "file": path.name,
                "status": "rechunked" if problem is None else f"mismatch: {problem}",
                "source_mb": round(path.stat().st_size / 1e6, 2),
                "stored_mb": round(dest.stat().st_size / 1e6, 2),
            }
        )
    return records
