"""Per-process cache of open read-only HDF5 handles.

The M4Raw loader read every slice of every repetition through a fresh
``h5py.File`` open; the opens alone were ~2.5% of a profiled self_supervised run
(sabine, 2026-10-21). A handle is reused only by the process that opened it (a
forked DataLoader worker never touches its parent's) and only while the file is
the same file: a rewrite changes its inode, mtime or size and forces a reopen.

The cap must hold a worker's whole working set (an n2n split reads ~3 repetition
files per group, hundreds in all) or a shuffled epoch evicts on nearly every read
(#3118). Handles are opened without a chunk cache: the slice-chunked tree is read
one chunk per call, so h5py's default 1 MiB cache per file bought no hits and cost
~0.5 MB of RSS per open handle; without it a handle is ~1 KB.
"""

from __future__ import annotations

import logging
import os
import resource
from collections import OrderedDict
from pathlib import Path

import h5py

logger = logging.getLogger(__name__)

#: Upper bound on the handles one process keeps.
OPEN_FILES_CEILING = 4096
#: Share of the soft RLIMIT_NOFILE one process may spend on cached handles: the main
#: process and every train and validation worker hold their own.
_FD_SHARE = 8


def _open_files_cap() -> int:
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return OPEN_FILES_CEILING
    return max(16, min(OPEN_FILES_CEILING, soft // _FD_SHARE))


#: Open files kept per process; the least recently used one is closed beyond it.
MAX_OPEN_FILES = _open_files_cap()

_handles: OrderedDict[tuple[str, int, int, int], h5py.File] = OrderedDict()
_owner_pid: int | None = None


def open_h5(path: Path | str) -> h5py.File:
    """A read-only (``swmr``) handle on ``path``, opened once per process and file version.

    The caller must not close it. Raises whatever ``os.stat`` / ``h5py.File`` raise.
    """
    global _owner_pid
    if _owner_pid != os.getpid():
        # Inherited across fork: the parent's handles are not ours to use or close.
        _handles.clear()
        _owner_pid = os.getpid()
        logger.debug("[h5] process %d caches up to %d open HDF5 handles", _owner_pid, MAX_OPEN_FILES)
    stat = os.stat(path)
    key = (str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size)
    handle = _handles.get(key)
    if handle is not None and handle.id.valid:
        _handles.move_to_end(key)
        return handle
    handle = h5py.File(str(path), "r", swmr=True, rdcc_nbytes=0)
    _handles[key] = handle
    while len(_handles) > MAX_OPEN_FILES:
        _, oldest = _handles.popitem(last=False)
        oldest.close()
    return handle


def close_all() -> None:
    """Close every handle this process holds."""
    while _handles:
        _, handle = _handles.popitem()
        if handle.id.valid:
            handle.close()


__all__ = ["MAX_OPEN_FILES", "OPEN_FILES_CEILING", "close_all", "open_h5"]
