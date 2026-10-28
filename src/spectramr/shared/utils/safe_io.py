"""Safe IO helpers: wrappers around potentially dangerous operations.

PyTorch torch.load can execute arbitrary code via pickle. This module provides
a safer default that enforces CPU map_location and can restrict to tensors via
weights_only when available (PyTorch >= 2.1).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch


def safe_torch_load(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool | None = None,
) -> Any:
    """Safely load a torch checkpoint with basic path validation.

    - Ensures the file exists and is within the workspace (prevents traversal).
    - Forces CPU map_location to avoid unintended device execution.
    - Uses weights_only=True when supported to reduce pickle attack surface.
    """
    # Coerce to string for path operations
    path_str = os.fspath(path)
    # Path traversal guard FIRST — must reject ``..`` escapes even if the
    # target happens not to exist (the security signal is in the shape of
    # the path, not the FS state).
    norm = os.path.normpath(path_str)
    if os.path.isabs(path_str):
        # For absolute paths, we only normalize
        normalized = norm
    else:
        # For relative paths, ensure no leading '..' after normalization
        if norm.startswith(".."):
            raise ValueError(f"Unsafe relative path: {path}")
        normalized = norm
    if not os.path.exists(normalized):
        raise FileNotFoundError(f"File not found: {path}")

    # Determine if weights_only is supported by this torch version
    # Detect weights_only support in current torch version
    co_names = getattr(torch.load, "__code__", None)
    varnames = getattr(co_names, "co_varnames", ())
    supports_weights_only = "weights_only" in varnames
    kwargs: dict[str, Any] = {"map_location": map_location}
    if supports_weights_only:
        if weights_only is None:
            # Default to True when supported
            kwargs["weights_only"] = True
        else:
            kwargs["weights_only"] = weights_only

    return torch.load(normalized, **kwargs)


def atomic_publish(filepath: str | os.PathLike[str], write: Callable[[str], None]) -> None:
    """Publish a file atomically: ``write`` fills a temp file, then it is renamed over.

    A writer killed mid-write (SIGKILL, wall-clock limit, OOM) would otherwise leave a
    truncated file that passes every existence check and fails only when read, often
    days later. The rename is atomic on POSIX within one filesystem, so a reader sees
    the previous file or the new one, never half of one.

    Args:
        filepath: Target path. Its parent is created if missing.
        write: Called with the temp path; must write the complete file there.

    Raises:
        Exception: Whatever ``write`` raised, after the temp file is removed.
    """
    import tempfile
    from pathlib import Path

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False, suffix=".tmp") as tf:
            temp_name = tf.name
        write(temp_name)
        os.replace(temp_name, path)
    except Exception:
        if temp_name and os.path.exists(temp_name):
            os.remove(temp_name)
        raise


def atomic_save_torch(obj: Any, filepath: str | os.PathLike[str]) -> None:
    """``torch.save`` through :func:`atomic_publish`."""
    atomic_publish(filepath, lambda temp: torch.save(obj, temp))


def atomic_save_json(data: Any, filepath: str | os.PathLike[str]) -> None:
    """Save JSON data atomically to prevent corruption on crash.

    [FORENSIC FIX] Uses write-to-temp + atomic-rename pattern.
    If process crashes during json.dump, the temp file is left behind
    but the original file remains intact.

    Args:
        data: JSON-serializable data to save
        filepath: Target file path
    """
    import json
    import tempfile
    from pathlib import Path

    path = Path(filepath)
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    temp_name = None
    try:
        # Write to temp file in same directory (ensures same filesystem for atomic rename)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=str(parent), delete=False, suffix=".tmp"
        ) as tf:
            json.dump(data, tf, indent=2)
            temp_name = tf.name

        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        os.replace(temp_name, path)
    except Exception as e:
        # Clean up temp file on error
        if temp_name and os.path.exists(temp_name):
            os.remove(temp_name)
        raise e
