"""A checkpoint reference: a file path, or a run directory plus a selector."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from spectramr.infrastructure.checkpointing.layout import CheckpointLayout
from spectramr.infrastructure.checkpointing.select import select

CheckpointRef = str | Path | Mapping[str, str]

_RUN_REF_KEYS = frozenset({"run_dir", "select"})


def resolve(ref: CheckpointRef) -> Path:
    """Turn ``ref`` into an existing checkpoint file.

    Raises:
        FileNotFoundError: The path is missing, or the run holds no matching checkpoint.
        ValueError: A run reference without ``run_dir`` or with unknown keys.
        TypeError: Anything that is neither a path nor a mapping.
    """
    if isinstance(ref, Mapping):
        unknown = set(ref) - _RUN_REF_KEYS
        if unknown or "run_dir" not in ref:
            raise ValueError(
                f"a run checkpoint reference takes {sorted(_RUN_REF_KEYS)} with run_dir "
                f"required; got {sorted(ref)}"
            )
        which = ref.get("select", "best")
        path = select(ref["run_dir"], which)
        if path is None:
            where = CheckpointLayout.for_run(ref["run_dir"]).directory
            raise FileNotFoundError(f"no {which!r} checkpoint under {where}")
        return path
    if isinstance(ref, (str, Path)):
        path = Path(ref)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint {path} does not exist")
        return path
    raise TypeError(f"a checkpoint reference is a path or a mapping, not {type(ref).__name__}")


__all__ = ["CheckpointRef", "resolve"]
