"""Precomputed NEX-registration shifts for the M4Raw repetition route (#2977).

``nex_registration: translation`` needs one in-plane shift per repetition of a group.
Estimating it costs a read of the group's central slices from every repetition plus a
phase correlation, and the estimate lived in a per-process cache that every DataLoader
worker rebuilds each epoch. A store holds the shifts once: ``<root>/nex_shifts.json``,
keyed by the group's loaded repetition file names, written by
``scripts/precompute_m4raw_nex_shifts.py`` with the same :func:`group_shifts` the loader
falls back to. A header carries the window and upsampling the shifts were estimated
with; a store made with other settings is refused.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from spectramr.data.datasets.m4raw_registration import REGISTRATION_SLICES, SHIFT_UPSAMPLE

__all__ = [
    "VALID_NEX_SHIFTS_MISSING",
    "NexShiftMismatchError",
    "NexShiftMissingError",
    "NexShiftStore",
    "read_shift_groups",
    "shift_key",
    "write_shift_groups",
]

#: ``raise`` stops on a group absent from the store; ``estimate`` falls back to the loader's estimate.
VALID_NEX_SHIFTS_MISSING: frozenset[str] = frozenset({"raise", "estimate"})

_FILENAME = "nex_shifts.json"
_FORMAT = 1


class NexShiftMismatchError(ValueError):
    """The store was written with another format, window or upsampling than the loader uses."""


class NexShiftMissingError(KeyError):
    """A group the loader registers has no entry in the store."""


def shift_key(rep_paths: Sequence[str | Path]) -> str:
    """The store key of a group: the loaded repetition file names, in order."""
    return "|".join(Path(p).name for p in rep_paths)


def _header() -> dict[str, int]:
    return {
        "format": _FORMAT,
        "registration_slices": REGISTRATION_SLICES,
        "upsample": SHIFT_UPSAMPLE,
    }


def read_shift_groups(root: str | Path) -> dict[str, list[tuple[float, float]]]:
    """Every group's shifts under ``root``; ``{}`` when nothing has been written there."""
    path = Path(root) / _FILENAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    header = {k: payload.get(k) for k in _header()}
    if header != _header():
        raise NexShiftMismatchError(
            f"{path} was written with {header}, this loader estimates with {_header()}; "
            "write the store again with scripts/precompute_m4raw_nex_shifts.py."
        )
    return {k: [(float(dy), float(dx)) for dy, dx in v] for k, v in payload["groups"].items()}


def write_shift_groups(
    root: str | Path, groups: Mapping[str, Sequence[tuple[float, float]]]
) -> Path:
    """Write ``groups`` to ``root`` atomically, so a loader never reads a half-written file."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / _FILENAME
    payload = {**_header(), "groups": {k: [list(s) for s in v] for k, v in sorted(groups.items())}}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)
    return path


class NexShiftStore:
    """Read side of the store: one lookup per group, loaded once with the dataset."""

    def __init__(self, groups: Mapping[str, list[tuple[float, float]]], missing: str) -> None:
        if missing not in VALID_NEX_SHIFTS_MISSING:
            raise ValueError(
                f"[M4Raw] Unknown nex_shifts_missing: {missing!r}. "
                f"Valid: {sorted(VALID_NEX_SHIFTS_MISSING)}"
            )
        self._groups = dict(groups)
        self._missing = missing

    @classmethod
    def declared(cls, root: str | Path, missing: str) -> NexShiftStore:
        """The store under ``root``; an empty one is an error, not an empty cache."""
        groups = read_shift_groups(root)
        if not groups:
            raise FileNotFoundError(
                f"[M4Raw] data.nex_shift_root={str(root)!r} holds no {_FILENAME}; run "
                "scripts/precompute_m4raw_nex_shifts.py --out that directory."
            )
        return cls(groups, missing)

    def get(self, rep_paths: Sequence[str | Path]) -> list[tuple[float, float]] | None:
        """The group's shifts, or ``None`` under ``estimate`` when it has none; raises under ``raise``."""
        entry = self._groups.get(shift_key(rep_paths))
        if entry is None and self._missing == "raise":
            raise NexShiftMissingError(
                f"[M4Raw] no precomputed shifts for {shift_key(rep_paths)!r}; extend the store "
                "with scripts/precompute_m4raw_nex_shifts.py or set data.nex_shifts_missing: estimate."
            )
        if entry is not None and len(entry) != len(rep_paths):
            raise NexShiftMismatchError(
                f"[M4Raw] the store holds {len(entry)} shift(s) for a group of {len(rep_paths)} "
                f"repetition(s): {shift_key(rep_paths)!r}."
            )
        return entry
