"""``checkpoint.keep_last_n``: which step checkpoints survive a save.

The unit is a step: every format of ``step_N`` plus, under a strategy-native adapter,
the ``step_N`` tag directory under each native root. They are deleted together, files
first, so an interrupted prune leaves an orphan directory and never a discoverable file
whose shards are gone. Best files and the tag DeepSpeed's ``latest`` names are never
candidates.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

from spectramr.infrastructure.checkpointing.layout import CheckpointLayout, native_tag, parse

logger = logging.getLogger(__name__)

#: The file DeepSpeed writes beside its tag directories, naming the newest tag.
NATIVE_LATEST_FILE = "latest"
_NATIVE_TAG_RE = re.compile(r"step_(?P<step>\d{8,})")


def validate_keep_last_n(value: object) -> int:
    """Return ``value`` if it is an integer >= 1 (the schema's bound), else raise.

    Raises:
        ValueError: For anything else; ``0`` is not "keep all".
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"checkpoint.keep_last_n must be an integer >= 1, got {value!r}")
    return value


def native_latest_tags(native_roots: Sequence[Path]) -> set[str] | None:
    """The tags DeepSpeed's ``latest`` files name, or ``None`` if one exists but is unreadable."""
    tags: set[str] = set()
    for root in native_roots:
        latest = root / NATIVE_LATEST_FILE
        if not latest.exists():
            continue
        try:
            tags.add(latest.read_text().strip())
        except OSError as exc:
            logger.warning("[checkpoint] retention skipped: cannot read %s: %s", latest, exc)
            return None
    return tags


def _log_rmtree_error(_func: object, failed: str, exc: BaseException) -> None:
    logger.warning("[checkpoint] retention could not remove %s: %s", failed, exc)


def remove_artifact(path: Path) -> bool:
    """Delete one file or tag directory; a failure costs disk, not correctness, so it is logged."""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, onexc=_log_rmtree_error)
            return not path.exists()
        path.unlink()
    except OSError as exc:
        logger.warning("[checkpoint] retention could not remove %s: %s", path, exc)
        return False
    return True


def _units(
    layout: CheckpointLayout, native_roots: Sequence[Path], *, newest: int
) -> dict[int, list[Path]]:
    units: dict[int, list[Path]] = {}
    for entry in layout.entries():
        if entry.role == "step" and entry.step is not None and entry.step < newest:
            units.setdefault(entry.step, []).append(entry.path)
    for root in native_roots:
        for entry in root.iterdir():
            match = _NATIVE_TAG_RE.fullmatch(entry.name)
            if match is None or entry.is_symlink() or not entry.is_dir():
                continue
            if int(match.group("step")) < newest:
                units.setdefault(int(match.group("step")), []).append(entry)
    return units


def prune(
    layout: CheckpointLayout,
    just_written: Path,
    *,
    keep_last_n: int,
    native_roots: Sequence[Path] = (),
) -> list[Path]:
    """Keep the step of ``just_written`` and the next ``keep_last_n - 1`` lower steps.

    Steps above ``just_written`` are left alone: an earlier launch into the same run
    directory may have written them, and ranking against them would delete what this
    run just saved. Filesystem errors are logged, never raised, because the checkpoint
    itself has already landed.

    Returns:
        The paths deleted.
    """
    keep_last_n = validate_keep_last_n(keep_last_n)
    written = parse(just_written)
    if written is None or written.role != "step" or written.step is None:
        raise ValueError(f"{just_written.name!r} is not a step checkpoint")
    keep_tag = native_tag(written.step)
    invisible = [root / keep_tag for root in native_roots if not (root / keep_tag).is_dir()]
    if invisible:
        logger.warning(
            "[checkpoint] keep_last_n not applied: the tag directory just written is not "
            "visible from this rank (%s); nothing was deleted.",
            ", ".join(str(p) for p in invisible),
        )
        return []
    protected = native_latest_tags(native_roots)
    if protected is None:
        return []
    try:
        units = _units(layout, native_roots, newest=written.step)
    except OSError as exc:
        logger.warning("[checkpoint] retention could not list %s: %s", exc.filename, exc)
        return []
    ranked = sorted((s for s in units if native_tag(s) not in protected), reverse=True)
    removed: list[Path] = []
    for step in ranked[keep_last_n - 1 :]:
        for path in units[step]:
            if not remove_artifact(path):
                break  # keep the rest of this step, so a surviving file keeps its shards
            removed.append(path)
    if removed:
        logger.info(
            "[checkpoint] keep_last_n=%d removed %d artifact(s): %s",
            keep_last_n,
            len(removed),
            ", ".join(os.path.relpath(p, layout.directory) for p in removed),
        )
    return removed


__all__ = [
    "NATIVE_LATEST_FILE",
    "native_latest_tags",
    "prune",
    "remove_artifact",
    "validate_keep_last_n",
]
