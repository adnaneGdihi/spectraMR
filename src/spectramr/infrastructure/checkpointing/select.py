"""The one rule for which checkpoint of a run is 'latest', 'best' or a given step.

Every format counts, so a run whose ``checkpoint.format`` changed between resumes still
resumes from its newest step. Where two files tie (two formats of one name), the newer
write wins.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from spectramr.infrastructure.checkpointing.layout import CheckpointLayout, CheckpointName

_STEP_SELECTOR = re.compile(r"step:(\d+)")
#: Written by the converter before it publishes, removed as its very last action.
CONVERSION_JOURNAL = ".convert_v1.journal.json"


class ConversionPendingError(RuntimeError):
    """The run's pre-1.0 conversion was interrupted while publishing."""


def conversion_pending(run_dir: str | Path) -> Path | None:
    """The journal of an interrupted pre-1.0 conversion, or ``None``."""
    journal = CheckpointLayout.for_run(run_dir).directory / CONVERSION_JOURNAL
    return journal if journal.is_file() else None


def refuse_pending_conversion(run_dir: str | Path) -> None:
    """Raise while a conversion is half-published.

    ``latest`` may then name an older step than the run reached, and ``best`` may still
    be the legacy alias, so no reader may choose a file until the converter has finished.

    Raises:
        ConversionPendingError: The converter's journal is present.
    """
    if (journal := conversion_pending(run_dir)) is not None:
        from spectramr.infrastructure.checkpointing.payload import CONVERTER

        raise ConversionPendingError(
            f"{journal.parent} holds a partial checkpoint conversion ({journal.name}); its "
            "latest step may be older than the run's. Finish the conversion first: "
            f"python {CONVERTER} {run_dir}"
        )


def _newest(candidates: Iterable[CheckpointName]) -> Path | None:
    found = list(candidates)
    if not found:
        return None
    return max(found, key=lambda name: name.path.stat().st_mtime_ns).path


def select(run_dir: str | Path, which: str) -> Path | None:
    """Return the run's checkpoint for ``which``, or ``None`` if there is none.

    Args:
        run_dir: The run's output directory (not its ``checkpoints/`` subdirectory).
        which: ``"latest"`` (highest step, else best), ``"best"`` or ``"step:<N>"``.

    Raises:
        ValueError: For any other selector, before the directory is read.
        ConversionPendingError: A pre-1.0 conversion of the run is half-published.
    """
    step_match = _STEP_SELECTOR.fullmatch(which)
    if which not in ("latest", "best") and step_match is None:
        raise ValueError(
            f"unknown checkpoint selector {which!r}; expected 'latest', 'best' or 'step:<N>'"
        )
    refuse_pending_conversion(run_dir)
    entries = CheckpointLayout.for_run(run_dir).entries()
    bests = [e for e in entries if e.role == "best"]
    steps = [e for e in entries if e.role == "step"]
    if which == "best":
        return _newest(bests)
    if step_match is not None:
        wanted = int(step_match.group(1))
        return _newest(e for e in steps if e.step == wanted)
    if steps:
        top = max(e.step for e in steps if e.step is not None)
        return _newest(e for e in steps if e.step == top)
    return _newest(bests)


def best_then_latest(run_dir: str | Path) -> tuple[Path | None, str | None]:
    """The checkpoint a finished run is read at, and which rule chose it (``None, None`` if none).

    The one owner of "best, else the latest step": the campaign orchestrator, the SLURM
    resolver and ``campaign infer`` all call it.
    """
    best = select(run_dir, "best")
    if best is not None:
        return best, "best"
    latest = select(run_dir, "latest")
    return (latest, "latest") if latest is not None else (None, None)


_PRE_V1_NAME = re.compile(r"checkpoint_.+\.(?:pt|pth|safetensors)")


def pre_v1_names(run_dir: str | Path) -> list[str]:
    """Files a pre-1.0 run left under ``checkpoints/``: never selected, only reported.

    The old writers named files ``checkpoint_*`` and published ``best.*`` as a symlink;
    ``parse`` refuses both, so a run holding only these has nothing to resume from
    until the converter has run.
    """
    directory = CheckpointLayout.for_run(run_dir).directory
    if not directory.is_dir():
        return []
    return sorted(
        p.name
        for p in directory.iterdir()
        if _PRE_V1_NAME.fullmatch(p.name) or (p.is_symlink() and p.name.startswith("best."))
    )


__all__ = [
    "CONVERSION_JOURNAL",
    "ConversionPendingError",
    "conversion_pending",
    "pre_v1_names",
    "refuse_pending_conversion",
    "select",
]
