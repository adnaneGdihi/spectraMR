r"""Which metrics a comparison reports: every one the arms logged, or a stated list.

The aggregator already turns training, validation and inference results into one
frame (``split`` = ``train`` / ``val`` / ``test``); nothing there favours PSNR or
SSIM. This module is the one place that decides which base metrics a comparison
shows, so the figures, the tables and the statistics cannot disagree.

``None`` keeps a caller's own default (``--compare`` shows its headline panels),
``"all"`` is every base metric found, and a list is checked against what was
logged: a name no phase has raises rather than producing an empty panel.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .arms import Arms

PHASES: tuple[str, ...] = ("val", "train", "test")
#: Shown first, in this order, when present; the rest follow alphabetically.
LEADING: tuple[str, ...] = ("psnr", "ssim", "nmse", "hfen", "lpips")


def _is_reference(name: str) -> bool:
    """``zf_<metric>`` is the zero-filled input's score, a reference line, not a result."""
    return name.startswith("zf_")


def check_phases(phases: Sequence[str]) -> tuple[str, ...]:
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise ValueError(f"unknown phase(s) {unknown}; choose from {list(PHASES)}")
    return tuple(dict.fromkeys(phases))


def _ordered(names: set[str]) -> list[str]:
    lead = [m for m in LEADING if m in names]
    return lead + sorted(names - set(lead))


def discover_metrics(arms: Arms, phases: Sequence[str] = PHASES) -> dict[str, list[str]]:
    """``{phase: base metrics}`` that at least one arm logged in that phase."""
    frame = arms.df
    found: dict[str, list[str]] = {}
    for phase in check_phases(phases):
        rows = frame[(frame["split"] == phase) & frame["headline"] & frame["value"].notna()]
        names = {str(m) for m in rows["base_metric"].unique() if not _is_reference(str(m))}
        found[phase] = _ordered(names)
    return found


def select_metrics(
    spec: str | Sequence[str] | None, found: dict[str, list[str]]
) -> dict[str, list[str]] | None:
    """The metrics to report in each phase, or ``None`` when ``spec`` is ``None``.

    Raises:
        ValueError: a requested name that no phase logged.
    """
    if spec is None:
        return None
    everything = _ordered({m for names in found.values() for m in names})
    if isinstance(spec, str):
        if spec != "all":
            raise ValueError(f"metrics must be 'all' or a list of names, got {spec!r}")
        return {phase: list(names) for phase, names in found.items()}
    unknown = [m for m in spec if m not in everything]
    if unknown:
        raise ValueError(f"no phase logged {unknown}; logged metrics are {everything}")
    return {phase: [m for m in spec if m in names] for phase, names in found.items()}


def union(selected: dict[str, list[str]]) -> list[str]:
    """Every selected metric once, in display order."""
    return _ordered({m for names in selected.values() for m in names})


def frame_of(arms: Arms, phase: str) -> pd.DataFrame:
    """Headline rows of one phase, references dropped."""
    rows = arms.df[(arms.df["split"] == phase) & arms.df["headline"] & arms.df["value"].notna()]
    return rows[~rows["base_metric"].astype(str).str.startswith("zf_")]


__all__ = [
    "LEADING",
    "PHASES",
    "check_phases",
    "discover_metrics",
    "frame_of",
    "select_metrics",
    "union",
]
