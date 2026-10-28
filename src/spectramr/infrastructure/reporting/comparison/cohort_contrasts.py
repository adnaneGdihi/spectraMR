r"""Contrasts drawn from a cohort directory, for any cohort without writing a file.

``vs_baseline`` compares every other arm with one named arm and Holm-corrects them
together; ``pairs`` writes every pair of arms, each alone and uncorrected (a pair
chosen after looking at the data has no family to correct against). Arms are the
run directories under the cohort root, labelled by their directory name.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

from .arms import resolve_runs
from .contrasts import ArmRef, Contrast, ContrastSet

MODES: tuple[str, ...] = ("vs_baseline", "pairs")
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _folder_name(*parts: str) -> str:
    name = _UNSAFE.sub("_", "__vs__".join(parts)).lstrip("._-")
    return name or "contrast"


def cohort_contrasts(
    cohort: str | Path,
    *,
    mode: str = "vs_baseline",
    baseline: str | None = None,
    view: str = "selected",
    metrics: tuple[str, ...] | None = None,
) -> ContrastSet:
    """The contrasts of every arm found under ``cohort``.

    Raises:
        ValueError: unknown ``mode``, fewer than two runs, a ``vs_baseline`` set
            without a baseline, or a baseline that is not a run directory name.
    """
    if mode not in MODES:
        raise ValueError(f"unknown contrast mode {mode!r}; choose from {list(MODES)}")
    runs = resolve_runs([cohort])
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"run directory names under {cohort} are not unique: {names}")
    if len(runs) < 2:
        raise ValueError(f"{cohort} holds {len(runs)} run(s); a contrast needs at least two")
    refs = {
        r.name: ArmRef(run=str(r), label=r.name, group="baseline" if r.name == baseline else None)
        for r in runs
    }
    if mode == "vs_baseline":
        if baseline is None:
            raise ValueError("--baseline names the arm every other arm is tested against")
        if baseline not in refs:
            raise ValueError(f"baseline {baseline!r} is not a run under {cohort}; runs are {names}")
        contrasts = tuple(
            Contrast(
                name=_folder_name(n, baseline),
                arms=(refs[baseline], refs[n]),
                baseline=baseline,
                family="vs_" + _folder_name(baseline),
            )
            for n in names
            if n != baseline
        )
    else:
        contrasts = tuple(
            Contrast(name=_folder_name(a, b), arms=(refs[b], refs[a]), baseline=b)
            for a, b in combinations(names, 2)
        )
    return ContrastSet(view=view, metrics=metrics, contrasts=contrasts)


__all__ = ["MODES", "cohort_contrasts"]
