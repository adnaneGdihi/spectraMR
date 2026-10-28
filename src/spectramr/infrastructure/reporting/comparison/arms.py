r"""Load a set of runs ("arms") as one comparable frame for the cross-run report.

Three inputs, each read at the pass :mod:`..views` selects for that arm:

* the tidy metric frame (:func:`..aggregator.aggregate_many`), one ``method`` per arm;
* per-case rows from each arm's ``per_call_metrics.csv`` (:mod:`..per_case`), or
  from its per-slice ``validation_batches.csv`` when it wrote none (:mod:`.slice_cases`);
* image cases, from the validation PNG drops when every arm has them (the same
  validation slices for every arm) and from ``report_cases/`` otherwise.

Image cases are aligned across arms *by target content*, never by position or
name: two arms' "case 0" are compared only if they show the same reference
slice. Arms whose references cannot be aligned contribute no mosaic row.

An arm whose ``run_summary.json`` records a crash (:mod:`..run_status`) is
dropped before any of the three is read, so no table or figure shows its
numbers; :attr:`Arms.excluded` keeps it for the summary and the manifest.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..aggregator import aggregate_many
from ..batch import discover_run_dirs
from ..cases.loader import load_image_pair_cases, load_report_cases
from ..per_case import call_unit, per_case_rows, read_per_call, validation_batch_size
from .slice_cases import slice_case_rows
from ..run_status import RunStatus, log_status, read_run_status
from ..views import checkpoint_steps

logger = logging.getLogger(__name__)


@dataclass
class Arms:
    """Everything the comparison plotters read, keyed by arm label."""

    labels: list[str]
    run_dirs: dict[str, Path]
    df: pd.DataFrame
    steps: dict[str, int] = field(default_factory=dict)
    per_case: pd.DataFrame = field(default_factory=pd.DataFrame)
    cases: dict[str, list[dict]] = field(default_factory=dict)
    view: str = "selected"
    call_unit: str = "validation calls (batch means)"
    #: completion state of every tabulated arm (``ok`` or ``unknown``)
    status: dict[str, RunStatus] = field(default_factory=dict)
    #: crashed arms left out of every table and figure, with their run dirs
    excluded: dict[str, RunStatus] = field(default_factory=dict)
    excluded_dirs: dict[str, Path] = field(default_factory=dict)
    #: ``metadata.comparison_note`` of each arm that declares one
    notes: dict[str, str] = field(default_factory=dict)


def comparison_note(run: Path) -> str | None:
    """``metadata.comparison_note`` from the run's ``resolved_config.json``.

    A run without the file, or one trained before the field existed, has no
    note; a note that is present but not a string raises.
    """
    import json

    try:
        meta = json.loads((Path(run) / "resolved_config.json").read_text()).get("metadata")
    except OSError:
        return None
    note = (meta or {}).get("comparison_note")
    if note is not None and not isinstance(note, str):
        raise ValueError(f"{run}: metadata.comparison_note must be a string, got {note!r}")
    return note or None


def short_labels(names: list[str]) -> list[str]:
    """Strip the underscore-delimited prefix and suffix every name shares."""
    if len(names) < 2:
        return list(names)
    parts = [n.split("_") for n in names]
    head = 0
    while all(len(p) > head + 1 for p in parts) and len({p[head] for p in parts}) == 1:
        head += 1
    tail = 0
    while all(len(p) > head + tail + 1 for p in parts) and len({p[-1 - tail] for p in parts}) == 1:
        tail += 1
    out = ["_".join(p[head : len(p) - tail]) for p in parts]
    return out if len(set(out)) == len(out) else list(names)


def resolve_runs(roots: list[str | Path]) -> list[Path]:
    """Every run directory named by, or found under, ``roots``; order kept."""
    runs: list[Path] = []
    for root in roots:
        for run in discover_run_dirs(root):
            if run not in runs:
                runs.append(run)
    return runs


def _target_key(case: dict) -> str:
    """Content key for a case: its acceleration and its reference slice.

    Insensitive to float round-off and scale; the acceleration is part of the
    key because one reference slice is reconstructed once per R.
    """
    arr = np.asarray(case["target"], dtype=np.float64)
    peak = float(np.abs(arr).max()) or 1.0
    digest = hashlib.sha1(np.round(arr / peak, 3).tobytes(), usedforsecurity=False).hexdigest()
    return f"{case.get('acceleration')}|{digest}"


def _image_cases(run: Path, step: int | None, prefer_png: bool) -> list[dict]:
    if prefer_png:
        cases = load_image_pair_cases(run, step=step)
        if cases:
            return cases
    payload = load_report_cases(run, task="reconstruction", step=step)
    return list(payload.get("cases_kspace", []))


def align_cases(cases: dict[str, list[dict]], max_rows: int) -> list[str]:
    """Case keys present in every arm, at most ``max_rows``, spread across accelerations.

    Taken round-robin over R so a four-row mosaic of a three-R cascade shows
    every R rather than four slices of the lowest one.
    """
    keyed = {label: {_target_key(c): c for c in cs} for label, cs in cases.items()}
    if not keyed or any(not v for v in keyed.values()):
        return []
    first = next(iter(keyed.values()))
    common = [k for k in first if all(k in v for v in keyed.values())]
    by_r: dict[str, list[str]] = {}
    for key in common:
        by_r.setdefault(key.split("|", 1)[0], []).append(key)
    picked: list[str] = []
    while len(picked) < max_rows and any(by_r.values()):
        for keys in by_r.values():
            if keys and len(picked) < max_rows:
                picked.append(keys.pop(0))
    return picked


def _case_rows(run: Path, *, step: int | None, label: str) -> pd.DataFrame:
    """Per-call rows when the run recorded them, else its per-slice rows (#2500)."""
    per_call = read_per_call(run)
    if per_call is not None:
        return per_case_rows(per_call, step=step, run_id=label, method=label)
    return slice_case_rows(run, step=step, run_id=label, method=label)


def load_arms(
    runs: list[str | Path],
    *,
    labels: list[str] | None = None,
    view: str = "selected",
    with_images: bool = True,
) -> Arms:
    """Load ``runs`` for comparison; ``labels`` default to :func:`short_labels`."""
    run_dirs = [Path(r) for r in runs]
    if len(run_dirs) < 2:
        raise ValueError(f"a comparison needs at least two runs, got {len(run_dirs)}")
    if not labels:
        names = [r.name for r in run_dirs]
        if len(set(names)) != len(names):
            # same basename in different folders (seeds, repeats): qualify by parent
            names = [f"{r.parent.name}_{r.name}" for r in run_dirs]
        labels = short_labels(names)
    labels = list(labels)
    if len(labels) != len(run_dirs):
        raise ValueError(f"{len(labels)} labels for {len(run_dirs)} runs")
    if len(set(labels)) != len(labels):
        raise ValueError(f"arm labels must be unique, got {labels}; pass distinct --labels")
    # Labels are fixed over every run first, so excluding one never renames the rest.
    status = {lab: read_run_status(run) for lab, run in zip(labels, run_dirs, strict=True)}
    excluded = {lab: s for lab, s in status.items() if s.state == "failed"}
    for lab, s in status.items():
        log_status(lab, s, action="excluded" if lab in excluded else "kept and marked")
    pairs = list(zip(labels, run_dirs, strict=True))
    excluded_dirs = {lab: run for lab, run in pairs if lab in excluded}
    kept = [(lab, run) for lab, run in pairs if lab not in excluded]
    if len(kept) < 2:
        named = ", ".join(f"{lab} ({s.label})" for lab, s in excluded.items())
        raise ValueError(
            f"a comparison needs at least two finished runs; {len(kept)} left after "
            f"excluding crashed runs: {named}"
        )
    labels = [lab for lab, _ in kept]
    run_dirs = [run for _, run in kept]
    df = aggregate_many(run_dirs, method_names=labels)
    by_run = checkpoint_steps(df, view)
    steps = {
        lab: by_run[f"{lab}_run{i}"] for i, lab in enumerate(labels) if f"{lab}_run{i}" in by_run
    }
    frames = [
        _case_rows(run, step=steps.get(lab), label=lab)
        for lab, run in zip(labels, run_dirs, strict=True)
    ]
    frames = [f for f in frames if not f.empty]
    per_case = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    arms = Arms(
        labels,
        dict(zip(labels, run_dirs, strict=True)),
        df,
        steps,
        per_case,
        view=view,
        call_unit=call_unit([validation_batch_size(r) for r in run_dirs]),
        status={lab: status[lab] for lab in labels},
        excluded=excluded,
        excluded_dirs=excluded_dirs,
    )
    arms.notes = {
        lab: note
        for lab, run in zip(labels, run_dirs, strict=True)
        if (note := comparison_note(run)) is not None
    }
    if with_images:
        prefer_png = all(load_image_pair_cases(r, max_cases=1) for r in run_dirs)
        arms.cases = {
            lab: _image_cases(run, steps.get(lab), prefer_png)
            for lab, run in zip(labels, run_dirs, strict=True)
        }
    return arms


__all__ = ["Arms", "align_cases", "load_arms", "resolve_runs", "short_labels"]
