r"""Write each declared contrast alone into ``<out-root>/<name>/``.

A folder holds everything one comparison of a paper needs and nothing from the
others: the rendered figures and tables (:func:`.report.generate_comparison`),
the numbers behind them as CSV, the paired per-subject statistics, and a
``contrast.json`` naming the runs, the pass read for each arm and the settings.

    <name>/
      contrast.json            declared contrast, resolved runs, passes, settings
      comparison_summary.md    figures and tables written
      report_manifest.json     per-figure status (written by generate_comparison)
      figures/  tables/        rendered by the --compare report
      data/arm_metrics.csv     tidy metric frame (every logged metric and pass) of each arm
      data/all_metrics.csv     one row per arm: every validation metric at the pass read
      data/metrics_long.csv    arm x phase x metric: mean, spread, n, unit, direction
      data/metrics_<phase>.csv one row per arm, one column per metric (val, train, test)
      data/per_case.csv        per-case rows behind the bar and sweep figures
      data/paired_stats.csv    one row per (metric, arm) tested against the baseline
      data/paired_subjects/    per-subject values behind each test
      data/unavailable.json    comparisons that could not be made, with the reason

Holm correction runs across every contrast of the set that shares a ``family``,
so a p-value in one folder depends on its siblings; the family and its size are
written into each row so the folder stays interpretable alone.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..metadata import run_provenance
from .arms import load_arms
from .contrast_data import write_data
from .contrast_stats import (
    apply_holm,
    base_of,
    contrast_stats,
)
from .contrasts import Contrast, ContrastSet
from .metric_selection import check_phases
from .report import generate_comparison

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContrastResult:
    name: str
    folder: Path
    arms: list[str]
    baseline: str
    n_tests: int
    unavailable: list[dict[str, str]]


def _select(contrast_set: ContrastSet, only: list[str] | None) -> tuple[Contrast, ...]:
    if not only:
        return contrast_set.contrasts
    known = {c.name for c in contrast_set.contrasts}
    unknown = sorted(set(only) - known)
    if unknown:
        raise ValueError(f"unknown contrast(s) {unknown}; the set declares {sorted(known)}")
    return tuple(c for c in contrast_set.contrasts if c.name in only)


def _run_dirs(
    contrast_set: ContrastSet, contrast: Contrast, results_root: Path | None
) -> list[Path]:
    dirs = [contrast_set.run_dir(a, results_root) for a in contrast.arms]
    absent = [str(d) for d in dirs if not d.is_dir()]
    if absent:
        raise FileNotFoundError(f"contrast {contrast.name!r}: run directories not found: {absent}")
    return dirs


def _write_contrast_json(
    folder: Path,
    contrast: Contrast,
    contrast_set: ContrastSet,
    arms: Any,
    figures: dict,
    used: dict,
) -> None:
    record = {
        "phases": {
            phase: {
                "metrics": used.get(phase, []),
                "paired_tests": {
                    "val": "per-slice rows, over subjects",
                    "test": "inference cases",
                }.get(phase, "none: training writes no per-item rows; see data/metrics_train.csv"),
            }
            for phase in used
        },
        "figures": {
            fid: (p.relative_to(folder).as_posix() if p else None) for fid, p in figures.items()
        },
        "contrast": contrast.model_dump(),
        "view": contrast_set.view,
        "metrics": list(contrast_set.metrics)
        if contrast_set.metrics
        else "every metric both arms logged per slice",
        "runs": {
            lab: {
                "run_dir": str(arms.run_dirs[lab]),
                "pass_read": arms.steps.get(lab),
                **run_provenance(arms.run_dirs[lab]),
            }
            for lab in arms.labels
        },
        "excluded_runs": {lab: s.label for lab, s in arms.excluded.items()},
    }
    (folder / "contrast.json").write_text(json.dumps(record, indent=2, default=str) + "\n")


def write_contrasts(
    contrast_set: ContrastSet,
    out_root: str | Path,
    *,
    only: list[str] | None = None,
    results_root: str | Path | None = None,
    phases: tuple[str, ...] | None = None,
    train_window: int = 50,
    max_panels: int = 12,
    style: str = "nature",
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    html_report: bool = True,
) -> list[ContrastResult]:
    """Render and store every selected contrast; returns one result per folder.

    Statistics are computed for the selected contrasts and the rest of their Holm
    families before any folder is written, so a corrected p-value does not depend
    on which contrasts were selected.
    """
    out_root = Path(out_root)
    phases = check_phases(phases or contrast_set.phases)
    selected = _select(contrast_set, only)
    root = Path(results_root) if results_root is not None else None
    # A Holm family is corrected as a whole, so a sibling outside --only is still tested
    # (and needs its runs) even though its folder is not written.
    families = {c.family for c in selected if c.family is not None}
    tested = tuple(c for c in contrast_set.contrasts if c in selected or c.family in families)
    loaded = []
    for contrast in tested:
        dirs = _run_dirs(contrast_set, contrast, root)
        arms = load_arms(dirs, labels=contrast.labels, view=contrast_set.view, with_images=False)
        stats = contrast_stats(
            contrast.name,
            arms,
            contrast_set.metrics,
            contrast.baseline_label,
            contrast.family,
            phases,
        )
        loaded.append((contrast, dirs, arms, stats))
    apply_holm([row for _, _, _, stats in loaded for row in stats.rows])
    loaded = [entry for entry in loaded if entry[0] in selected]
    results = []
    for contrast, dirs, arms, stats in loaded:
        folder = out_root / contrast.name
        folder.mkdir(parents=True, exist_ok=True)
        report = generate_comparison(
            dirs,
            folder,
            labels=contrast.labels,
            baseline=contrast.baseline_label,
            view=contrast_set.view,
            metrics=[base_of(m) for m in contrast_set.metrics] if contrast_set.metrics else "all",
            phases=phases,
            max_panels=max_panels,
            train_window=train_window,
            stats=stats,
            ladder=contrast.ladder,
            groups=contrast.group_map,
            style=style,
            formats=formats,
            dpi=dpi,
            html_report=html_report,
        )
        used = write_data(
            folder,
            arms,
            stats,
            stats.rows,
            contrast.baseline_label,
            (contrast_set, phases, train_window),
        )
        _write_contrast_json(folder, contrast, contrast_set, arms, report["figures"], used)
        results.append(
            ContrastResult(
                contrast.name,
                folder,
                arms.labels,
                contrast.baseline_label,
                len(stats.rows),
                stats.unavailable,
            )
        )
    _write_index(out_root, results, selected)
    return results


def _write_index(
    out_root: Path, results: list[ContrastResult], selected: tuple[Contrast, ...]
) -> None:
    """``INDEX.csv`` lists every contrast folder; a partial run replaces only its own rows."""
    by_name = {c.name: c for c in selected}
    fresh = pd.DataFrame(
        [
            {
                "contrast": r.name,
                "folder": r.name,
                "arms": ";".join(r.arms),
                "baseline": r.baseline,
                "family": by_name[r.name].family,
                "tests": r.n_tests,
                "unavailable": len(r.unavailable),
            }
            for r in results
        ]
    )
    path = out_root / "INDEX.csv"
    if path.is_file():
        kept = pd.read_csv(path)
        fresh = pd.concat(
            [kept[~kept["contrast"].isin(fresh["contrast"])], fresh], ignore_index=True
        )
    fresh.to_csv(path, index=False)


__all__ = ["ContrastResult", "write_contrasts"]
