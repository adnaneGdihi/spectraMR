r"""Batch reporting: run :func:`generate_report` over every run under a root.

A "cohort" here is any directory tree of run outputs (e.g. the whole
``experiments/results/`` tree, or one paradigm's subtree). ``discover_run_dirs``
walks the tree, treating a directory as a *run* as soon as it carries a run
marker (``logs/training_metrics.csv`` / ``final_metrics.json`` / ...), and stops
descending into it. ``generate_reports`` then produces one report per run and a
top-level dashboard ``index.html`` linking them all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from spectramr.infrastructure.run_layout import PROFILE_SUBDIRS

from .pipeline import generate_report

logger = logging.getLogger(__name__)

# A directory is a run if it carries any of these markers.
_RUN_MARKERS = (
    "logs/training_metrics.csv",
    "logs/validation_metrics.csv",
    "final_metrics.json",
    "run_summary.json",
    "per_call_metrics.csv",
)

# Artifact subdirectories never worth descending into while discovering runs.
_SKIP_DIRS = {
    "report",
    "reports",
    "checkpoints",
    "logs",
    "debug_snapshots",
    "snapshots",
    "report_cases",
    "__pycache__",
    "figures",
    "tables",
    "images",
    "real_images",
    "fake_images",
    "tikz",
    "sensitivity_maps",
    ".git",
    # `spectramr profile` files a throwaway run under
    # `<arm run dir>/profile/<run_id>/run/`. It carries the same
    # run markers as a real one, so without this it would be discovered and
    # reported as an experiment -- but only for an arm that has NOT been trained
    # yet, because `_is_run_dir` stops the descent at a real run. That makes the
    # phantom appear and disappear depending on unrelated history, which is the
    # worst version of the bug. A profiling run is never an experiment run.
    *PROFILE_SUBDIRS,
}


def _is_run_dir(p: Path) -> bool:
    return any((p / m).exists() for m in _RUN_MARKERS)


def discover_run_dirs(root: str | Path) -> list[Path]:
    """Return every run directory at/under ``root`` (a run stops the descent)."""
    root = Path(root)
    if _is_run_dir(root):
        return [root]
    found: list[Path] = []
    stack = [root]
    while stack:
        d = stack.pop()
        if _is_run_dir(d):
            found.append(d)
            continue  # a run dir is a leaf for discovery — don't recurse in
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            continue
        for child in children:
            if child.name not in _SKIP_DIRS:
                stack.append(child)
    return sorted(found)


def write_batch_index(root: str | Path, rows: list[dict]) -> Path | None:
    """Write the cohort dashboard (``<root>/index.html``) linking every run's report. None if empty."""
    if not rows:
        return None
    from .dashboard import build_site

    out = build_site(root, batch_rows=rows)["index"]
    logger.info("batch report index: %s", out)
    return out


def generate_reports(root: str | Path, **report_kwargs: Any) -> dict[str, Any]:
    """Generate one report per run under ``root`` plus a linking index.

    ``report_kwargs`` are forwarded verbatim to :func:`generate_report` for each
    run (``method_name`` is overridden per-run with the run dir name). Per-run
    failures are captured, not raised, so one bad run cannot abort the batch.

    Returns ``{"runs": [...], "index": Path | None, "n_runs": int, "n_ok": int}``.
    """
    root = Path(root)
    report_kwargs.pop("method_name", None)  # per-run below
    run_dirs = discover_run_dirs(root)
    rows: list[dict] = []
    for d in run_dirs:
        row: dict[str, Any] = {"run": d}
        try:
            res = generate_report(d, method_name=d.name, **report_kwargs)
            row.update(
                ok=True,
                n_fig=sum(1 for v in res["figures"].values() if v is not None),
                n_tab=sum(1 for v in res["tables"].values() if v is not None),
                html=res.get("html"),
            )
        except Exception as exc:
            logger.warning("batch report: %s failed: %s", d, exc)
            row.update(ok=False, error=str(exc))
        rows.append(row)
    index = write_batch_index(root, rows)
    return {
        "runs": rows,
        "index": index,
        "n_runs": len(run_dirs),
        "n_ok": sum(1 for r in rows if r["ok"]),
    }
