r"""Cohort-level task-ablation figure assembly.

Turns a directory of completed arm runs (one sub-dir per arm, each carrying
``resolved_config.json`` + ``logs/validation_metrics.csv``) into the three
per-task validation-metric bar figures produced by the
``cohort_task_ablation_bars`` plotter.

Grouping is derived self-contained from each arm's ``resolved_config.json``:

* **task**   — ``metadata.tags.task`` (``"1"`` / ``"2"`` / ``"3"``).
* **family** — the ``bNN`` token parsed from the arm directory name (robust
  even when the tags are sparse; cross-checked against ``metadata.tags.arm``).
* **role**   — ``"ablation"`` if ``_ablate_`` appears in the arm name, else
  ``"method"`` (so variant siblings like ``b15_brenier_intensity_*`` are
  methods, not ablations).

Per-arm scalars reuse the reporting SSOT reader
(:func:`spectramr.infrastructure.reporting.aggregator.aggregate`) — the
best-over-iterations validation value (max, or min for ``lower_is_better``
metrics), matching each run's ``restore_best_weights`` early-stopping semantics.
Arms with no validation metrics (e.g. a calibration-only run) contribute no
rows and are skipped.

Completion follows the ``--compare`` policy, read through
:func:`.run_status.read_run_status`: an arm whose ``run_summary.json`` records
``success: false`` is left out of every figure and named in the log and the
figure footnote, because its metric files may belong to an earlier attempt. An
arm whose status is unknown is kept and marked ‡.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.aggregator import aggregate
from spectramr.infrastructure.reporting.plotters import get as get_plotter
from spectramr.infrastructure.reporting.run_status import (
    RunStatus,
    log_status,
    read_run_status,
    status_notes,
)

logger = logging.getLogger(__name__)

DEFAULT_METRICS: tuple[str, ...] = ("ssim", "psnr", "lpips")
LOWER_IS_BETTER: tuple[str, ...] = (
    "lpips",
    "nmse",
    "nrmse",
    "rmse",
    "mae",
    "mse",
    "hfen",
    "gaussian_nll",
)

_ARM_FAMILY_RE = re.compile(r"^(?:mrixfields_)?(b\d+)_")
# Strips a trailing ablation-role suffix off metadata.tags.arm (e.g.
# "bloch_synth_ablate" -> "bloch_synth"; "b11_ablation" -> "b11") so a
# non-bNN method/ablation pair collapses onto one family key (see _family_of).
_TAG_ABLATION_SUFFIX_RE = re.compile(r"_ablat(?:e|ion).*$")

# Task display names for the mrixfields2026 cohort (metadata.tags.task -> title).
TASK_TITLES: dict[str, str] = {
    "1": "Task 1 — any→7T synthesis",
    "2": "Task 2 — ULF 0.1T restoration",
    "3": "Task 3 — any-to-any cross-field translation",
}

# Arms that collapsed to a degenerate / measurement-independent output in the
# 2026-07 cohort run (negative PSNR, near-zero SSIM, identity collapse). Shown
# honestly and hatched in the figure — never hidden (anti-facade discipline).
DEGENERATE_ARMS: frozenset[str] = frozenset(
    {
        "mrixfields_b29_heteroscedastic_ulf",
        "mrixfields_b29_ablate_var_prior",
        "mrixfields_b35_field_cfg",
        "mrixfields_b35_ablate_guidance",
        "mrixfields_b33_field_bridge",
        "mrixfields_b110_ablate_pin",
    }
)


@dataclass(frozen=True)
class ArmInfo:
    """Grouping facts for a single arm result directory."""

    arm: str
    dir: Path
    task: str | None
    family: str
    role: str
    paradigm: str | None
    baseline: str | None
    primary_metric: str | None
    status: RunStatus = field(default_factory=lambda: RunStatus("unknown", "not read"))


def _family_of(arm: str, tag_arm: str | None = None) -> str:
    """Parse the ``bNN`` family token from an arm name.

    Non-``bNN`` arms (paradigm-named families like ``bloch_synth``, or external
    ``ref_*`` baselines) have no numeric token to parse, so a method arm
    (``bloch_synth_7t``) and its ablation sibling (``bloch_synth_ablate_dispersion``)
    would each become their own one-arm "family" — two adjacent single-bar groups
    whose header labels collide in the figure. Fall back to ``metadata.tags.arm``
    with any ablation-role suffix stripped, so siblings collapse onto one shared
    key; final fallback is the arm name itself.
    """
    m = _ARM_FAMILY_RE.match(arm)
    if m:
        return m.group(1)
    if tag_arm:
        stem = _TAG_ABLATION_SUFFIX_RE.sub("", tag_arm)
        if stem:
            return stem
    return arm


def _read_resolved_config(arm_dir: Path) -> dict:
    path = arm_dir / "resolved_config.json"
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("resolved_config.json unreadable at %s: %s", path, exc)
        return {}


def discover_cohort_arms(cohort_root: str | Path, *, prefix: str = "mrixfields_") -> list[ArmInfo]:
    """Discover every ``{prefix}*`` arm directory under ``cohort_root``.

    Reads each arm's ``resolved_config.json`` for task / baseline / primary
    metric and derives family + role from the directory name.
    """
    root = Path(cohort_root)
    arms: list[ArmInfo] = []
    for arm_dir in sorted(p for p in root.glob(f"{prefix}*") if p.is_dir()):
        arm = arm_dir.name
        cfg = _read_resolved_config(arm_dir)
        meta = cfg.get("metadata") or {}
        tags = meta.get("tags") or {}
        task = tags.get("task")
        task = str(task) if task is not None else None
        tag_arm = tags.get("arm")
        family = _family_of(arm, tag_arm)
        if tag_arm and tag_arm != family:
            logger.debug(
                "family token mismatch for %s: parsed %r vs tags.arm %r",
                arm,
                family,
                tag_arm,
            )
        role = "ablation" if "_ablate_" in arm else "method"
        arms.append(
            ArmInfo(
                arm=arm,
                dir=arm_dir,
                task=task,
                family=family,
                role=role,
                paradigm=tags.get("paradigm"),
                baseline=meta.get("baseline"),
                primary_metric=meta.get("primary_metric"),
                status=read_run_status(arm_dir),
            )
        )
    return arms


def best_val_metrics(
    arm_dir: str | Path,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    *,
    lower_is_better: tuple[str, ...] = LOWER_IS_BETTER,
) -> dict[str, float]:
    """Best-over-iterations validation scalar for each requested metric.

    Reuses :func:`aggregator.aggregate` (the reporting SSOT reader), filters to
    the ``val`` split, matches ``val_<m>`` (or bare ``<m>``) and reduces with
    ``max`` (or ``min`` for ``lower_is_better``). Returns ``{}`` when the arm
    has no validation metrics.
    """
    df = aggregate(arm_dir)
    if df.empty or "split" not in df.columns:
        return {}
    val = df[df["split"] == "val"].copy()
    if val.empty:
        return {}
    out: dict[str, float] = {}
    for metric in metrics:
        # Validation-CSV columns melt to ``val_<m>``; accept a bare ``<m>`` too.
        for key in (f"val_{metric}", metric):
            series = val.loc[val["metric"] == key, "value"].dropna()
            if not series.empty:
                arr = series.to_numpy().astype(float)
                out[metric] = float(arr.min() if metric in lower_is_better else arr.max())
                break
    return out


def build_task_frame(
    arms: list[ArmInfo],
    task: str,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Tidy ``[family, role, arm, metric, value]`` frame for one task; crashed arms are left out."""
    rows: list[dict] = []
    for arm in arms:
        if arm.task != task:
            continue
        log_status(arm.arm, arm.status, action=_status_action(arm.status))
        if arm.status.state == "failed":
            continue
        best = best_val_metrics(arm.dir, metrics)
        for metric, value in best.items():
            rows.append(
                {
                    "family": arm.family,
                    "role": arm.role,
                    "arm": arm.arm,
                    "metric": metric,
                    "value": value,
                }
            )
    return pd.DataFrame(rows, columns=pd.Index(["family", "role", "arm", "metric", "value"]))


def _status_action(status: RunStatus) -> str:
    return "excluded from the figure" if status.state == "failed" else "kept and marked"


def status_kwargs(arms: list[ArmInfo], task: str) -> dict[str, dict[str, str]]:
    """Plotter kwargs that mark unknown-status arms and name excluded ones for ``task``."""
    in_task = {a.arm: a.status for a in arms if a.task == task}
    excluded = {name: s.label for name, s in in_task.items() if s.state == "failed"}
    kept = {name: s for name, s in in_task.items() if s.state != "failed"}
    return {"run_notes": status_notes(kept), "excluded_runs": excluded}


def generate_task_ablation_figures(
    cohort_root: str | Path,
    out_dir: str | Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    tasks: tuple[str, ...] = ("1", "2", "3"),
    prefix: str = "mrixfields_",
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    degenerate: frozenset[str] = DEGENERATE_ARMS,
) -> dict[str, Path | None]:
    """Emit one task-ablation bar figure per task.

    Returns ``{task: primary_output_path | None}`` (None when a task has no
    arms with validation metrics).
    """
    arms = discover_cohort_arms(cohort_root, prefix=prefix)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Provenance stamp — best-effort git detection; never fatal for a plot run.
    try:
        from spectramr.infrastructure.reporting.metadata import detect_metadata

        meta = detect_metadata(dataset_version="mrixfields2026")
    except Exception as exc:  # provenance is defensive, not load-bearing
        logger.debug("metadata detection skipped: %s", exc)
        meta = None

    make = get_plotter("cohort_task_ablation_bars")
    written: dict[str, Path | None] = {}
    for task in tasks:
        frame = build_task_frame(arms, task, metrics)
        fig_stem = f"task{task}_ablation_bars"
        out_path = out_dir / f"{fig_stem}.{formats[0]}"
        title = TASK_TITLES.get(task, f"Task {task}")
        if frame.empty:
            logger.warning("task %s: no arms with validation metrics — skipped", task)
            written[task] = None
            continue
        written[task] = make(
            frame,
            out_path,
            metrics=metrics,
            lower_is_better=LOWER_IS_BETTER,
            degenerate=degenerate,
            title=title,
            formats=formats,
            dpi=dpi,
            metadata=meta,
            **status_kwargs(arms, task),
        )
    return written


# External-baseline metric key -> figure metric name. Baselines report
# ``lpips_alex`` (mapped onto the ``lpips`` panel) and ``ssim``; ``psnr`` is not
# produced by the challenge baselines, so no baseline bar appears on that panel.
_BASELINE_METRIC_MAP: dict[str, str] = {
    "ssim": "ssim",
    "lpips_alex": "lpips",
    "psnr": "psnr",
    "nrmse_l2": "nrmse_l2",
}


def merge_baseline_results(
    task_frame: pd.DataFrame,
    baseline_results: list[dict],
    task: str,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Append ``role="baseline"`` overlay rows to a task frame.

    Each external baseline *method* (``cut`` / ``cyclegan`` / ``stargan_v2``) is
    aggregated to ONE bar per figure metric = the mean of that method's
    per-eval-unit metric over all its specialized units in ``task`` (a task-3
    StarGAN spans many field pairs; a task-1 CUT spans field-contrast units).
    ``family`` = method (a non-``bNN`` family, so it sinks to the right of the
    method arms), ``role`` = ``"baseline"``.

    ``baseline_results`` is the parsed ``baseline_results.json`` list written by
    :func:`spectramr.pipelines.eval_baselines.run_baseline_evaluation`.
    """
    task_int = int(task)
    by_method: dict[str, list[dict]] = {}
    for r in baseline_results:
        try:
            if int(r.get("task", -1)) == task_int:
                by_method.setdefault(str(r["method"]), []).append(r)
        except (TypeError, ValueError):
            continue

    rows: list[dict] = []
    for method, results in sorted(by_method.items()):
        for fig_metric in metrics:
            src_keys = [bk for bk, fm in _BASELINE_METRIC_MAP.items() if fm == fig_metric]
            vals: list[float] = []
            for res in results:
                mm = res.get("metrics_mean", {}) or {}
                for bk in src_keys:
                    v = mm.get(bk)
                    if v is not None and v == v:  # drop None + NaN (v == v is False for NaN)
                        vals.append(float(v))
            if not vals:
                continue
            rows.append(
                {
                    "family": method,
                    "role": "baseline",
                    "arm": f"baseline_{method}",
                    "metric": fig_metric,
                    "value": sum(vals) / len(vals),
                }
            )
    if not rows:
        return task_frame
    cols = ["family", "role", "arm", "metric", "value"]
    return pd.concat([task_frame, pd.DataFrame(rows, columns=pd.Index(cols))], ignore_index=True)


def generate_baseline_overlay_figures(
    cohort_root: str | Path,
    baseline_results_json: str | Path,
    out_dir: str | Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    tasks: tuple[str, ...] = ("1", "2", "3"),
    prefix: str = "mrixfields_",
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    degenerate: frozenset[str] = DEGENERATE_ARMS,
) -> dict[str, Path | None]:
    """Regenerate the per-task bar figures with external baselines overlaid.

    Reads the eval runner's ``baseline_results.json``, aggregates each baseline
    method to one bar per metric (:func:`merge_baseline_results`), overlays it on
    the method-cohort task frame, and saves ``task{N}_with_baselines.{pdf,png}``.
    Returns ``{task: primary_output_path | None}``.
    """
    baseline_results = json.loads(Path(baseline_results_json).read_text())
    arms = discover_cohort_arms(cohort_root, prefix=prefix)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from spectramr.infrastructure.reporting.metadata import detect_metadata

        meta = detect_metadata(dataset_version="mrixfields2026")
    except Exception as exc:  # provenance is defensive, not load-bearing
        logger.debug("metadata detection skipped: %s", exc)
        meta = None

    make = get_plotter("cohort_task_ablation_bars")
    written: dict[str, Path | None] = {}
    for task in tasks:
        frame = build_task_frame(arms, task, metrics)
        frame = merge_baseline_results(frame, baseline_results, task, metrics=metrics)
        out_path = out_dir / f"task{task}_with_baselines.{formats[0]}"
        title = f"{TASK_TITLES.get(task, f'Task {task}')} (+ baselines)"
        if frame.empty:
            logger.warning("task %s: no method arms and no baselines — skipped", task)
            written[task] = None
            continue
        written[task] = make(
            frame,
            out_path,
            metrics=metrics,
            lower_is_better=LOWER_IS_BETTER,
            degenerate=degenerate,
            title=title,
            formats=formats,
            dpi=dpi,
            metadata=meta,
            **status_kwargs(arms, task),
        )
    return written
