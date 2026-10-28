r"""Paired per-subject statistics of one contrast, and Holm correction across a family.

Every non-baseline arm is paired with the baseline slice by slice
(:func:`..per_subject_stats.pair_on_slices`), each subject's matched slices are
averaged, and the tests run on subjects. The pass read for an arm is the one the
aggregate tables read (``Arms.steps``), so a figure and its statistics never
describe different checkpoints.

A comparison that cannot be made is recorded with its reason in ``unavailable``
and never as a p-value of 1: a run without per-slice rows, a metric the rows lack,
or fewer than two shared subjects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from spectramr.core.metrics.statistical_tests import StatisticalTests

from ..metric_names import SPLIT_PREFIXES
from ..per_slice import SlicePass, find_rows_files, read_pass, slice_values
from ..per_subject_stats import PairedSubjects, compare_subjects, pair_on_slices
from .arms import Arms
from .phase_tables import inference_unavailable

logger = logging.getLogger(__name__)


#: Columns of ``paired_stats.csv``; fixed so a contrast with nothing testable still has a header.
PAIRED_COLUMNS: tuple[str, ...] = (
    "contrast",
    "phase",
    "metric",
    "variant",
    "baseline",
    "iteration_variant",
    "iteration_baseline",
    "scale",
    "unit",
    "n_subjects",
    "n_slices",
    "nonfinite_variant",
    "nonfinite_baseline",
    "mean_diff",
    "ci_low",
    "ci_high",
    "cohens_d",
    "p_ttest",
    "p_wilcoxon",
    "holm_family",
    "holm_m",
    "p_holm",
)


@dataclass
class ContrastStats:
    """What one contrast produced: tested rows, the paired subject tables, and what was skipped."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    paired: dict[tuple[str, str], PairedSubjects] = field(default_factory=dict)
    unavailable: list[dict[str, str]] = field(default_factory=list)


def base_of(name: str) -> str:
    """A metric name without its split prefix: ``val_psnr`` and ``psnr`` are one metric."""
    for prefix in SPLIT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _slice_passes(arms: Arms) -> tuple[dict[str, SlicePass], dict[str, str]]:
    passes: dict[str, SlicePass] = {}
    missing: dict[str, str] = {}
    for label in arms.labels:
        files = find_rows_files(arms.run_dirs[label])
        if not files:
            missing[label] = (
                f"{arms.run_dirs[label]} wrote no per-slice rows (validation_batches.csv)"
            )
            continue
        try:
            passes[label] = read_pass(files, arms.steps.get(label))
        except ValueError as exc:  # rows that cannot be paired (batch means, no identity)
            missing[label] = str(exc)
    return passes, missing


def _shared_metrics(a: SlicePass, b: SlicePass) -> list[str]:
    """Metric columns both passes logged with at least one finite value each.

    The rows file can carry text the loader attached beyond the identifiers the
    reader strips; a column with no number in it is not a metric.
    """
    shared = sorted(set(a.metric_columns) & set(b.metric_columns))
    return [m for m in shared if slice_values(a, m)[0].size and slice_values(b, m)[0].size]


def _row(contrast, metric, variant, baseline, paired, tests, family, phase, iterations) -> dict:
    ci = tests["bootstrap_ci"]
    return {
        "contrast": contrast,
        "phase": phase,
        "metric": metric,
        "variant": variant,
        "baseline": baseline,
        "iteration_variant": iterations[0],
        "iteration_baseline": iterations[1],
        "scale": paired.scale,
        "unit": "cases" if phase == "test" else "subjects",
        "n_subjects": paired.n_subjects,
        "n_slices": paired.n_slices,
        "nonfinite_variant": paired.nonfinite_dropped["a"],
        "nonfinite_baseline": paired.nonfinite_dropped["b"],
        "mean_diff": tests["paired_ttest"]["mean_diff"],
        "ci_low": ci["ci_lower"],
        "ci_high": ci["ci_upper"],
        "cohens_d": tests["effect_size"]["cohens_d"],
        "p_ttest": tests["paired_ttest"]["p_value"],
        "p_wilcoxon": tests["wilcoxon"]["p_value"],
        "holm_family": family,
        "holm_m": None,
        "p_holm": None,
    }


def _record(out, contrast, metric, variant, baseline, paired, family, phase, iterations) -> None:
    """Test one pairing, or note why it cannot be tested."""
    if paired is None or paired.n_subjects < 2:
        n = 0 if paired is None else paired.n_subjects
        reason = (
            f"{n} unit(s) scored by both arms" if paired is not None else "an arm has no values"
        )
        out.unavailable.append(
            {"phase": phase, "metric": metric, "variant": variant, "reason": reason}
        )
        return
    out.paired[(phase, metric, variant)] = paired
    tests = compare_subjects(paired)
    out.rows.append(
        _row(contrast, metric, variant, baseline, paired, tests, family, phase, iterations)
    )


def _column(passes: list[SlicePass], base: str) -> str | None:
    """The per-slice column carrying ``base`` in every pass: ``val_<base>``, else ``base``."""
    for candidate in (f"val_{base}", base):
        if all(candidate in p.metric_columns for p in passes):
            return candidate
    return None


def _val_stats(out, contrast, arms, metrics, baseline, family) -> None:
    passes, missing = _slice_passes(arms)
    for variant in (lab for lab in arms.labels if lab != baseline):
        reason_pair = missing.get(variant) or missing.get(baseline)
        wanted = metrics
        if wanted is None:
            wanted = (
                tuple(base_of(c) for c in _shared_metrics(passes[variant], passes[baseline]))
                if reason_pair is None
                else ("(all)",)
            )
        for metric in wanted:
            pair = None if reason_pair else [passes[variant], passes[baseline]]
            column = _column(pair, metric) if pair else None
            reason = reason_pair or (None if column else f"{metric} is not a column of the rows")
            if reason is not None:
                out.unavailable.append(
                    {"phase": "val", "metric": metric, "variant": variant, "reason": reason}
                )
                logger.warning(
                    "%s: %s vs %s on %s: %s", contrast, variant, baseline, metric, reason
                )
                continue
            paired = pair_on_slices(passes[variant], passes[baseline], column, column)
            iterations = (passes[variant].iteration, passes[baseline].iteration)
            _record(out, contrast, metric, variant, baseline, paired, family, "val", iterations)


def _test_stats(out, contrast, arms, metrics, baseline, family) -> None:
    from .inference_rows import pair_on_cases
    from .metric_selection import frame_of

    test = frame_of(arms, "test")
    logged = {lab: set(test.loc[test["method"] == lab, "base_metric"]) for lab in arms.labels}
    reasons = {
        lab: {u["metric"]: u["reason"] for u in inference_unavailable(arms.run_dirs[lab], lab)}
        for lab in arms.labels
    }
    for variant in (lab for lab in arms.labels if lab != baseline):
        shared = sorted(logged[variant] & logged[baseline])
        for metric in metrics if metrics is not None else shared:
            lacking = [lab for lab in (variant, baseline) if metric not in logged[lab]]
            if lacking:
                why = "; ".join(
                    f"{lab}: {reasons[lab].get(metric, 'not in final_eval.json')}"
                    for lab in lacking
                )
                out.unavailable.append(
                    {"phase": "test", "metric": metric, "variant": variant, "reason": why}
                )
                continue
            paired = pair_on_cases(arms, variant, baseline, metric)
            _record(out, contrast, metric, variant, baseline, paired, family, "test", (None, None))


def contrast_stats(
    contrast: str,
    arms: Arms,
    metrics: tuple[str, ...] | None,
    baseline: str,
    family: str | None,
    phases: tuple[str, ...] = ("val",),
) -> ContrastStats:
    """Test every non-baseline arm against ``baseline`` on each metric, per phase.

    ``metrics=None`` tests every metric the two arms both logged. A declared
    metric an arm lacks is reported as unavailable with the reason, never dropped.
    ``val`` pairs per-slice rows on subjects, ``test`` pairs inference cases; ``train``
    has no per-item rows, so it is compared descriptively in the phase tables only.
    """
    out = ContrastStats()
    if metrics is not None:
        metrics = tuple(dict.fromkeys(base_of(m) for m in metrics))
    if "val" in phases:
        _val_stats(out, contrast, arms, metrics, baseline, family)
    if "test" in phases:
        _test_stats(out, contrast, arms, metrics, baseline, family)
    return out


def apply_holm(rows: list[dict[str, Any]]) -> None:
    """Fill ``p_holm`` and ``holm_m`` in place: Holm within each (family, phase, metric).

    A row whose contrast declared no family keeps ``p_holm = None``, which the
    CSV carries as empty: it was not corrected, and nothing pretends it was.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["holm_family"] is not None:
            key = (row["holm_family"], row["phase"], row["metric"])
            groups.setdefault(key, []).append(row)
    for members in groups.values():
        adjusted = StatisticalTests.holm_bonferroni_correction([r["p_ttest"] for r in members])
        for row, p in zip(members, adjusted["corrected_p_values"], strict=True):
            row["p_holm"], row["holm_m"] = float(p), len(members)


def paired_subject_frame(paired: PairedSubjects, variant: str, baseline: str) -> pd.DataFrame:
    """The per-subject values behind a test: variant, baseline and their difference."""
    frame = paired.table.rename(columns={"a": variant, "b": baseline})
    frame["difference"] = frame[variant] - frame[baseline]
    return frame


__all__ = [
    "PAIRED_COLUMNS",
    "ContrastStats",
    "apply_holm",
    "contrast_stats",
    "paired_subject_frame",
]
