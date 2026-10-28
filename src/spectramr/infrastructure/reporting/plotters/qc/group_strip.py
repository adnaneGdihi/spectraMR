r"""QC group IQM distribution — one horizontal strip per metric.

For every metric/loss the run computed, draw a horizontal box (q1-q3, median,
1.5*IQR whiskers) with a jittered per-case strip, flagging Tukey outliers
(beyond 1.5·IQR) in a highlight colour and labelling them by ``case_id``.

The metric set is *discovered* from the data — preferring the unbounded
``per_call_metrics.csv`` (passed as ``per_call_df``), then the recorded-case
``predictions_df``, then the aggregate frame — so it tracks whatever the run
computed rather than a hardcoded IQM list. Returns None when no per-case metric
values are available (soft-skip), or when none of them has any spread (#503).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import (
    RunMetadata,
    stamp_figure,
    write_sidecar,
)
from spectramr.infrastructure.reporting.plotters.qc._strip_series import (  # noqa: F401
    _NON_METRIC,
    _collect,
    _one_pass,
    _series_from_long,
    _series_from_wide,
)
from spectramr.infrastructure.reporting.style import (
    column_width,
    panel_label,
    pretty_label,
    save_figure,
    use_default_style,
)

logger = logging.getLogger(__name__)

# Fewer observations than this and there are no quartiles to speak of.
_MIN_BOX_N = 4

def _tukey_outliers(vals: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (outlier_mask, whisker_lo, whisker_hi) via the 1.5·IQR rule."""
    if vals.size < 4:
        return np.zeros(vals.size, bool), float(vals.min()), float(vals.max())
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (vals < lo) | (vals > hi)
    inl = vals[~mask]
    wlo = float(inl.min()) if inl.size else float(vals.min())
    whi = float(inl.max()) if inl.size else float(vals.max())
    return mask, wlo, whi


def _draw_hbox(ax, vals: np.ndarray, *, y: float = 1.0, h: float = 0.5) -> np.ndarray:
    """Draw a horizontal box (q1-q3, median, whiskers). Return the outlier mask.

    Below ``_MIN_BOX_N`` observations the box is OMITTED rather than collapsed to
    ``q1 = med = q3``. A rectangle asserts a spread, and the degenerate one was
    not even invisible: ``max(q3 - q1, 1e-9)`` clamped it to a thin sliver, which
    renders as an extremely TIGHT distribution — the most confident claim on the
    chart drawn from the least data (#503). The median tick and the min-max span
    still show, because those are things the data actually says.
    """
    mask, wlo, whi = _tukey_outliers(vals)
    ax.plot([wlo, whi], [y, y], color="#888888", lw=0.6, zorder=1)
    if vals.size >= _MIN_BOX_N:
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.add_patch(
            plt.Rectangle(
                (q1, y - h / 2),
                q3 - q1,
                h,
                facecolor="#56B4E9",
                alpha=0.35,
                edgecolor="#2c6f9c",
                lw=0.6,
                zorder=2,
            )
        )
    else:
        med = float(np.median(vals))
    ax.plot([med, med], [y - h / 2, y + h / 2], color="#333333", lw=1.0, zorder=3)
    return mask


def _has_spread(vals: np.ndarray) -> bool:
    """True when the values could describe a distribution at all.

    #503: on ``exp_vf_01`` the per-case sink received the batch AGGREGATE — one
    row per validation call, not per case — so eight "cases" all carried the
    single value ``48.71364215``. This plotter drew them as a box-and-whisker
    while its siblings (``fig_1_5_predicted_vs_true``, ``mri_a1_sr_triptych``,
    ``mri_a11_cohort_table``) honestly reported "skipped (no data)" on the SAME
    report. One repeated scalar is not a distribution, and a panel that draws one
    is read as evidence.
    """
    return vals.size >= 2 and float(np.ptp(vals)) > 0.0


def _drop_degenerate(series: dict) -> dict:
    """Filter out metrics with no spread, naming what was dropped.

    Never a SILENT skip: a missing panel has to be explainable from the run log,
    or "this metric was flat" and "this plotter is broken" look identical.
    """
    kept = {m: s for m, s in series.items() if _has_spread(s[0])}
    dropped = sorted(set(series) - set(kept))
    if dropped:
        logger.warning(
            "qc_group_strip: %d metric(s) have no spread across cases and were "
            "not drawn — a single value repeated is not a distribution (#503): %s",
            len(dropped),
            ", ".join(dropped),
        )
    return kept


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    per_call_df=None,
    predictions_df=None,
    metrics=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    checkpoint_step: int | None = None,
    view: str = "selected",
    **_ignored,
) -> Path | None:
    series = _collect(df, per_call_df, predictions_df, checkpoint_step, view)
    if metrics:
        series = {m: series[m] for m in metrics if m in series}
    series = _drop_degenerate(series)
    if not series:
        return None
    names = sorted(series)
    use_default_style()
    n = len(names)
    # Accelerations are separate distributions: one sub-row per R, never pooled.
    levels = [
        np.unique(series[m][2]) if series[m][2] is not None else np.array([np.nan])
        for m in names
    ]
    heights = [0.3 * len(lv) + 0.25 for lv in levels]
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(column_width("double"), sum(heights) + 0.6),
        squeeze=False,
        gridspec_kw={"height_ratios": heights},
    )
    for i, name in enumerate(names):
        ax = axes[i][0]
        vals, _, groups = series[name]
        rows = np.ones(vals.size)
        out_mask = np.zeros(vals.size, dtype=bool)
        if groups is None:
            out_mask = _draw_hbox(ax, vals)
        else:
            for k, level in enumerate(levels[i], start=1):
                sel = groups == level
                rows[sel] = k
                if sel.sum():
                    out_mask[sel] = _draw_hbox(ax, vals[sel], y=float(k))
        # seeded local RNG → reproducible jitter, never touches the global RNG
        rng = np.random.default_rng(i)
        yj = rows + rng.uniform(-0.16, 0.16, vals.size)
        ax.scatter(vals[~out_mask], yj[~out_mask], s=6, color="#0072B2", alpha=0.6, zorder=4)
        ax.scatter(
            vals[out_mask],
            yj[out_mask],
            s=16,
            color="#D55E00",
            edgecolor="black",
            linewidth=0.3,
            zorder=5,
        )
        ax.set_ylim(0.4, len(levels[i]) + 0.6)
        if groups is None:
            ax.set_yticks([])
        else:
            ax.set_yticks(range(1, len(levels[i]) + 1))
            ax.set_yticklabels([f"{lv:g}\u00d7" for lv in levels[i]], fontsize=4.5)
        ax.set_ylabel(pretty_label(name), rotation=0, ha="right", va="center", fontsize=6)
        panel_label(ax, chr(ord("a") + i)) if n > 1 else None
    fig.suptitle("Group IQM distribution", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
