r"""Cross-arm metric comparison with honest error bars.

With per-case rows (``per_call_metrics.csv`` at each arm's checkpoint pass):
one panel per metric, x = acceleration, one colour per arm; every case is a
faint point, the arm's mean is a marker with a bootstrap 95 % CI, ``n`` is
printed, and arms are tested against ``baseline`` with a paired bootstrap on
the shared cases, Holm-corrected within each (metric, R).

Without per-case rows: one value per arm at its checkpoint, drawn as a dot with
*no* bar, and the panel says so -- a bar drawn over validation passes would be
the training trajectory, not uncertainty.

The zero-filled identity score (``zf_<metric>``), when logged, is a dotted grey
reference: an arm below it has learned nothing the input did not hold.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from spectramr.core.metrics.metric_directions import resolve_direction

from ..legend import arm_palette, collect_handles, reserve_legend_space, shared_legend, sparse_ticks
from ..metadata import RunMetadata, stamp_figure, write_sidecar
from ..plotters._stats import bootstrap_ci, holm_bonferroni, paired_bootstrap_pvalue
from ..style import (
    column_width,
    panel_label,
    pretty_label,
    save_figure,
    use_default_style,
)
from ..tables._headline import _headline_rows
from ..views import snapshot
from ._grid import MAX_PANELS, PANEL_COLUMNS
from ._grid import chunks as _chunks
from .arms import Arms

DEFAULT_METRICS: tuple[str, ...] = ("psnr", "ssim", "nmse", "hfen", "lpips")


def _label(metric: str) -> str:
    higher = resolve_direction(metric)
    arrow = "" if higher is None else (" ↑" if higher else " ↓")
    return pretty_label(metric) + arrow


def _pvalues(rows: pd.DataFrame, labels: list[str], baseline: str | None) -> dict[str, float]:
    """Holm-adjusted paired p-values of each arm against ``baseline`` on shared cases."""
    if baseline is None or baseline not in labels:
        return {}
    key = "pair_key" if "pair_key" in rows.columns else "case_id"
    base = rows[rows["method"] == baseline].set_index(key)["value"]
    raw: dict[str, float] = {}
    for lab in labels:
        if lab == baseline:
            continue
        other = rows[rows["method"] == lab].set_index(key)["value"]
        shared = base.index.intersection(other.index)
        if len(shared) >= 3:
            raw[lab] = paired_bootstrap_pvalue(base[shared].to_numpy(), other[shared].to_numpy())
    if not raw:
        return {}
    return dict(zip(raw, holm_bonferroni(list(raw.values())), strict=True))


def _per_case_panel(
    ax, rows, fallback, labels, colours, baseline, identity, metric, stats
) -> None:
    """Per-case panel; arms without per-case rows are hollow markers at their logged value."""
    groups = sorted(rows["acceleration"].dropna().unique()) or [np.nan]
    width = 0.8 / len(labels)
    for g_i, r in enumerate(groups):
        sub = rows[rows["acceleration"] == r] if not np.isnan(r) else rows
        pvals = _pvalues(sub, labels, baseline)
        for a_i, lab in enumerate(labels):
            x = g_i - 0.4 + width * (a_i + 0.5)
            vals = sub.loc[sub["method"] == lab, "value"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                logged = fallback[
                    (fallback["method"] == lab)
                    & (
                        fallback["acceleration"] == r
                        if not np.isnan(r)
                        else fallback["acceleration"].isna()
                    )
                ]["value"]
                if not logged.empty:
                    ax.plot(
                        x,
                        float(logged.iloc[0]),
                        "o",
                        ms=4,
                        mfc="none",
                        mec=colours[lab],
                        mew=1.0,
                        zorder=4,
                        label=lab if g_i == 0 else None,
                    )
                continue
            jitter = np.random.default_rng(a_i).uniform(-width / 4, width / 4, vals.size)
            ax.scatter(x + jitter, vals, s=3, color=colours[lab], alpha=0.25, linewidths=0)
            lo, hi = bootstrap_ci(vals) if vals.size > 1 else (np.nan, np.nan)
            mean = float(vals.mean())
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - lo], [hi - mean]] if np.isfinite(lo) else None,
                fmt="o",
                ms=3.5,
                color=colours[lab],
                mec="black",
                mew=0.3,
                capsize=2,
                label=lab if g_i == 0 else None,
                zorder=4,
            )
            stats.append(
                {
                    "metric": metric,
                    "group": "all" if np.isnan(r) else r,
                    "arm": lab,
                    "n": int(vals.size),
                    "mean": mean,
                    "p_holm": pvals.get(lab, np.nan),
                }
            )
        if identity is not None:
            ref = identity[identity["acceleration"] == r] if not np.isnan(r) else identity
            if not ref.empty:
                ax.hlines(
                    ref["value"].mean(),
                    g_i - 0.45,
                    g_i + 0.45,
                    colors="#777777",
                    linestyles=":",
                    linewidth=0.9,
                )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["all" if np.isnan(r) else f"R = {r:g}" for r in groups])


def _snapshot_panel(ax, rows, labels, colours, identity) -> None:
    for a_i, lab in enumerate(labels):
        vals = rows.loc[rows["method"] == lab, "value"]
        if not vals.empty:
            ax.plot(a_i, float(vals.iloc[0]), "o", ms=5, color=colours[lab], mec="black", mew=0.3)
    if identity is not None and not identity.empty:
        ax.axhline(float(identity["value"].mean()), color="#777777", ls=":", lw=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    arms: Arms | None = None,
    metrics: list[str] | None = None,
    baseline: str | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    max_panels: int = MAX_PANELS,
    **_ignored,
) -> Path | None:
    """Draw one figure of ``metrics`` panels, or several when there are more than ``max_panels``.

    The first figure is written to ``out_path``; the rest beside it as
    ``<stem>_2``, ``<stem>_3`` and so on. Returns the first.
    """
    if arms is None or len(arms.labels) < 2:
        return None
    if max_panels < 1:
        raise ValueError(f"max_panels must be at least 1, got {max_panels}")
    colours = arm_palette(arms.labels)
    per_case = arms.per_case
    covered = set(per_case["method"]) if not per_case.empty else set()
    by_case = bool(covered)
    # one owner for "which value is this arm's headline": the results table's picker
    headline = _headline_rows(arms.df, arms.view)
    logged = snapshot(arms.df, arms.view)
    source = per_case if by_case else headline
    if source.empty:
        return None
    labels = list(arms.labels)
    missing = [lab for lab in labels if lab not in covered] if by_case else []
    # a metric the headline picker did not take (it prefers the test split) is still
    # drawn from the validation pass the report reads
    headline = pd.concat([headline, logged[logged["headline"]]], ignore_index=True)
    present = set(source["base_metric"]) | set(headline["base_metric"])
    wanted = [m for m in (metrics or DEFAULT_METRICS) if m in present]
    if not wanted:
        return None
    notes = []
    if missing:
        notes.append("hollow: no per-case rows (" + ", ".join(missing) + ")")
    if by_case and baseline is not None and baseline not in covered:
        notes.append(f"no tests: baseline {baseline} has no per-case rows")
    if by_case:
        notes.append(f"n = {arms.call_unit}")
    lacking = [m for m in wanted if m not in set(source["base_metric"])]
    if lacking or not by_case:
        notes.append("one value per arm where there are no per-case rows")
    out_path = Path(out_path)
    primary = None
    for k, chunk in enumerate(_chunks(list(wanted), max_panels), start=1):
        target = out_path if k == 1 else out_path.with_name(f"{out_path.stem}_{k}{out_path.suffix}")
        stats: list[dict] = []
        drawn = _figure(
            chunk,
            target,
            source,
            headline,
            logged,
            labels,
            colours,
            missing,
            by_case,
            baseline,
            arms,
            notes,
            stats,
        )
        if metadata is not None:
            stamp_figure(drawn, metadata)
        written = save_figure(drawn, target.parent, target.stem, formats=formats, dpi=dpi)
        plt.close(drawn)
        if stats:
            pd.DataFrame(stats).to_csv(target.with_suffix(".stats.csv"), index=False)
        path = written.get(formats[0])
        if metadata is not None and path is not None:
            write_sidecar(path, metadata)
        primary = primary or path
    return primary


def _figure(
    chunk,
    target,
    source,
    headline,
    logged,
    labels,
    colours,
    missing,
    by_case,
    baseline,
    arms,
    notes,
    stats,
):
    use_default_style()
    cols = min(len(chunk), PANEL_COLUMNS)
    rows_n = -(-len(chunk) // cols)
    width = column_width("double")
    fig, axes = plt.subplots(rows_n, cols, figsize=(width, width * 0.36 * rows_n), squeeze=False)
    for i, metric in enumerate(chunk):
        ax = axes[i // cols][i % cols]
        rows = source[source["base_metric"] == metric]
        # a metric logged only once per pass (no per-case rows) falls back to that value
        per_case_panel = by_case and not rows.empty
        rows = rows if per_case_panel else headline[headline["base_metric"] == metric]
        ident = source[source["base_metric"] == f"zf_{metric}"]
        ident = ident if not ident.empty else None
        if per_case_panel:
            fallback = logged[(logged["base_metric"] == metric) & logged["method"].isin(missing)]
            _per_case_panel(ax, rows, fallback, labels, colours, baseline, ident, metric, stats)
        else:
            _snapshot_panel(ax, rows, labels, colours, ident)
        sparse_ticks(ax)
        ax.set_ylabel(_label(metric))
        if len(chunk) > 1:
            panel_label(ax, chr(ord("a") + i))
    for j in range(len(chunk), rows_n * cols):
        axes[j // cols][j % cols].set_visible(False)
    if by_case:
        handles, names = collect_handles(axes[0])
        if source[source["base_metric"].str.startswith("zf_")].shape[0]:
            handles.append(Line2D([], [], color="#777777", ls=":", lw=0.9))
            names.append("zero-filled")
        shared_legend(fig, handles, names, anchor_y=-0.04)
    if notes:
        fig.suptitle("; ".join(notes), fontsize="x-small")
    fig.tight_layout(rect=(0, reserve_legend_space(fig), 1, 0.95 if notes else 1))
    return fig
