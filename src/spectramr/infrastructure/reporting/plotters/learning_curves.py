r"""Figure 1.2 — Learning curves.

Default: one panel per quantity (loss, PSNR, SSIM) with train and validation on
the same axes, the noisy per-step training trace drawn faint under its EMA, the
zero-filled identity as a reference line, and the early-stopping pass marked
(:mod:`._curve_panels`). With an explicit ``metrics`` list: one panel per metric
with a +/-1 sigma band over seeds sharing a ``method``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.legend import (
    arm_palette,
    collect_handles,
    reserve_legend_space,
    shared_legend,
    sparse_ticks,
)
from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters._curve_panels import Panel, default_panels, ema
from spectramr.infrastructure.reporting.style import (
    column_width,
    panel_label,
    pretty_label,
    save_figure,
    use_default_style,
)
from spectramr.infrastructure.reporting.views import recorded_steps


def _log_ok(values: pd.Series) -> bool:
    # matplotlib warns (not raises) on a non-positive log axis; warnings are not OK
    v = values.dropna()
    return bool(len(v)) and bool((v > 0).all()) and float(v.max() / v.min()) >= 20.0


def _draw_panel(ax, panel: Panel, methods: list[str], selected: dict[str, int]) -> None:
    many = len(methods) > 1
    palette = arm_palette(methods)
    for method in methods:
        colour = palette[method]
        val_ls = "-"
        tr = panel.train[panel.train["method"] == method].groupby("step")["value"].mean()
        if not tr.empty and len(methods) <= 3:
            if not many:
                ax.plot(tr.index, tr.values, color=colour, alpha=0.18, lw=0.5)
            ax.plot(tr.index, ema(tr.to_numpy()), color=colour, lw=0.9, ls="--" if many else "-",
                    label=None if many else "train (EMA)")
        va = panel.val[panel.val["method"] == method].groupby("step")["value"].mean()
        if not va.empty:
            ax.plot(va.index, va.values, color=colour if many else "#D55E00", marker="o",
                    ms=2.2, lw=1.0, ls=val_ls, label=method if many else "validation")
        run_ids = panel.val.loc[panel.val["method"] == method, "run_id"].unique()
        step = selected.get(str(run_ids[0])) if len(run_ids) else None
        if step is not None and step in va.index:
            ax.plot([step], [va.loc[step]], marker="*", ms=6, color=colour if many else "#D55E00",
                    markeredgecolor="black", markeredgewidth=0.3, zorder=5)
    if not panel.identity.empty:
        ident = panel.identity.groupby("step")["value"].mean()
        ax.axhline(float(ident.iloc[-1]), color="#777777", lw=0.7, ls=":",
                   label="zero-filled input")
    values = pd.concat([panel.train["value"], panel.val["value"]])
    if panel.title == "loss" and _log_ok(values):
        ax.set_yscale("log")
    ax.set_title(pretty_label(panel.title))
    ax.set_xlabel("training step")


def _default_figure(df: pd.DataFrame):
    panels = default_panels(df)
    if not panels:
        return None
    use_default_style()
    rows = pd.concat([p.val for p in panels] + [p.train for p in panels])
    present = set(rows["method"].astype(str))
    # the frame's own order (the arm order of a comparison) keeps colours stable
    methods = [str(m) for m in pd.unique(df["method"]) if str(m) in present]
    selected = recorded_steps(df)
    width = column_width("double")
    fig, axes = plt.subplots(1, len(panels), figsize=(width, width * 0.3), squeeze=False)
    for idx, panel in enumerate(panels):
        ax = axes[0][idx]
        _draw_panel(ax, panel, methods, selected)
        if len(panels) > 1:
            panel_label(ax, chr(ord("a") + idx))
    handles, labels = collect_handles(axes[0])
    if len(methods) > 1 and len(methods) <= 3 and any(not p.train.empty for p in panels):
        # several arms: colour is the arm, line style is the split
        from matplotlib.lines import Line2D

        handles += [Line2D([], [], color="#555555", marker="o", ms=2.2, lw=1.0),
                    Line2D([], [], color="#555555", ls="--", lw=0.9)]
        labels += ["validation", "train (EMA)"]
    shared_legend(fig, handles, labels)
    for ax in axes[0]:
        sparse_ticks(ax)
    fig.tight_layout(rect=(0, reserve_legend_space(fig), 1, 1))
    return fig


def _explicit_figure(df: pd.DataFrame, metrics: list[str], log_y: bool, band: bool):
    sub = df[df["metric"].isin(metrics)]
    sub = sub[sub["step"].fillna(-1) >= 0]
    metrics = [m for m in metrics if (sub["metric"] == m).any()]
    if not metrics:
        return None
    use_default_style()
    n = len(metrics)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 1.9 * rows), squeeze=False)
    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        per_metric = sub[sub["metric"] == metric]
        palette = arm_palette([str(m) for m in sub["method"].unique()])
        for method in per_metric["method"].unique():
            grouped = per_metric[per_metric["method"] == method].groupby("step")["value"]
            mean = grouped.mean()
            # ddof=1: the band is +/-1 sigma over seeds, which are a sample.
            std = grouped.std(ddof=1).fillna(0)
            colour = palette[str(method)]
            ax.plot(mean.index, mean.values, color=colour, label=str(method))
            if band and not (std == 0).all():
                ax.fill_between(mean.index, mean - std, mean + std, color=colour, alpha=0.2, lw=0)
        ax.set_title(pretty_label(metric))
        if log_y and (per_metric["value"].dropna() > 0).all() and len(per_metric):
            ax.set_yscale("log")
        if idx + cols >= n:
            ax.set_xlabel(pretty_label("step"))
        if n > 1:
            panel_label(ax, chr(ord("a") + idx))
        sparse_ticks(ax)
    for empty in range(n, rows * cols):
        axes[empty // cols][empty % cols].set_axis_off()
    shared_legend(fig, *collect_handles(axes.ravel()))
    fig.tight_layout(rect=(0, reserve_legend_space(fig), 1, 1))
    return fig


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    metrics: list[str] | None = None,
    log_y: bool = True,
    band: bool = True,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the learning-curves figure; ``None`` when no stepped rows exist."""
    if df is None or df.empty or "metric" not in df.columns or "step" not in df.columns:
        return None
    if "method" not in df.columns:
        df = df.assign(method="run")
    if "run_id" not in df.columns:
        df = df.assign(run_id=df["method"])
    fig = (
        _explicit_figure(df, metrics, log_y, band)
        if metrics is not None
        else _default_figure(df)
    )
    if fig is None:
        return None
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    plt.close(fig)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
