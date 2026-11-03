"""``test/report.html``: the held-out test pass as one page.

Summary tables (overall, per contrast, per acquisition situation), one distribution panel per
metric, and the gallery of every rendered slice with its own metrics. Everything is read from
the files the test pass wrote, so the page can be rebuilt from a downloaded run.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.style import save_plot
from spectramr.infrastructure.reporting.test_split.slice_writer import (
    ERROR_WINDOW_FRACTION,
)
from spectramr.infrastructure.reporting.test_split.summary import (
    situation_columns,
    write_summary,
)

REPORT_HTML = "report.html"
DISTRIBUTIONS_PNG = "distributions.png"
#: Gallery columns, when the run reports them.
_GALLERY_METRICS = ("val_psnr", "val_ssim", "val_zf_psnr", "val_zf_ssim", "val_hfen")


def _group_label(row: pd.Series, keys: list[str]) -> str:
    parts = []
    for key in keys:
        value = row[key]
        if isinstance(value, float) and math.isnan(value):
            continue
        parts.append(f"R={value:g}" if "acceleration" in key else str(value))
    return " · ".join(parts) or "all"


def plot_distributions(long: pd.DataFrame, out_path: Path) -> Path | None:
    """Box + jittered strip per metric, one box per contrast x situation group."""
    long = long[np.isfinite(long["value"])]
    if long.empty:
        return None
    keys = [k for k in ("contrast", *situation_columns(long)) if k in long.columns]
    long = long.assign(group=long.apply(_group_label, axis=1, keys=keys))
    metrics = sorted(long["metric"].unique())
    ncols = min(3, len(metrics))
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, metric in zip(axes.flat, metrics, strict=False):
        part = long[long["metric"] == metric]
        groups = sorted(part["group"].unique())
        data = [part.loc[part["group"] == g, "value"].to_numpy() for g in groups]
        ax.boxplot(data, showfliers=False, widths=0.5)
        for i, values in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.18, 0.18, values.size), values, s=4, alpha=0.35)
        ax.set_xticks(range(1, len(groups) + 1), groups, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=10)
    for ax in list(axes.flat)[len(metrics) :]:
        ax.set_visible(False)
    fig.tight_layout()
    saved = save_plot(fig, out_path, dpi=110)
    plt.close(fig)
    return saved


def _summary_html(summary: pd.DataFrame) -> str:
    from ..dashboard.shell import table_html

    return "\n".join(
        f"<h3>{html.escape(str(grouping))}</h3>" + table_html(part.drop(columns="grouping"))
        for grouping, part in summary.groupby("grouping", sort=False)
    )


def _gallery_html(table: pd.DataFrame) -> str:
    if "image" not in table.columns or table["image"].isna().all():
        return "<p>No slice was rendered for this arm.</p>"
    order = [c for c in ("subject_id", "contrast", "slice_index") if c in table.columns]
    if "slice_index" in order:
        table = table.assign(_s=pd.to_numeric(table["slice_index"], errors="coerce"))
        order = [c if c != "slice_index" else "_s" for c in order]
    rows = []
    for _, row in table.sort_values(order).iterrows():
        if not isinstance(row.get("image"), str):
            continue
        ident = " / ".join(
            str(row[c])
            for c in ("subject_id", "contrast", "slice_index")
            if c in row and pd.notna(row[c])
        )
        cells = " ".join(
            f"{m.removeprefix('val_')}={row[m]:.4g}"
            for m in _GALLERY_METRICS
            if m in row and pd.notna(row[m])
        )
        rows.append(
            f"<figure><img loading='lazy' src='{html.escape(row['image'])}' alt='{html.escape(ident)}'>"
            f"<figcaption><span>{html.escape(ident)}</span><span>{html.escape(cells)}</span></figcaption></figure>"
        )
    return f"<div class='gallery'>{''.join(rows)}</div>"


def write_test_report(test_dir: Path, summary_json: Path | None = None) -> Path | None:
    """Write the tables, the distribution figure and ``report.html`` under ``test_dir``."""
    test_dir = Path(test_dir)
    table, long, summary = write_summary(test_dir)
    if table.empty:
        return None
    figure = plot_distributions(long, test_dir / DISTRIBUTIONS_PNG)
    meta = json.loads(summary_json.read_text()) if summary_json and summary_json.is_file() else {}
    from ..dashboard.shell import Section, figure_html, render_page

    out = test_dir / REPORT_HTML
    dist = figure_html(figure, out, caption="Metric distributions", wide=True) if figure else None
    sections = [
        Section("summary", "Summary", _summary_html(summary)),
        Section("distributions", "Distributions",
                f"<div class='gallery'>{dist}</div>" if dist else "<p class='empty'>No finite values.</p>"),
        Section("slices", "Every slice", _gallery_html(table),
                note="Panels: input, prediction, target and |prediction - target|, on the target's "
                f"99.5th-percentile window; the error panel uses {ERROR_WINDOW_FRACTION:g} of it."),
    ]
    prov = {k.replace("_", " ").capitalize(): meta[k]
            for k in ("manifest", "n_cases", "weights", "weights_iteration") if k in meta}
    render_page(out, title="Held-out test split", sections=sections, provenance=prov,
                crumbs=[(test_dir.parent.name, None), ("Test split", None)])
    return out


__all__ = ["DISTRIBUTIONS_PNG", "REPORT_HTML", "plot_distributions", "write_test_report"]
