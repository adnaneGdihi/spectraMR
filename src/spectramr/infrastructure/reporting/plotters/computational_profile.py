r"""Figure 1.15 — Computational profile.

Bar chart of per-stage runtime and peak memory (logarithmic where the
dynamic range is wide), plus one small panel per run fact (parameters, throughput,
wall-time) sourced from the ``split="run"`` rows that the aggregator folds
in from ``run_summary.json``. Real runs rarely log per-step stage timings,
so the run-fact panels are what make this figure fire outside synthetic
fixtures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    panel_label,
    pretty_label,
    save_figure,
    use_default_style,
)

# Run-level facts (metric key → axis unit), one small panel each: the units differ, so a
# shared axis would mislead and a value label would be clutter.
_RUN_FACTS: tuple[tuple[str, str], ...] = (
    ("params_m", "M"),
    ("iterations_per_sec", "it/s"),
    ("duration_min", "min"),
    ("effective_batch", "samples"),
)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    runtime_metrics: tuple[str, ...] = (
        "data_load_s",
        "forward_s",
        "backward_s",
        "optim_s",
        "log_s",
    ),
    memory_metrics: tuple[str, ...] = ("gpu_mem_gb", "cpu_mem_gb"),
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Build the runtime + memory + run-facts profile from training metrics."""
    out_path = Path(out_path)
    if df.empty or "metric" not in df.columns:
        return None

    rt_rows = df[df["metric"].isin(runtime_metrics)]
    mem_rows = df[df["metric"].isin(memory_metrics)]
    run_rows = df[df["split"] == "run"] if "split" in df.columns else df.iloc[0:0]
    run_facts = {
        k: run_rows[run_rows["metric"] == k]["value"].iloc[-1]
        for k, _fmt in _RUN_FACTS
        if not run_rows[run_rows["metric"] == k].empty
    }
    if rt_rows.empty and mem_rows.empty and not run_facts:
        return None

    use_default_style()
    n_panels = sum(1 for present in (not rt_rows.empty, not mem_rows.empty) if present)
    n_panels += len(run_facts)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.5 * n_panels, 2.4), squeeze=False)
    axes_flat = list(axes[0])
    panel_i = 0

    # Runtime bars (median across steps)
    if not rt_rows.empty:
        ax = axes_flat[panel_i]
        med = rt_rows.groupby("metric")["value"].median().reindex(runtime_metrics).dropna()
        x = np.arange(len(med))
        ax.bar(x, med.values, color="#0072B2", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_s", "") for m in med.index], rotation=20, ha="right", fontsize=7
        )
        ax.set_ylabel("median per-step time (s)")
        ax.set_title("runtime")
        if med.values.max() / max(med.values.min(), 1e-12) > 100:
            ax.set_yscale("log")
        if n_panels > 1:
            panel_label(ax, chr(ord("a") + panel_i))
        panel_i += 1

    # Memory bars (peak across steps)
    if not mem_rows.empty:
        ax = axes_flat[panel_i]
        peak = mem_rows.groupby("metric")["value"].max().reindex(memory_metrics).dropna()
        x = np.arange(len(peak))
        ax.bar(x, peak.values, color="#D55E00", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_gb", "") for m in peak.index], rotation=20, ha="right", fontsize=7
        )
        ax.set_ylabel("peak memory (GB)")
        ax.set_title("memory")
        if n_panels > 1:
            panel_label(ax, chr(ord("a") + panel_i))
        panel_i += 1

    for key, unit in _RUN_FACTS:
        if key not in run_facts:
            continue
        ax = axes_flat[panel_i]
        ax.bar([0], [run_facts[key]], width=0.5, color="#009E73", alpha=0.85)
        ax.set_xticks([])
        ax.set_ylabel(unit)
        ax.set_title(pretty_label(key))
        if n_panels > 1:
            panel_label(ax, chr(ord("a") + panel_i))
        panel_i += 1

    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary
