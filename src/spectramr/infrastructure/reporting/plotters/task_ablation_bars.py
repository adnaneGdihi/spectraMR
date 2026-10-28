r"""Cohort task-ablation bar figure.

One figure per cohort *task*: a vertical stack of sub-panels, one per
validation metric (default SSIM / PSNR / LPIPS — they live on different
scales so they cannot share a y-axis). Within each panel, one bar per arm,
grouped by *method family* (``bNN``) with the full method(s) drawn first and
their one-knob *ablation* control(s) immediately after, coloured by role so a
reader sees the ablation delta family-by-family.

The plotter consumes a tidy per-arm frame with columns
``["family", "role", "arm", "metric", "value"]`` (one row per (arm, metric);
``value`` is the best-over-iterations scalar). It is deliberately generic —
the mrixfields-specific discovery/grouping lives in
:mod:`spectramr.infrastructure.reporting.cohort_ablation`.

Degenerate / collapsed arms (measurement-independent output, negative PSNR,
identity collapse) are shown honestly with their real value and marked with a
hatch rather than hidden — the anti-facade discipline applied to plots.

Run completion is marked as in the main-results table: a plotted arm whose
completion is not confirmed carries ‡ on its tick label and a footnote, and an
arm left out because it crashed is named in a "Not plotted" footnote.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    colour_for,
    panel_label,
    pretty_label,
    save_figure,
    use_default_style,
)

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_REQUIRED_COLS = frozenset({"family", "role", "arm", "metric", "value"})


# method / ablation / baseline are not registered method names, so colour_for
# returns the Okabe-Ito fallback at these distinct indices (blue / vermilion /
# green). ``baseline`` overlays external pretrained baselines (Task 8).
_ROLE_FALLBACK: dict[str, int] = {"method": 5, "ablation": 6, "baseline": 2}


def _role_colour(role: str) -> str:
    """Deterministic role colour, routed through the style palette (SSOT)."""
    return colour_for(f"__role_{role}", fallback_index=_ROLE_FALLBACK.get(role, 4))


def _family_sort_key(family: str) -> tuple[int, str]:
    """Natural ``bNN`` order (b11, b13, …, b19, b110); non-``bNN`` sink last."""
    digits = family[1:] if family.startswith("b") else family
    return (int(digits), family) if digits.isdigit() else (10**9, family)


def _order_arms(df: pd.DataFrame) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return arm plotting order + per-arm family/role maps.

    Arms are grouped by family (natural ``bNN`` order); within a family the
    method(s) come first, then the ablation(s), each sub-group sorted by name.
    """
    meta = df.drop_duplicates("arm")
    arms = meta["arm"].tolist()
    arm_family = dict(zip(arms, meta["family"].tolist(), strict=True))
    arm_role = dict(zip(arms, meta["role"].tolist(), strict=True))
    families = sorted(set(arm_family.values()), key=_family_sort_key)
    order: list[str] = []
    for fam in families:
        fam_arms = [a for a, f in arm_family.items() if f == fam]
        methods = sorted(a for a in fam_arms if arm_role[a] not in ("ablation", "baseline"))
        ablations = sorted(a for a in fam_arms if arm_role[a] == "ablation")
        baselines = sorted(a for a in fam_arms if arm_role[a] == "baseline")
        order.extend(methods + ablations + baselines)
    return order, arm_family, arm_role


def _arm_label(arm: str) -> str:
    """Short per-bar x label: strip the ``mrixfields_bNN_`` / ``bNN_`` prefix."""
    stem = arm[len("mrixfields_") :] if arm.startswith("mrixfields_") else arm
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].startswith("b") else stem


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    metrics: tuple[str, ...] = ("ssim", "psnr", "lpips"),
    lower_is_better: tuple[str, ...] = ("lpips", "nmse", "nrmse", "rmse", "mae", "mse", "hfen"),
    degenerate: set[str] | frozenset[str] | None = None,
    title: str | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    run_notes: dict[str, str] | None = None,
    excluded_runs: dict[str, str] | None = None,
) -> Path | None:
    """Faceted grouped-bar figure of best validation metrics per method family.

    Args:
        df: Tidy frame with ``[family, role, arm, metric, value]`` rows.
        metrics: Metrics to facet (one sub-panel each), in top-to-bottom order.
            Metrics absent from ``df`` are silently dropped.
        lower_is_better: Metrics annotated "lower = better" (LPIPS et al.).
        degenerate: Arm names to hatch + flag as collapsed / measurement-
            independent output (shown honestly, not hidden).
        title: Figure suptitle (e.g. the task name).
        run_notes: ``{arm: note}`` for plotted arms whose completion is not
            confirmed; each tick label gains ‡ and the note becomes a footnote.
        excluded_runs: ``{arm: note}`` for arms left out (e.g. crashed), named
            in a footnote so the omission is visible on the figure itself.

    Returns:
        Path to the primary (first-format) output, or ``None`` if the frame is
        empty / missing required columns / has none of the requested metrics.
    """
    if df is None or len(df) == 0 or not _REQUIRED_COLS.issubset(df.columns):
        return None
    present = [m for m in metrics if (df["metric"] == m).any()]
    if not present:
        return None

    degenerate = set(degenerate or ())
    out_path = Path(out_path)
    use_default_style()

    arm_order, _arm_family, arm_role = _order_arms(df)
    x = np.arange(len(arm_order))
    n = len(present)

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(max(6.0, 0.32 * len(arm_order) + 1.6), 1.9 * n + 0.7),
        sharex=True,
        squeeze=False,
    )
    axes = list(axes[:, 0])

    any_degenerate = False
    for i, metric in enumerate(present):
        ax = axes[i]
        values = df[df["metric"] == metric].set_index("arm")["value"].to_dict()
        heights = [float(values.get(arm, np.nan)) for arm in arm_order]
        colours = [_role_colour(arm_role[arm]) for arm in arm_order]
        bars = ax.bar(x, heights, width=0.8, color=colours, alpha=0.9)
        for xi, arm in enumerate(arm_order):
            if arm in degenerate:
                any_degenerate = True
                bars[xi].set_hatch("////")
                bars[xi].set_edgecolor("#333333")
                bars[xi].set_linewidth(0.6)
        ax.axhline(0.0, color="#999999", lw=0.5)  # visible zero line for negative PSNR
        ax.set_ylabel(pretty_label(metric))
        direction = "lower = better" if metric in lower_is_better else "higher = better"
        ax.text(
            0.997,
            1.02,
            direction,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5,
            color="#666666",
        )
        panel_label(ax, _LETTERS[i])

    # Family group separators (every panel) + family headers above the top panel
    # (kept clear of the rotated per-arm tick labels at the bottom).
    top = axes[0]
    bottom = axes[-1]
    prev_family = None
    group_start = 0
    for xi, arm in enumerate(arm_order):
        fam = _arm_family[arm]
        if prev_family is not None and fam != prev_family:
            for ax in axes:
                ax.axvline(xi - 0.5, color="#dddddd", lw=0.6, zorder=0)
            _family_header(top, group_start, xi - 1, prev_family)
            group_start = xi
        prev_family = fam
    if arm_order and prev_family is not None:
        _family_header(top, group_start, len(arm_order) - 1, prev_family)

    bottom.set_xticks(x)
    notes = {a: n for a, n in (run_notes or {}).items() if a in arm_order}
    labels = [f"{_arm_label(a)} ‡" if a in notes else _arm_label(a) for a in arm_order]
    bottom.set_xticklabels(labels, rotation=90, fontsize=5)
    footnotes = [f"‡ {_arm_label(a)}: {n}." for a, n in notes.items()]
    footnotes += [f"Not plotted: {_arm_label(a)} ({n})." for a, n in (excluded_runs or {}).items()]
    bottom.set_xlim(-0.7, len(arm_order) - 0.3)

    handles = [
        Patch(facecolor=_role_colour("method"), label="full method"),
        Patch(facecolor=_role_colour("ablation"), label="ablation"),
    ]
    if any(r == "baseline" for r in arm_role.values()):
        handles.append(Patch(facecolor=_role_colour("baseline"), label="baseline"))
    if any_degenerate:
        handles.append(
            Patch(facecolor="#ffffff", edgecolor="#333333", hatch="////", label="degenerate output")
        )

    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold", y=0.985)
    # Reserve a strip below the tick labels for the status footnotes, one line
    # each, kept clear of the provenance stamp along the bottom edge.
    foot_h = 0.03 + 0.02 * len(footnotes) if footnotes else 0.0
    fig.tight_layout(rect=(0.0, foot_h, 1.0, 0.88))
    # Legend outside the data area (a single row below the title) so it never
    # overlaps the bars.
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(handles),
        frameon=False,
        fontsize=6,
    )
    if footnotes:
        fig.text(0.01, 0.025, "\n".join(footnotes), ha="left", va="bottom", fontsize=5)
    if metadata is not None:
        stamp_figure(fig, metadata)

    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0], out_path)
    if metadata is not None:
        write_sidecar(primary, metadata)
    return primary


def _family_header(ax, i0: int, i1: int, family: str) -> None:
    """Centre a bold family label just above the top panel's group.

    Sits at y=1.09 (clear of the per-panel "higher/lower = better" hint at y=1.02)
    — the rightmost family's header otherwise lands in the same top-right corner as
    that fixed annotation and the two overlap illegibly whenever the last family
    group is narrow (1-2 bars, e.g. an external ``ref_*`` baseline pair).
    """
    ax.text(
        (i0 + i1) / 2.0,
        1.09,
        family,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
        color="#333333",
    )
