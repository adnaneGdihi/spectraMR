"""Diagnostic figure generator for the six-method meta-evaluation pipeline.

Each function takes the ``MetaEvaluationOutput`` produced by
``MetaEvaluationPipeline.run`` and writes one PNG to ``out_dir``. The
suite mirrors the figures discussed in
``docs/meta_evaluation_rankers.md`` plus two presentation extras:

1. ``spectral_shape_grid``       — CDSCR per-metric descriptor sensitivity
2. ``lgdr_eigenscale_curve``     — Spearman ρ vs. diffusion time per metric
3. ``sfa_polar``                 — score-alignment polar plot
4. ``esd_fingerprint_matrix``    — invariance / sensitivity heatmap
5. ``fspd_biplot``               — FPCA mode 1 vs. mode 2 metric loadings
6. ``sim2rank_breakdown``        — ADR / SCVR / CDRS bar grid
7. ``consensus_rank_bump``       — final-rank z-score bump chart
9. ``metric_trajectories``       — per-metric severity curves + contact sheet
10. ``metric_correlation_clustering`` — clustered Pearson heatmap
11. ``sensitivity_scatter``      — ADR vs. SCVR plane
12. ``degradation_mosaic``       — family × severity image grid
13. ``normalized_leaderboard``   — headline [0,1] consensus leaderboard
14. ``severity_response_heatmap``— metric × family Spearman ρ(value, θ)

``render_all`` runs every figure back-to-back and returns the written
paths. Styling, the constrained-layout figure factory, and the
top-3/bottom-3 helpers live in :mod:`._report_style` (the package-local
SSOT — ``src/`` may not import ``scripts/sim2rank/style.py``). Figures are
built head-less (``Agg`` backend) so no DISPLAY is required.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from . import _report_style as rs
from .normalization import direction_aware_minmax
from .pipeline import MetaEvaluationOutput

MATPLOTLIB_AVAILABLE = rs.MATPLOTLIB_AVAILABLE

logger = logging.getLogger(__name__)


class FigureRenderError(RuntimeError):
    """A figure in the suite raised while rendering.

    Raised by :func:`render_all` in strict mode. Distinct from a renderer that
    returns ``None`` because its diagnostics are absent -- that is an expected,
    benign skip. This is a bug, and a silently-missing figure is one nobody looks
    for.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_method(output: MetaEvaluationOutput, name: str):
    for r in output.per_method:
        if r.method == name:
            return r
    return None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# Re-exported so external callers / tests keep a stable name. The logic now
# lives in the package-local style SSOT.
_top_bottom_indices = rs.top_bottom_indices

# Palette aliases (kept for readability inside this module).
_DIM_GREY = rs.DIM_GREY
_ACCENT = rs.ACCENT
_BAR_COLOR = rs.BAR


# ---------------------------------------------------------------------------
# 1. CDSCR — per-metric descriptor sensitivity
# ---------------------------------------------------------------------------


def spectral_shape_grid(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    """Bar chart of CDSCR partial-correlation per metric.

    The diagnostic dict on ``CDSCRRanker`` stores a single scalar per metric
    (``scores``) and the descriptor-PCA dimensionality. With those two we
    plot a horizontal bar chart sorted descending — readers immediately see
    which metrics respond to *residual shape* once energy and content are
    partialed out.
    """
    cdscr = _find_method(output, "CDSCR")
    if cdscr is None or not cdscr.scores:
        return None
    items = sorted(cdscr.scores.items(), key=lambda kv: kv[1])
    names = [rs.human_label(k) for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = rs.figure(figsize=rs.barh_figsize(len(names)))
    bars = ax.barh(names, vals, color=_BAR_COLOR)
    top, bottom = rs.top_bottom_indices(vals, k=3)
    top_set, bottom_set = set(top), set(bottom)
    for i, b in enumerate(bars):
        if i in top_set:
            b.set_color(_ACCENT)
        elif i in bottom_set:
            b.set_color(_BAR_COLOR)
        else:
            b.set_color(_DIM_GREY)
            b.set_alpha(0.7)
    rs.annotate_top_bottom(ax, vals, k=3, orientation="h")
    ax.set_xlabel("Partial corr. R(M_k; ζ(r) | ‖r‖, C)")
    ax.set_title(
        f"CDSCR — spectral-shape sensitivity\n"
        f"({cdscr.diagnostics.get('descriptor_dim', '?')}-D descriptor, "
        f"{cdscr.diagnostics.get('n_descriptor_pcs', '?')} PCs)"
    )
    ax.set_xlim(0, max(vals + [1.0]) * 1.05)
    ax.grid(axis="x", alpha=0.3)
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig1_cdscr_spectral_shape.png")


# ---------------------------------------------------------------------------
# 2. LGDR — Spearman ρ vs. diffusion time per metric
# ---------------------------------------------------------------------------


def lgdr_eigenscale_curve(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    """Per-metric Spearman ρ vs. diffusion time.

    Renders the bulk of metrics as dim grey lines (alpha=0.3), then
    overdraws the top-3 (accent) and bottom-3 (bar-color) by their
    ρ at the largest diffusion time. Two dashed horizontal lines
    mark the top-3 and bottom-3 boundaries. The legend has at most
    6 named metrics + the boundary lines + 1 "other (n=N)" handle,
    instead of 86 stacked labels.
    """
    from matplotlib.lines import Line2D

    lgdr = _find_method(output, "LGDR")
    if lgdr is None or not lgdr.diagnostics.get("per_t_curves"):
        return None
    curves = lgdr.diagnostics["per_t_curves"]
    times = sorted(lgdr.diagnostics["diffusion_times"])
    if not times:
        return None

    names = list(curves.keys())

    def _y_for(name: str) -> list[float]:
        c = curves[name]
        return [c.get(t, c.get(str(t), 0.0)) for t in times]

    final_vals = [_y_for(n)[-1] for n in names]
    top, bottom = rs.top_bottom_indices(final_vals, k=3)
    top_set, bottom_set = set(top), set(bottom)
    extreme = top_set | bottom_set

    fig, ax = rs.figure(figsize=(7.5, 4.5))

    # 1. Dim bulk first so the highlighted lines draw on top.
    for i, name in enumerate(names):
        if i in extreme:
            continue
        ax.plot(times, _y_for(name), color=_DIM_GREY, alpha=0.30, linewidth=0.8, marker=None)

    # 2. Extremes — bold + markered.
    legend_handles: list = []
    for i in top:
        (h,) = ax.plot(
            times,
            _y_for(names[i]),
            color=_ACCENT,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=rs.human_label(names[i]),
        )
        legend_handles.append(h)
    for i in bottom:
        (h,) = ax.plot(
            times,
            _y_for(names[i]),
            color=_BAR_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=rs.human_label(names[i]),
        )
        legend_handles.append(h)

    # 3. Dashed boundary lines at the top-3 and bottom-3 thresholds.
    if top and bottom:
        top_thr = min(final_vals[i] for i in top)
        bot_thr = max(final_vals[i] for i in bottom)
        h_top = ax.axhline(
            top_thr,
            color=_ACCENT,
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
            label=f"top-3 boundary (ρ = {top_thr:+.2f})",
        )
        h_bot = ax.axhline(
            bot_thr,
            color=_BAR_COLOR,
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
            label=f"bottom-3 boundary (ρ = {bot_thr:+.2f})",
        )
        legend_handles.extend([h_top, h_bot])

    # 4. Single grey legend handle for the dimmed bulk.
    n_other = len(names) - len(extreme)
    if n_other > 0:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=_DIM_GREY,
                alpha=0.6,
                linewidth=0.8,
                label=f"other (n = {n_other})",
            )
        )

    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time t (log scale)")
    ax.set_ylabel("Spearman ρ(M_k, D̃_t)")
    ax.set_title("LGDR — eigenscale curve  (only top-3 / bottom-3 named)")
    ax.axhline(0.0, color="black", linestyle=":", alpha=0.4)
    ax.legend(
        handles=legend_handles,
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        ncol=1,
    )
    ax.grid(alpha=0.3)
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig2_lgdr_eigenscale.png")


# ---------------------------------------------------------------------------
# 3. SFA — score-alignment polar plot
# ---------------------------------------------------------------------------


def sfa_polar(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    sfa = _find_method(output, "SFA")
    if sfa is None or not sfa.scores:
        return None

    names = list(sfa.scores.keys())
    scores = np.array([sfa.scores[n] for n in names], dtype=np.float64)
    # Map score in [-1, 1] to angle in [pi, 0] (so +1 lands at angle 0).
    angles = np.arccos(np.clip(scores, -1.0, 1.0))
    radii = np.abs(scores) + 0.05

    fig, ax = rs.figure(figsize=(6, 6), polar=True)
    cmap = rs.pyplot().cm.tab10
    for i, name in enumerate(names):
        ax.plot(
            [angles[i], angles[i]],
            [0, radii[i]],
            color=cmap(i % 10),
            linewidth=2,
            label=f"{rs.human_label(name)}: {scores[i]:+.2f}",
        )
        ax.plot(angles[i], radii[i], "o", color=cmap(i % 10))
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("E")
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rmax(1.1)
    ax.set_title(
        "SFA — score-direction alignment\n(0° = aligned with manifold; 180° = anti-aligned)"
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.15, 1.0),
        fontsize=7,
        ncol=1 if len(names) <= 18 else 2,
        frameon=True,
    )
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig3_sfa_polar.png")


# ---------------------------------------------------------------------------
# 4. ESD — invariance / sensitivity heatmap
# ---------------------------------------------------------------------------


def esd_fingerprint_matrix(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    esd = _find_method(output, "ESD")
    if esd is None:
        return None
    ig = esd.diagnostics.get("invariance_gap", {})
    sg = esd.diagnostics.get("sensitivity_gain", {})
    if not ig or not sg:
        return None

    metrics = list(ig.keys())
    nuis = sorted({k for d in ig.values() for k in d.keys()})
    pert = sorted({k for d in sg.values() for k in d.keys()})
    columns = nuis + pert
    matrix = np.zeros((len(metrics), len(columns)), dtype=np.float64)
    for i, m in enumerate(metrics):
        for j, col in enumerate(columns):
            if col in nuis:
                # Invert sign so "higher = more invariant" = better.
                v = ig[m].get(col, 0.0)
                matrix[i, j] = -math.log10(v + 1e-6)
            else:
                matrix[i, j] = math.log10(sg[m].get(col, 0.0) + 1e-6)

    fig, ax = rs.figure(figsize=(max(6, 0.7 * len(columns) + 1), max(2.5, 0.4 * len(metrics) + 1)))
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=rs.DIVERGING_CMAP,
        vmin=-np.max(np.abs(matrix)),
        vmax=np.max(np.abs(matrix)),
    )
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        [f"{c}\n({'NUIS' if c in nuis else 'PERT'})" for c in columns],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([rs.human_label(m) for m in metrics])
    ax.axvline(len(nuis) - 0.5, color="black", linewidth=1)
    ax.set_title(
        "ESD — equivariance fingerprint\n(blue = invariant on G_NUIS; red = sensitive on G_PERT)"
    )
    fig.colorbar(im, ax=ax, label="log10 magnitude")
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig4_esd_fingerprint.png")


# ---------------------------------------------------------------------------
# 5. FSPD — FPCA biplot of the metric set
# ---------------------------------------------------------------------------


def fspd_biplot(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    fspd = _find_method(output, "FSPD")
    if fspd is None or "loadings" not in fspd.diagnostics:
        return None
    loadings = fspd.diagnostics["loadings"]
    if not loadings:
        return None

    names = list(loadings.keys())
    pts = np.array([loadings[n] for n in names], dtype=np.float64)
    if pts.shape[1] < 2:
        return None

    fig, ax = rs.figure(figsize=(6.5, 6.5))
    # Top-3 / bottom-3 by mode-1 loading. Mode-1 is the
    # severity-monotone direction, so the labelled points are the most
    # / least monotonic metrics — exactly what the reader needs.
    mode1 = list(pts[:, 0])
    top, bottom = rs.top_bottom_indices(mode1, k=3)
    highlight = set(top) | set(bottom)
    for i, name in enumerate(names):
        if i in top:
            color, alpha, size = _ACCENT, 1.0, 90
        elif i in bottom:
            color, alpha, size = _BAR_COLOR, 1.0, 90
        else:
            color, alpha, size = _DIM_GREY, 0.55, 36
        ax.scatter(
            pts[i, 0],
            pts[i, 1],
            s=size,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=alpha,
        )
        if i in highlight:
            ax.annotate(
                rs.human_label(name),
                (pts[i, 0], pts[i, 1]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold" if i in top else "normal",
            )
    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    ax.axvline(0, color="black", linestyle=":", alpha=0.4)
    ax.set_xlabel("Mode 1 loading (severity-monotone)")
    ax.set_ylabel("Mode 2 loading (artifact-class)")
    ax.set_title(
        "FSPD — FPCA biplot of metric response surfaces  "
        "(only top-3 / bottom-3 mode-1 metrics labelled)"
    )
    ax.grid(alpha=0.3)
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig5_fspd_biplot.png")


# ---------------------------------------------------------------------------
# 6. Sim2Rank — ADR / SCVR / CDRS breakdown
# ---------------------------------------------------------------------------


def sim2rank_breakdown(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    """Sim2Rank ADR / SCVR / CDRS — one figure per sub-axis.

    Previously emitted a 1×3 ``sharey=True`` grid. Splitting per
    sub-axis lets each figure be embedded independently in slides /
    LaTeX without ``y``-axis crowding. The 6a/6b/6c naming preserves
    chronological ordering in ``out_dir``; the first written path
    is returned for backwards-compat with the orchestrator.
    """
    s2r = _find_method(output, "Sim2Rank")
    if s2r is None:
        return None
    adr = s2r.diagnostics.get("adr", {})
    scvr = s2r.diagnostics.get("scvr", {})
    cdrs = s2r.diagnostics.get("cdrs", {})
    if not adr:
        return None
    metrics = list(adr.keys())
    out_dir = _ensure_dir(out_dir)

    panels = [
        ("ADR", adr, "fig6a_sim2rank_adr.png"),
        ("SCVR", scvr, "fig6b_sim2rank_scvr.png"),
        ("CDRS", cdrs, "fig6c_sim2rank_cdrs.png"),
    ]
    written: list[Path] = []
    for title, vals, fname in panels:
        # Sort within each panel so each figure is self-contained and
        # the per-axis ordering reflects that axis's ranking — readers
        # don't have to cross-reference a global metric order.
        ordered = sorted(metrics, key=lambda m: vals.get(m, 0.0))
        bars_v = [vals.get(m, 0.0) for m in ordered]
        labels = [rs.human_label(m) for m in ordered]
        fig, ax = rs.figure(figsize=rs.barh_figsize(len(ordered)))
        bars = ax.barh(labels, bars_v, color=_BAR_COLOR)
        # Top-3 / bottom-3 highlight (bottom keeps the bar color so it is
        # distinguishable from the dimmed bulk).
        top, bottom = rs.top_bottom_indices(bars_v, k=3)
        top_set, bottom_set = set(top), set(bottom)
        for i, b in enumerate(bars):
            if i in top_set:
                b.set_color(_ACCENT)
            elif i in bottom_set:
                b.set_color(_BAR_COLOR)
            else:
                b.set_color(_DIM_GREY)
                b.set_alpha(0.7)
        rs.annotate_top_bottom(ax, bars_v, k=3, orientation="h")
        ax.set_title(f"Sim2Rank — {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("metric")
        ax.grid(axis="x", alpha=0.3)
        written.append(rs.save_figure(fig, out_dir / fname))
    return written[0]


# ---------------------------------------------------------------------------
# 7. Final consensus rank-bump chart
# ---------------------------------------------------------------------------


def consensus_rank_bump(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    z_table = output.aggregate.per_method_z
    if not z_table:
        return None
    methods = list(z_table.keys())
    metrics = sorted(next(iter(z_table.values())).keys())
    labels = [rs.human_label(m) for m in metrics]
    matrix = np.array(
        [[z_table[m].get(metric, 0.0) for metric in metrics] for m in methods],
        dtype=np.float64,
    )

    fig, ax = rs.figure(figsize=(max(7, 0.6 * len(metrics) + 1), max(3, 0.4 * len(methods) + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    # Label only the top-3 / bottom-3 cells PER METHOD (one row at a
    # time). On an 86-metric × 6-method table that's 36 numbers
    # instead of 516 — readable, and still tells you each method's
    # extreme picks.
    for i in range(matrix.shape[0]):
        row = list(matrix[i])
        top, bottom = rs.top_bottom_indices(row, k=3)
        for j in sorted(set(top) | set(bottom)):
            v = matrix[i, j]
            if not math.isfinite(v):
                continue
            ax.text(
                j,
                i,
                f"{v:+.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
                fontweight="bold" if j in set(top) else "normal",
            )
    ax.set_title("Consensus z-score table — top-3 / bottom-3 per method labelled")
    fig.colorbar(im, ax=ax, label="z-score")
    path = rs.save_figure(fig, _ensure_dir(out_dir) / "fig7_consensus_table.png")

    # Bump panel below (separate figure).
    fig2, ax2 = rs.figure(figsize=(max(7, 0.6 * len(metrics) + 1), 3))
    final = output.aggregate.final_scores
    sorted_final = sorted(final.items(), key=lambda kv: -kv[1])
    flag = output.aggregate.defensibility_flags
    bar_vals = [v for _, v in sorted_final]
    ax2.bar(
        range(len(sorted_final)),
        bar_vals,
        color=[rs.DEFENSIBLE_GREEN if flag.get(k, False) else rs.FAIL_RED for k, _ in sorted_final],
    )
    rs.annotate_top_bottom(ax2, bar_vals, k=3, orientation="v")
    ax2.set_xticks(range(len(sorted_final)))
    ax2.set_xticklabels([rs.human_label(k) for k, _ in sorted_final], rotation=45, ha="right")
    ax2.set_ylabel("Final weighted z-score")
    ax2.set_title("Final consensus ranking (green = defensible, red = fails rule)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(0, color="black", linewidth=0.8)
    rs.save_figure(fig2, out_dir / "fig8_consensus_bars.png")
    return path


# ---------------------------------------------------------------------------
# 9. Metric trajectories — severity x metric, one panel per metric, lines per family
# ---------------------------------------------------------------------------


def _per_family_trajectory(
    output: MetaEvaluationOutput, metric_name: str
) -> tuple[list[float], dict[str, list[float]]]:
    """Average a metric across content per (family, severity)."""
    dataset = output.dataset
    severities = sorted({float(s.severity.get("theta", 0.0)) for s in dataset.samples})
    families = sorted({s.degradation_family for s in dataset.samples})
    sev_index = {v: i for i, v in enumerate(severities)}
    values = dataset.metric_values.get(metric_name, [])
    sums = {f: [0.0] * len(severities) for f in families}
    counts = {f: [0] * len(severities) for f in families}
    for s, v in zip(dataset.samples, values, strict=True):
        if not math.isfinite(float(v)):
            continue
        j = sev_index[float(s.severity.get("theta", 0.0))]
        sums[s.degradation_family][j] += float(v)
        counts[s.degradation_family][j] += 1
    out = {
        f: [
            sums[f][j] / counts[f][j] if counts[f][j] > 0 else float("nan")
            for j in range(len(severities))
        ]
        for f in families
    }
    return severities, out


def metric_trajectories(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    """Per-metric trajectories across severity, one line per family.

    Emits **one PNG per metric** under ``out_dir/trajectories/`` plus a
    ``fig9_metric_trajectories_index.png`` **contact sheet** — a grid of
    small severity-curve thumbnails so a reader can scan every metric at a
    glance instead of opening dozens of files. The full-size panels stay in
    ``trajectories/`` for slides / LaTeX.
    """
    dataset = output.dataset
    if dataset.n_samples == 0 or not dataset.metric_values:
        return None
    metric_names = sorted(dataset.metric_values.keys())
    out_dir = _ensure_dir(out_dir)
    traj_dir = _ensure_dir(out_dir / "trajectories")

    cmap = rs.pyplot().cm.tab10
    families_global = sorted({s.degradation_family for s in dataset.samples})
    family_color = {f: cmap(i % 10) for i, f in enumerate(families_global)}

    # Cache each metric's trajectory so the contact sheet doesn't recompute.
    traj_cache: dict[str, tuple[list[float], dict[str, list[float]]]] = {}
    for name in metric_names:
        severities, traj = _per_family_trajectory(output, name)
        traj_cache[name] = (severities, traj)
        fig, ax = rs.figure(figsize=(6.0, 4.0))
        for fam, ys in traj.items():
            ax.plot(
                severities,
                ys,
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=family_color[fam],
                label=fam,
            )
        ax.set_title(f"{rs.human_label(name)} — trajectory across severity")
        ax.set_xlabel("severity θ")
        ax.set_ylabel("metric value")
        ax.grid(alpha=0.3)
        ax.legend(
            fontsize=8,
            loc="best",
            frameon=True,
            ncol=min(len(families_global), 2),
        )
        # Slugify the metric name so we never hit illegal-path chars.
        safe = "".join(c if c.isalnum() or c in "_-." else "_" for c in name)
        rs.save_figure(fig, traj_dir / f"{safe}.png")

    return _trajectory_contact_sheet(
        metric_names,
        traj_cache,
        family_color,
        traj_dir,
        out_dir / "fig9_metric_trajectories_index.png",
    )


def _trajectory_contact_sheet(
    metric_names: list[str],
    traj_cache: dict[str, tuple[list[float], dict[str, list[float]]]],
    family_color: dict,
    traj_dir: Path,
    index_path: Path,
) -> Path:
    """Grid of small per-metric trajectory thumbnails for fast scanning."""
    k = len(metric_names)
    cols = min(4, k) if k else 1
    rows = math.ceil(k / cols) if k else 1
    fig, axes = rs.figure_grid(rows, cols, figsize=(3.0 * cols, 2.1 * rows + 0.6))
    for idx, name in enumerate(metric_names):
        ax = axes[idx // cols][idx % cols]
        severities, traj = traj_cache[name]
        for fam, ys in traj.items():
            ax.plot(severities, ys, linewidth=1.2, color=family_color[fam])
        ax.set_title(rs.human_label(name), fontsize=8)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)
    # Blank any unused grid cells.
    for idx in range(k, rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    # Shared family-color legend so the thumbnail line colors are decodable
    # without opening a full-size panel.
    from matplotlib.lines import Line2D

    handles = [Line2D([0], [0], color=family_color[f], linewidth=2, label=f) for f in family_color]
    # "outside" loc is reserved by constrained layout so the legend never
    # overlaps the bottom row of thumbnails.
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=min(len(handles), 6),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        f"Per-metric severity trajectories — {k} metrics (full-size panels in {traj_dir.name}/)",
        fontsize=11,
    )
    return rs.save_figure(fig, index_path)


# ---------------------------------------------------------------------------
# 10. Metric correlation clustering — Pearson correlation + agglomerative reorder
# ---------------------------------------------------------------------------


def _pearson_corr_matrix(
    values: dict[str, list[float]],
) -> tuple[list[str], np.ndarray]:
    names = sorted(values.keys())
    K = len(names)
    if K == 0:
        return names, np.zeros((0, 0))
    rows = []
    for n in names:
        rows.append(np.asarray(values[n], dtype=np.float64))
    if not rows:
        return names, np.zeros((K, K))
    L = min(len(r) for r in rows)
    X = np.stack([r[:L] for r in rows], axis=0)
    # Pairwise mask of finite values per metric pair.
    corr = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i, K):
            xi, xj = X[i], X[j]
            m = np.isfinite(xi) & np.isfinite(xj)
            if m.sum() < 3:
                # Off-diagonal degenerate → 0; diagonal kept at 1 by the
                # convention below.
                corr[i, j] = corr[j, i] = 0.0
                continue
            ai = xi[m] - xi[m].mean()
            bj = xj[m] - xj[m].mean()
            na = float(np.linalg.norm(ai))
            nb = float(np.linalg.norm(bj))
            if na < 1e-12 or nb < 1e-12:
                corr[i, j] = corr[j, i] = 0.0
            else:
                corr[i, j] = corr[j, i] = float((ai * bj).sum() / (na * nb))
    # Diagonal convention: a metric is perfectly correlated with itself by
    # definition, regardless of whether its variance is zero (a constant
    # metric is constant under any shift, so corr(x, x) = 1 is the natural
    # extension). This avoids confusing 0.0 entries on the heatmap diagonal.
    for i in range(K):
        corr[i, i] = 1.0
    return names, corr


def _single_linkage_order(distance: np.ndarray) -> list[int]:
    """Greedy single-linkage agglomerative ordering, pure numpy.

    Returns a leaf permutation. Adequate for reordering a small correlation
    heatmap (K typically < 20). For larger K, replace with scipy.linkage.
    """
    K = distance.shape[0]
    if K <= 2:
        return list(range(K))
    clusters: list[list[int]] = [[i] for i in range(K)]
    # Build mutable copies; pad diagonal with +inf so it never picks self.
    d = distance.copy()
    np.fill_diagonal(d, np.inf)
    cluster_ids = list(range(K))
    while len(clusters) > 1:
        idx = int(np.argmin(d))
        i, j = idx // d.shape[1], idx % d.shape[1]
        if i > j:
            i, j = j, i
        # Merge cluster j into cluster i.
        new_cluster = clusters[i] + clusters[j]
        # Recompute single-linkage distances from new cluster to all others.
        new_row = np.minimum(d[i], d[j])
        new_row[i] = np.inf
        d[i] = new_row
        d[:, i] = new_row
        # Drop row/col j.
        d = np.delete(d, j, axis=0)
        d = np.delete(d, j, axis=1)
        clusters[i] = new_cluster
        clusters.pop(j)
        cluster_ids.pop(j)
    return clusters[0]


def metric_correlation_clustering(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    """Clustered Pearson-correlation heatmap of the metric cohort.

    Reorders both axes by single-linkage agglomerative clustering on
    ``1 − |corr|`` distance. Diagonal blocks are clusters of metrics that
    respond identically across the simulator's samples — i.e. redundant.
    """
    if not output.dataset.metric_values:
        return None
    names, corr = _pearson_corr_matrix(output.dataset.metric_values)
    if corr.size == 0:
        return None
    distance = 1.0 - np.abs(corr)
    order = _single_linkage_order(distance)
    reord = corr[np.ix_(order, order)]
    reord_names = [rs.human_label(names[i]) for i in order]

    fig, ax = rs.figure(figsize=(max(4, 0.45 * len(names) + 2), max(4, 0.45 * len(names) + 2)))
    im = ax.imshow(reord, cmap=rs.DIVERGING_CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(reord_names)))
    ax.set_xticklabels(reord_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(reord_names)))
    ax.set_yticklabels(reord_names, fontsize=9)
    for i in range(len(reord_names)):
        for j in range(len(reord_names)):
            ax.text(
                j,
                i,
                f"{reord[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(reord[i, j]) > 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(
        "Metric correlation (single-linkage reordered)\nblocks of red = redundant clusters"
    )
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig10_metric_correlation_clustering.png")


# ---------------------------------------------------------------------------
# 11. Sensitivity scatter — ADR vs. SCVR (z-scored), bubble = defensible
# ---------------------------------------------------------------------------


def sensitivity_scatter(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    """Plot each metric in the (ADR, SCVR) plane.

    These are the two principal trajectory-quality axes from Sim2Rank.
    ADR is monotonicity+non-saturation; SCVR is severity-vs-anatomy
    variance ratio. The plane is the cleanest 2-D summary of "does this
    metric track severity well, and is the signal stronger than the
    anatomy nuisance?".
    """
    s2r = _find_method(output, "Sim2Rank")
    if s2r is None:
        return None
    adr = s2r.diagnostics.get("adr", {})
    scvr = s2r.diagnostics.get("scvr", {})
    if not adr or not scvr:
        return None
    names = sorted(set(adr.keys()) & set(scvr.keys()))
    if not names:
        return None
    xs = np.array([adr[n] for n in names], dtype=np.float64)
    ys = np.array([scvr[n] for n in names], dtype=np.float64)
    # Log10-compress SCVR — its tail is unbounded.
    ys_log = np.log10(np.clip(ys, 1e-6, None) + 1.0)

    defensibility = output.aggregate.defensibility_flags or {}
    colors = [rs.DEFENSIBLE_GREEN if defensibility.get(n, False) else rs.FAIL_RED for n in names]

    fig, ax = rs.figure(figsize=(7.5, 6))
    ax.scatter(xs, ys_log, s=120, c=colors, edgecolor="black", linewidth=0.7)
    for x, y, n in zip(xs, ys_log, names, strict=True):
        ax.annotate(
            rs.human_label(n),
            (x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("ADR (monotonicity × non-saturation)")
    ax.set_ylabel("log10(1 + SCVR)\n(severity-vs-anatomy variance ratio)")
    ax.set_title(
        "Sensitivity scatter — Sim2Rank ADR vs. SCVR\n"
        "green = defensible (all six axes), red = fails the asymmetric rule"
    )
    ax.grid(alpha=0.3)
    if xs.size:
        ax.set_xlim(min(0.0, xs.min() - 0.05), max(1.0, xs.max() + 0.05))
    ax.axhline(0, color="black", linestyle=":", alpha=0.4)
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig11_sensitivity_scatter.png")


# ---------------------------------------------------------------------------
# 12. Degradation mosaic — one reference, family × severity grid
# ---------------------------------------------------------------------------


def _as_2d(t):
    """Best-effort projection of a sample tensor to a 2-D magnitude image."""
    import torch as _torch

    if _torch.is_complex(t):
        t = t.abs()
    while t.ndim > 2:
        t = t[0] if t.shape[0] == 1 else t.mean(dim=0)
    return t.detach().float().cpu().numpy()


def degradation_mosaic(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    """Grid of degraded samples (family × severity) for one clean reference.

    Picks the first content_id in the dataset and lays out a row per
    degradation family across four severity levels (interpolated from
    the actual simulator grid). The clean reference is shown in the
    leftmost column. Sanity check that the simulator is doing what its
    docstrings claim.
    """
    dataset = output.dataset
    if dataset.n_samples == 0:
        return None
    samples = dataset.samples
    target_id = samples[0].content_id
    families = sorted({s.degradation_family for s in samples if s.content_id == target_id})
    severities = sorted(
        {float(s.severity.get("theta", 0.0)) for s in samples if s.content_id == target_id}
    )
    if not families or not severities:
        return None
    # Pick four representative severities — quartile positions of the grid.
    if len(severities) >= 4:
        picks = [severities[int(round(q * (len(severities) - 1)))] for q in (0.25, 0.5, 0.75, 1.0)]
    else:
        picks = severities
    picks = sorted(set(picks))

    by_key: dict[tuple[str, float], object] = {}
    clean_ref = None
    for s in samples:
        if s.content_id != target_id:
            continue
        if clean_ref is None:
            clean_ref = s.clean
        theta = float(s.severity.get("theta", 0.0))
        if theta in picks:
            by_key[(s.degradation_family, theta)] = s.degraded

    cols = 1 + len(picks)
    rows = len(families)
    fig, axes = rs.figure_grid(rows, cols, figsize=(2.2 * cols, 2.0 * rows + 0.5))
    clean_np = _as_2d(clean_ref) if clean_ref is not None else None
    # One shared display range for every panel, derived robustly from the clean
    # reference. Both halves matter: letting each panel autoscale independently
    # makes the θ progression incomparable (each gets its own contrast), and
    # autoscaling to .max() on p99-normalised data lets the sparse bright tail
    # crush the brain to black. Percentile clipping fixes both.
    if clean_np is not None:
        vmin = float(np.percentile(clean_np, 1.0))
        vmax = float(np.percentile(clean_np, 99.5))
        if vmax <= vmin:  # degenerate reference — fall back to autoscale
            vmin, vmax = None, None
    else:
        vmin, vmax = None, None

    for r, fam in enumerate(families):
        ax = axes[r][0]
        if clean_np is not None:
            ax.imshow(clean_np, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{fam}\nclean (θ=0)", fontsize=8)
        ax.axis("off")
        for c, theta in enumerate(picks):
            ax = axes[r][c + 1]
            t = by_key.get((fam, theta))
            if t is None:
                ax.axis("off")
                continue
            ax.imshow(_as_2d(t), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"θ={theta:.2f}", fontsize=8)
            ax.axis("off")

    fig.suptitle(f"Degradation mosaic — reference '{target_id}'")
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig12_degradation_mosaic.png")


# ---------------------------------------------------------------------------
# 13. Normalized consensus leaderboard — the headline figure
# ---------------------------------------------------------------------------


def normalized_leaderboard(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    """Headline leaderboard: consensus score min-maxed to ``[0, 1]``.

    The final consensus z-scores live on an unbounded axis that is hard to
    read at a glance. This figure direction-aware min-max-normalizes them so
    ``1.0`` = the best metric in the cohort, ``0.0`` = the worst — the single
    chart to drop in a slide. Bars are colored by the defensibility rule and
    rank-ordered top-to-bottom.
    """
    final = output.aggregate.final_scores
    if not final:
        return None
    norm = direction_aware_minmax(final, higher_is_better=None)  # z: larger=better
    flags = output.aggregate.defensibility_flags or {}
    # Best at the top of a horizontal bar chart → ascending so barh stacks
    # the largest last (top).
    ordered = sorted(norm.items(), key=lambda kv: (math.isnan(kv[1]), kv[1]))
    names = [rs.human_label(k) for k, _ in ordered]
    vals = [0.0 if math.isnan(v) else v for _, v in ordered]
    colors = [rs.DEFENSIBLE_GREEN if flags.get(k, False) else rs.FAIL_RED for k, _ in ordered]

    fig, ax = rs.figure(figsize=rs.barh_figsize(len(names)))
    ax.barh(names, vals, color=colors, edgecolor="black", linewidth=0.4)
    rs.annotate_top_bottom(ax, vals, k=3, orientation="h")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Normalized consensus score  (1.0 = best in cohort, direction-aware min-max)")
    ax.set_title(
        "Metric leaderboard — normalized consensus score\n"
        "green = defensible (passes the six-axis rule), red = fails"
    )
    ax.grid(axis="x", alpha=0.3)
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig13_normalized_leaderboard.png")


# ---------------------------------------------------------------------------
# 14. Severity-response heatmap — metric × degradation-family Spearman ρ
# ---------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ via average-rank Pearson (NaN pairs already dropped)."""
    if x.size < 3:
        return float("nan")

    def _ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(a.size, dtype=np.float64)
        ranks[order] = np.arange(a.size, dtype=np.float64)
        # Average ties.
        sa = a[order]
        i = 0
        while i < a.size:
            j = i
            while j + 1 < a.size and sa[j + 1] == sa[i]:
                j += 1
            if j > i:
                ranks[order[i : j + 1]] = (i + j) / 2.0
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    nx, ny = np.linalg.norm(rx), np.linalg.norm(ry)
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    return float((rx * ry).sum() / (nx * ny))


def _family_severity_matrix(
    output: MetaEvaluationOutput,
) -> tuple[list[str], list[str], np.ndarray]:
    """Spearman ρ(value, θ) for every (metric, degradation-family) pair."""
    dataset = output.dataset
    metric_names = sorted(dataset.metric_values.keys())
    families = sorted({s.degradation_family for s in dataset.samples})
    fam_index = {f: i for i, f in enumerate(families)}
    thetas = np.array(
        [float(s.severity.get("theta", 0.0)) for s in dataset.samples],
        dtype=np.float64,
    )
    fam_of = [s.degradation_family for s in dataset.samples]
    matrix = np.full((len(metric_names), len(families)), np.nan, dtype=np.float64)
    for mi, name in enumerate(metric_names):
        vals = np.asarray(dataset.metric_values.get(name, []), dtype=np.float64)
        if vals.size != thetas.size:
            continue
        for fam in families:
            mask = np.array([f == fam for f in fam_of]) & np.isfinite(vals)
            if mask.sum() < 3:
                continue
            matrix[mi, fam_index[fam]] = _spearman(thetas[mask], vals[mask])
    return metric_names, families, matrix


def severity_response_heatmap(output: MetaEvaluationOutput, out_dir: Path) -> Path | None:
    """Heatmap of Spearman ρ(metric, θ) per degradation family.

    Each cell answers "does this metric move monotonically with severity on
    this artifact family?". A near-zero (white) or wrong-signed row exposes a
    blind spot — a family the metric cannot track — which the aggregate
    consensus score alone hides.
    """
    metric_names, families, matrix = _family_severity_matrix(output)
    if not metric_names or not families or not np.isfinite(matrix).any():
        return None
    labels = [rs.human_label(m) for m in metric_names]

    fig, ax = rs.figure(
        figsize=(
            max(7.5, 0.8 * len(families) + 3.5),
            max(3, 0.4 * len(metric_names) + 1.5),
        )
    )
    im = ax.imshow(matrix, aspect="auto", cmap=rs.DIVERGING_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels(labels, fontsize=8)
    # Annotate cells when the grid is small enough to stay legible.
    if len(metric_names) * len(families) <= 240:
        for i in range(len(metric_names)):
            for j in range(len(families)):
                v = matrix[i, j]
                if not math.isfinite(v):
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if abs(v) > 0.6 else "black",
                )
    fig.colorbar(im, ax=ax, label="Spearman ρ(value, θ)")
    ax.set_title("Severity response per family\n(white / wrong-signed row = blind spot)")
    return rs.save_figure(fig, _ensure_dir(out_dir) / "fig14_severity_response_heatmap.png")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def render_all(output: MetaEvaluationOutput, out_dir: Path, *, strict: bool = True) -> list[Path]:
    """Render every figure that has the data it needs.

    Each individual renderer returns ``None`` when its source diagnostics
    are missing — that is the expected outcome when a user omitted a
    ranker from the pipeline, and it is *not* a failure. The list is the
    figures actually written.

    A renderer that *raises*, by contrast, is a bug. Those were previously
    logged at WARNING and skipped, so a figure could disappear from the report
    while the sweep still exited zero.

    Args:
        strict: raise if any renderer raised (the default). ``strict=False``
            collects the whole failure list in one pass instead of stopping at
            the first; failures are logged at ERROR either way.
    """
    out_dir = _ensure_dir(Path(out_dir))
    written: list[Path] = []
    failures: list[tuple[str, Exception]] = []
    for fn in (
        spectral_shape_grid,
        lgdr_eigenscale_curve,
        sfa_polar,
        esd_fingerprint_matrix,
        fspd_biplot,
        sim2rank_breakdown,
        consensus_rank_bump,
        metric_trajectories,
        metric_correlation_clustering,
        sensitivity_scatter,
        degradation_mosaic,
        normalized_leaderboard,
        severity_response_heatmap,
    ):
        try:
            path = fn(output, out_dir)
        except Exception as exc:  # reported, not swallowed
            logger.error("figure %s failed: %s: %s", fn.__name__, type(exc).__name__, exc)
            failures.append((fn.__name__, exc))
            continue
        if path is not None:
            written.append(path)

    if failures and strict:
        detail = "\n".join(f"  {n}: {type(e).__name__}: {e}" for n, e in failures)
        raise FigureRenderError(
            f"{len(failures)} figure(s) failed to render into {out_dir}:\n{detail}"
        ) from failures[0][1]
    return written
