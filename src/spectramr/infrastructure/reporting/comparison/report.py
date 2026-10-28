r"""Cross-run report: the figures and tables that compare arms, in one directory.

``generate_comparison(runs, out_dir)`` reads every arm at the same checkpoint
rule (:mod:`..views`) and renders through the same registry, style and manifest
as the per-run report, so a comparison figure and a per-run figure of the same
run never disagree about which pass they show.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .. import plotters, tables
from spectramr.core.run_artifacts import read_parity

from ..aggregator import eval_source
from ..metadata import detect_metadata, run_provenance
from ..plotters.distributions import SLICE_FIGURES
from ..run_status import status_notes
from ..style import set_panel_labels, use_default_style
from .all_metric_figures import cohort_figures, curve_names, more_curves, sweeps, table_metrics
from .arms import load_arms, resolve_runs
from .metric_selection import PHASES, check_phases, discover_metrics, select_metrics, union
from .phase_tables import latex_table, phase_table, wide

logger = logging.getLogger(__name__)

#: Figure ids rendered by a comparison, in report order.
COMPARISON_FIGURES: tuple[str, ...] = (
    "cmp_metric_bars",
    "fig_1_2_learning_curves",
    "acceleration_sweep",
    "cmp_mosaic",
    "fig_3_1_metric_violin",
    "fig_3_2_subject_bars",
    "fig_3_3_metric_scatter",
    "cmp_paired_slice_scatter",
    "cmp_rank_heatmap",
    "fig_3_6_error_decomposition",
    "fig_3_7_metric_embedding",
    "fig_4_1_weight_trajectory",
    "fig_4_2_layer_drift",
    "fig_4_4_image_embedding",
)
COMPARISON_TABLES: tuple[str, ...] = ("tab_2_1_main_results",)


def _with_selection_notes(arms) -> dict[str, str]:
    """``arms.notes`` plus a note for each arm selected at its first validation pass."""
    from ..selection_note import first_pass_selection_notes

    merged = dict(first_pass_selection_notes(arms.run_dirs))
    for arm, note in arms.notes.items():
        merged[arm] = f"{note}; {merged[arm]}" if arm in merged else note
    return merged


_SOURCE_LABELS = {
    "inference": "inference/ re-run (`campaign infer`)",
    "training": "training-time test pass",
    "none": "none",
}


def _source_cell(run_dir) -> str:
    source = eval_source(run_dir)
    if source != "inference":
        return _SOURCE_LABELS[source]
    return f"{_SOURCE_LABELS[source]}, parity {read_parity(run_dir)}"


def _status_label(arms, lab: str) -> str:
    status = arms.status.get(lab)
    return status.label if status is not None else "not read"


def _summary(path: Path, arms, figures, reasons, tables_, view: str) -> None:
    lines = [f"# Comparison — {len(arms.labels)} arms", "", f"- Checkpoint view: `{view}`"]
    lines += ["", "| arm | run directory | step | status | inference source |", "|---|---|---|---|---|"]
    lines += [
        f"| {lab} | `{arms.run_dirs[lab]}` | {arms.steps.get(lab, '—')} | {_status_label(arms, lab)} "
        f"| {_source_cell(arms.run_dirs[lab])} |"
        for lab in arms.labels
    ]
    if arms.excluded:
        lines += ["", "## Excluded arms", "", "Left out of every table and figure:", ""]
        lines += [
            f"- **{lab}** (`{arms.excluded_dirs[lab]}`): {s.label}"
            + (f" — {s.error_line}" if s.error_line else "")
            for lab, s in arms.excluded.items()
        ]
    lines += ["", "## Figures"]
    for fid, p in figures.items():
        if p is None:
            lines.append(f"- **{fid}**: _skipped ({reasons.get(fid, 'no data')})_")
        else:
            rel = p.relative_to(path.parent)
            lines += [
                f"- **{fid}**: [{p.name}]({rel})",
                "",
                f"  ![{fid}]({rel.with_suffix('.png')})",
                "",
            ]
    lines += ["", "## Tables"]
    for tid, group in tables_.items():
        links = ", ".join(
            f"[{p.name}]({p.relative_to(path.parent)})" for p in (group or {}).values()
        )
        lines.append(f"- **{tid}**: {links or '_skipped_'}")
    path.write_text("\n".join(lines) + "\n")


def generate_comparison(
    runs: list[str | Path],
    out_dir: str | Path,
    *,
    labels: list[str] | None = None,
    baseline: str | None = None,
    view: str = "selected",
    style: str = "nature",
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    panel_labels: bool = True,
    html_report: bool = True,
    metrics: str | list[str] | None = None,
    phases: tuple[str, ...] = PHASES,
    max_panels: int = 12,
    train_window: int = 50,
    stats: Any = None,
    ladder: Any = None,
    groups: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Render the comparison of ``runs`` (run dirs or roots holding them) into ``out_dir``.

    ``baseline`` names the arm the others are tested against (a label after
    shortening). Returns ``{"figures", "tables", "out_dir", "summary", "html"}``.

    ``metrics`` is ``None`` for the headline panels, ``"all"`` for every metric any
    phase logged, or a list of base names. ``phases`` limits which of ``val`` /
    ``train`` / ``test`` (inference) are read. The long table of every metric by
    arm and phase is always written to ``data/``.
    """
    run_dirs = resolve_runs(list(runs))
    arms = load_arms(run_dirs, labels=labels, view=view)
    if baseline is not None and baseline in arms.excluded:
        raise ValueError(
            f"baseline {baseline!r} is excluded ({arms.excluded[baseline].label}); "
            f"pick a finished arm from {arms.labels}"
        )
    if baseline is not None and baseline not in arms.labels:
        raise ValueError(f"baseline {baseline!r} is not an arm; arms are {arms.labels}")
    out_dir = Path(out_dir)
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    phases = check_phases(phases)
    found = discover_metrics(arms, phases)
    chosen = select_metrics(metrics, found)
    use_default_style(style)
    set_panel_labels(panel_labels)
    datasets = {run_provenance(d)["dataset_version"] for d in arms.run_dirs.values()}
    metadata = detect_metadata(dataset_version="; ".join(sorted(d for d in datasets if d)) or None)
    common: dict[str, Any] = {"arms": arms, "baseline": baseline, "metadata": metadata}
    per_fig: dict[str, dict[str, Any]] = {
        "acceleration_sweep": {
            "per_case_df": arms.per_case,
            "view": view,
            "call_unit": arms.call_unit,
            "expected_methods": arms.labels,
        },
    }
    for fid in SLICE_FIGURES:
        per_fig.setdefault(fid, {})["slice_df"] = arms.per_case
    for fid in ("fig_4_1_weight_trajectory", "fig_4_2_layer_drift"):
        per_fig.setdefault(fid, {})["weight_runs"] = arms.run_dirs
    per_fig.setdefault("fig_4_4_image_embedding", {})["arm_cases"] = arms.cases
    wanted = union(chosen) if chosen is not None else None
    if wanted is not None:
        per_fig["cmp_metric_bars"] = {"metrics": wanted, "max_panels": max_panels}
        curves = curve_names(arms, wanted)
        per_fig["fig_1_2_learning_curves"] = {"metrics": curves[:max_panels]}
    figures: dict[str, Path | None] = {}
    reasons: dict[str, str] = {}
    for fid in COMPARISON_FIGURES:
        out = plotters.dispatch_detailed(
            arms.df, [fid], fig_dir, formats=formats, dpi=dpi, **common, **per_fig.get(fid, {})
        )[fid]
        figures[fid] = out.path
        if out.reason:
            reasons[fid] = out.reason
    for extra in sorted(fig_dir.glob(f"cmp_metric_bars_*.{formats[0]}")):
        figures[extra.stem] = extra
    if wanted is not None:
        more_curves(arms, curves[max_panels:], fig_dir, formats, dpi, max_panels, metadata, figures)
    table = phase_table(arms, discover_metrics(arms, phases), train_window=train_window)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "data" / "metrics_long.csv", index=False)
    for phase in phases:
        wide(table, phase).to_csv(out_dir / "data" / f"metrics_{phase}.csv", index=False)
    if wanted is not None:
        tab_dir.mkdir(parents=True, exist_ok=True)
        for phase in phases:
            shown_phase = table[(table["phase"] == phase) & table["metric"].isin(wanted)]
            if not shown_phase.empty:
                (tab_dir / f"metrics_{phase}.tex").write_text(
                    latex_table(shown_phase, phase) + "\n"
                )
        sweeps(arms, wanted, fig_dir, formats, dpi, view, metadata, figures, reasons)
        shown = table[table["metric"].isin(wanted) & table["phase"].isin(phases)]
        outcome = plotters.dispatch_detailed(
            arms.df,
            ["cmp_phase_metrics"],
            fig_dir,
            formats=formats,
            dpi=dpi,
            phase_df=shown,
            arms=arms.labels,
            max_panels=max_panels,
            metadata=metadata,
        )["cmp_phase_metrics"]
        figures["cmp_phase_metrics"] = outcome.path
        if outcome.reason:
            reasons["cmp_phase_metrics"] = outcome.reason
        for extra in sorted(fig_dir.glob(f"cmp_phase_metrics_*.{formats[0]}")):
            figures[extra.stem] = extra
        more, why = cohort_figures(
            arms,
            shown,
            stats,
            baseline,
            ladder,
            phases,
            fig_dir,
            formats,
            dpi,
            max_panels,
            metadata,
            groups,
        )
        figures.update(more)
        reasons.update(why)
    tables_out: dict[str, Any] = {}
    for tid in COMPARISON_TABLES:
        tables_out.update(
            tables.dispatch(
                arms.df,
                [tid],
                tab_dir,
                baseline_method=baseline,
                view=view,
                run_notes=status_notes(arms.status),
                excluded_runs={lab: s.label for lab, s in arms.excluded.items()},
                arm_notes=_with_selection_notes(arms),
                **({"metrics": [m for m in wanted if m in table_metrics(arms)]} if wanted else {}),
            )
        )
    summary = out_dir / "comparison_summary.md"
    _summary(summary, arms, figures, reasons, tables_out, view)
    from ..report_manifest import write_report_manifest

    runs = [
        {"arm": lab, "run_dir": str(arms.run_dirs[lab]), "included": True, **s.to_dict()}
        for lab, s in arms.status.items()
    ] + [
        {"arm": lab, "run_dir": str(arms.excluded_dirs[lab]), "included": False, **s.to_dict()}
        for lab, s in arms.excluded.items()
    ]
    write_report_manifest(
        out_dir,
        figures=figures,
        tables=tables_out,
        metadata=metadata,
        figure_reasons=reasons,
        runs=runs,
    )
    html = None
    if html_report:
        try:
            from ..html_report import build_html_report

            html = build_html_report(
                out_dir,
                figures=figures,
                tables=tables_out,
                metadata=metadata,
                aggregated_df=arms.df,
                interactive=False,
                method="comparison",
                task=f"{len(arms.labels)} arms, {view} view",
                filename="comparison.html",
            )
        except Exception as exc:  # the figures are the deliverable; HTML is a view of them
            logger.warning("comparison html failed: %s", exc)
    return {
        "figures": figures,
        "tables": tables_out,
        "out_dir": out_dir,
        "summary": summary,
        "html": html,
        "arms": arms.labels,
    }
