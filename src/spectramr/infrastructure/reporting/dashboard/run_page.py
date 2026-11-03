r"""The experiment (single-run) page: the per-run QC report.

Interactive group plots, per-subject viewers and the remaining figures are laid out
as sections of the shared dashboard page (:mod:`.dashboard`). Figures are linked from
``figures/`` rather than base64-embedded, and plotly.js is written once to
``_assets/plotly.min.js``, so the page stays small and still renders with no network.
Interactive sections need :mod:`plotly` (``pip install spectramr[viz]``) and fall back
to the static PNG without it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata

from .shell import Section, esc, figure_html, gallery, rel, render_page, write_plotly_asset

logger = logging.getLogger(__name__)

# QC figures are surfaced first, each with a human caption.
_FEATURED = [
    ("qc_group_strip", "Group IQM distribution"),
    ("qc_subject_mosaic", "Per-subject QC mosaic"),
    ("qc_carpet", "Carpet + spike diagnostics"),
]


def _slug(title: str) -> str:
    return re.sub(r"\W+", "-", title.lower()).strip("-")


def _static_fallback(
    title: str, fig_id: str, figures: dict, page: Path, *, note: bool = True
) -> Section | None:
    """A figure's static tile as the fallback for an interactive section."""
    p = figures.get(fig_id)
    tile = figure_html(p, page, caption=title, wide=True) if p is not None else None
    if not tile:
        return None
    hint = (
        (
            "<p class='meta'>Install <code>plotly</code> (<code>pip install spectramr[viz]</code>) "
            "for the interactive version.</p>"
        )
        if note
        else ""
    )
    return Section(_slug(title), title, gallery([tile]) + hint)


def _provenance(metadata: RunMetadata | None, task: str) -> dict[str, object]:
    prov: dict[str, object] = {"Task": task}
    if metadata is not None:
        prov.update(
            {
                "Commit": metadata.git_commit[:10] + ("*" if metadata.git_dirty else ""),
                "Seed": metadata.seed,
                "Data": metadata.dataset_version,
                "Generated": str(metadata.timestamp_utc)[:16],
            }
        )
    return prov


def _tables_section(tables: dict | None, page: Path) -> Section | None:
    if not tables:
        return None
    rows = []
    for tid, group in tables.items():
        if not group:
            continue
        links = " ".join(
            f"<a href='{esc(rel(p, page))}'>{esc(Path(p).name)}</a>" for p in group.values()
        )
        rows.append(f"<li><b>{esc(tid)}</b>: {links}</li>")
    return Section("tables", "Tables", "<ul>" + "".join(rows) + "</ul>") if rows else None


def _interactive_sections(
    *,
    per_call_df,
    predictions_df,
    aggregated_df,
    cases,
    figures,
    shown,
    page,
) -> tuple[list[Section], bool]:
    """Build the plotly interactive sections (with static-PNG fallbacks).

    Returns ``(sections, used_plotly)``; each entry is a page :class:`Section`;
    ``used_plotly`` is True when at least one genuinely interactive div was
    emitted (so the assembler injects plotly.js once).
    """
    from .. import interactive as itv

    sections: list[Section] = []
    used = False

    def add(title: str, div: str | None, fallback_fid: str | None) -> None:
        nonlocal used
        if div is not None:
            sections.append(Section(_slug(title), title, div))
            used = True
            if fallback_fid:
                shown.add(fallback_fid)
            return
        if fallback_fid:
            # The install hint only when plotly is missing; otherwise the data was.
            fb = _static_fallback(title, fallback_fid, figures, page, note=not itv.has_plotly())
            if fb is not None:
                sections.append(fb)
                shown.add(fallback_fid)

    add(
        "Group IQM distribution",
        itv.group_iqm_div(per_call_df=per_call_df, predictions_df=predictions_df),
        "qc_group_strip",
    )
    add("Per-subject viewer (2-D)", itv.subject_viewer_2d_div(cases), "qc_subject_mosaic")
    add("Volume viewer (3-D)", itv.volume_viewer_3d_div(cases), None)
    add("Training dynamics", itv.learning_curve_div(aggregated_df), "fig_1_2_learning_curves")
    add(
        "Per-case metric distribution",
        itv.metric_distribution_div(per_call_df=per_call_df, predictions_df=predictions_df),
        "metric_distribution",
    )
    add(
        "Method agreement (Bland-Altman)",
        itv.comparison_div(predictions_df=predictions_df),
        "bland_altman",
    )
    return sections, used


def build_html_report(
    out_dir: str | Path,
    *,
    figures: dict,
    tables: dict | None = None,
    metadata: RunMetadata | None = None,
    per_call_df: pd.DataFrame | None = None,
    predictions_df: pd.DataFrame | None = None,
    aggregated_df: pd.DataFrame | None = None,
    cases: list | None = None,
    interactive: bool = True,
    method: str = "report",
    task: str = "default",
    filename: str = "qc_report.html",
) -> Path | None:
    """Write ``<out_dir>/<filename>`` and return its path (None if nothing to show).

    Args:
        figures: ``{fig_id: Path | None}`` from the plotter dispatch.
        tables: ``{table_id: {ext: Path}}`` (optional; rendered as links).
        metadata: run provenance for the header.
        per_call_df: per-case metric frame (interactive group / distribution plots).
        predictions_df: per-case long frame (distribution / Bland-Altman).
        aggregated_df: tidy metric frame (interactive learning curves).
        cases: recorded image cases (2-D + optional ``*_volume`` for the viewers).
        interactive: when True and plotly is installed, emit interactive plotly
            sections; otherwise every section falls back to the embedded PNG.

    Static figures are linked relative to the page, and plotly.js is written once
    beside it, so the report renders offline (no CDN) as long as the run folder
    travels whole.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page = out_dir / filename
    figures = figures or {}
    sections: list[Section] = []
    shown = {"qc_group_strip"}
    used_plotly = False

    if interactive:
        interactive_sections, used_plotly = _interactive_sections(
            per_call_df=per_call_df,
            predictions_df=predictions_df,
            aggregated_df=aggregated_df,
            cases=cases,
            figures=figures,
            shown=shown,
            page=page,
        )
        sections.extend(interactive_sections)
    else:
        fb = _static_fallback("Group IQM distribution", "qc_group_strip", figures, page, note=False)
        if fb is not None:
            sections.append(fb)

    # Featured static QC reportlets not already shown (carpet, and mosaic when
    # the 2-D viewer fell back / was skipped).
    for fid, cap in _FEATURED:
        if fid in shown:
            continue
        p = figures.get(fid)
        tile = figure_html(p, page, caption=cap, wide=True) if p is not None else None
        if tile:
            sections.append(Section(_slug(cap), cap, gallery([tile])))
        shown.add(fid)

    from .figure_index import caption

    rest = gallery(
        [
            figure_html(p, page, caption=caption(fid))
            for fid, p in figures.items()
            if fid not in shown and p is not None
        ]
    )
    if rest:
        sections.append(Section("figures", "Figures", rest))

    tables_section = _tables_section(tables, page)
    if tables_section is not None:
        sections.append(tables_section)

    if not sections:
        logger.info("html report: no figures to embed; skipping")
        return None

    plotly_js = write_plotly_asset(out_dir / "_assets") if used_plotly else None
    render_page(
        page,
        title=f"QC report: {method}",
        sections=sections,
        provenance=_provenance(metadata, task),
        crumbs=[("Run " + method, None)],
        plotly_js=plotly_js,
    )
    logger.info("html report: wrote %s", page)
    return page
