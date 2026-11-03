"""The campaign dashboard: one ``index.html`` over a campaign tree, plus its sub-pages.

Layout it reads (all optional, each section appears when its input exists)::

    <root>/campaign_state.json          runs, roles, status, wall time
    <root>/comparison*/data/...         written by ``report --compare``
    <root>/contrasts/<name>/...         written by ``report --contrasts`` / ``--contrast-pairs``
    <root>/<run>/report/qc_report.html  written by the end-of-run report

Pages are re-rendered from those files; no figure or metric is recomputed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from . import charts
from .catalog import catalog_page_from_disk
from .pages import comparison_page, headline_metrics, metric_sections, pick_phase, read_long
from .runs import run_records, runs_section
from .shell import Section, cards, rel, render_page, write_plotly_asset

logger = logging.getLogger(__name__)

ASSETS = "_assets"


def short_labels(labels: list[str]) -> dict[str, str]:
    """Strip the ``_``-delimited prefix and suffix every label shares (``n2n_a_x_m4raw`` -> ``a_x``)."""
    if len(labels) < 2:
        return {lab: lab for lab in labels}
    parts = [lab.split("_") for lab in labels]
    pre = len(os.path.commonprefix(parts))
    suf = len(os.path.commonprefix([p[::-1] for p in parts]))
    out = {}
    for lab, p in zip(labels, parts, strict=True):
        core = p[pre : len(p) - suf if suf else None]
        out[lab] = "_".join(core) if core else lab
    return out


def _state(root: Path) -> dict:
    p = root / "campaign_state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _baseline_label(state: dict, arms: list[str]) -> str | None:
    names = [e.get("name", "") for e in state.get("experiments", []) if e.get("role") == "baseline"]
    for arm in arms:
        if any(arm == n or arm in n or n in arm for n in names):
            return arm
    return None


def contrast_folders(root: Path) -> list[Path]:
    base = root / "contrasts"
    return sorted(p.parent for p in base.glob("*/contrast.json")) if base.is_dir() else []


def contrasts_section(
    root: Path, page: Path, folders: list[Path], phase: str | None
) -> Section | None:
    if not folders:
        return None
    frames = []
    for f in folders:
        stats = f / "data" / "paired_stats.csv"
        if stats.exists():
            frames.append((pd.read_csv(stats), read_long(f)))
    body = ""
    if frames and phase is not None:
        stats = pd.concat([s for s, _ in frames], ignore_index=True)
        dirs: dict[str, str] = {}
        for _, long_df in frames:
            dirs.update(charts.directions(long_df))
        eff = charts.oriented_effects(stats[stats["phase"] == phase], dirs)
        if not eff.empty:
            short = short_labels(sorted(eff["variant"].unique()))
            eff = eff.assign(label=eff["variant"].map(short))
            heads = [m for m in ("psnr", "ssim", "nmse", "lpips") if m in set(eff["metric"])]
            heads = heads or list(dict.fromkeys(eff["metric"]))[:4]
            div = charts.effect_dots_div(eff, rows="label", metrics=heads)
            if div:
                body += (
                    f"<div class='panel'>{div}</div><p class='howto'><b>How to read this.</b> Each row "
                    "is one contrast against its baseline. Right of zero is better; filled dots survive "
                    "Holm correction at 0.05.</p>"
                )
    items = []
    blocks = [json.loads((f / "contrast.json").read_text()).get("contrast", {}) for f in folders]
    short = short_labels(sorted({a["label"] for c in blocks for a in c.get("arms", [])}))
    for f, c in zip(folders, blocks, strict=True):
        base = c.get("baseline", "")
        others = [
            short.get(a["label"], a["label"]) for a in c.get("arms", []) if a["label"] != base
        ]
        stats = f / "data" / "paired_stats.csv"
        st = pd.read_csv(stats) if stats.exists() else pd.DataFrame()
        n_sig = int((st["p_holm"] < 0.05).sum()) if "p_holm" in st else 0
        items.append(
            (
                rel(f / "comparison.html", page),
                f"{', '.join(others)} vs {short.get(base, base)}",
                f"{n_sig} of {len(st)} paired tests significant after Holm",
            )
        )
    body += "<h3>Every contrast</h3>" + cards(items)
    return Section(
        "contrasts",
        "Contrasts",
        body,
        note=f"{len(folders)} declared contrasts, {phase or 'no'} split.",
    )


def build_site(
    root: str | Path,
    *,
    title: str | None = None,
    batch_rows: list[dict] | None = None,
    extra_pages: list[tuple[Path, str, str]] = (),
) -> dict[str, Path]:
    """Render the campaign dashboard and every comparison/contrast page under ``root``.

    ``batch_rows`` are :func:`~spectramr.infrastructure.reporting.batch.generate_reports` rows;
    they name failed runs, which a campaign without ``campaign_state.json`` cannot.
    ``extra_pages`` are ``(page, title, detail)`` links added under "More pages".
    """
    root = Path(root)
    state = _state(root)
    name = title or state.get("campaign_name") or root.name
    index = root / "index.html"
    plotly_js = write_plotly_asset(root / ASSETS)
    pages: dict[str, Path] = {}
    home = ("Campaign " + name, None)

    comparisons = sorted(p.parent.parent for p in root.glob("comparison*/data/metrics_long.csv"))
    main_long = read_long(comparisons[0]) if comparisons else pd.DataFrame()
    baseline = (
        _baseline_label(state, sorted(main_long["arm"].unique())) if not main_long.empty else None
    )
    for folder in comparisons:
        crumbs = [(home[0], rel(index, folder / "comparison.html")), (folder.name, None)]
        pages[folder.name] = comparison_page(
            folder,
            baseline=baseline,
            title=f"{name}: {folder.name}",
            crumbs=crumbs,
            plotly_js=plotly_js,
        )
    folders = contrast_folders(root)
    for f in folders:
        page = f / "comparison.html"
        crumbs = [
            (home[0], rel(index, page)),
            ("Contrasts", rel(index, page) + "#contrasts"),
            (f.name, None),
        ]
        pages[f"contrast:{f.name}"] = comparison_page(f, crumbs=crumbs, plotly_js=plotly_js)

    sections: list[Section] = []
    phase = pick_phase(main_long) if not main_long.empty else None
    if phase is not None:
        sections += metric_sections(main_long, baseline)
    sec = contrasts_section(root, index, folders, phase)
    if sec:
        sections.append(sec)
    sec = runs_section(index, run_records(root, state, batch_rows))
    if sec:
        sections.append(sec)
    items = [
        (
            rel(c / "comparison.html", index),
            c.name.replace("_", " ").capitalize(),
            f"{read_long(c)['arm'].nunique()} arms, every figure and table",
        )
        for c in comparisons
    ]
    catalog = root / "catalog" / "catalog.html"
    if catalog.exists():
        pages["catalog"] = catalog_page_from_disk(
            catalog.parent, crumbs=[(home[0], rel(index, catalog))]
        )
        items.append(
            (rel(catalog, index), "Figure catalog", "Every figure and table the report can draw")
        )
    items += [(rel(page, index), t, d) for page, t, d in extra_pages]
    if items:
        sections.append(Section("comparisons", "More pages", cards(items)))
    if not sections:
        sections.append(
            Section(
                "empty",
                "Nothing to show yet",
                "<p class='empty'>Run <code>spectramr report --compare</code> on this folder, "
                "then rebuild the dashboard.</p>",
            )
        )
    prov = {
        "Campaign": state.get("campaign_config_path"),
        "Updated": (state.get("last_updated") or "")[:16],
        "Baseline": baseline,
        "Headline": ", ".join(headline_metrics(main_long, phase)) if phase else None,
    }
    pages["index"] = render_page(
        index,
        title=name,
        lede=state.get("description"),
        sections=sections,
        crumbs=[home],
        provenance=prov,
        plotly_js=plotly_js,
        assets=root / ASSETS,
    )
    logger.info("dashboard: wrote %d page(s) under %s", len(pages), root)
    return pages
