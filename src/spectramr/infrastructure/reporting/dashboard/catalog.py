"""The figure catalog page: every registered figure and table, drawn from synthetic examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .pages import gallery_sections
from .shell import Section, esc, rel, render_page, table_html

LEDE = (
    "Every figure and table the report can draw, each rendered from a built-in synthetic "
    "example. The inputs are invented: a figure here shows what the plotter draws, not what "
    "any run measured."
)


def _tables_section(tables: Mapping[str, Mapping[str, Path] | None], page: Path) -> Section | None:
    blocks = []
    for tid, files in sorted(tables.items()):
        if not files:
            continue
        links = " ".join(
            f"<a href='{esc(rel(p, page))}'>{esc(fmt)}</a>" for fmt, p in sorted(files.items())
        )
        csv = files.get("csv")
        body = table_html(pd.read_csv(csv)) if csv is not None and Path(csv).exists() else ""
        blocks.append(f"<h3>{esc(tid)}</h3><p class='meta'>{links}</p>{body}")
    return Section("tables", "Tables", "".join(blocks)) if blocks else None


def catalog_page(
    out_dir: str | Path,
    *,
    figures: Mapping[str, Path | None],
    reasons: Mapping[str, str],
    tables: Mapping[str, Mapping[str, Path] | None],
    crumbs: Sequence[tuple[str, str | None]] = (),
) -> Path:
    """Write ``<out_dir>/catalog.html`` from a :func:`render_catalog` result."""
    out_dir = Path(out_dir)
    page = out_dir / "catalog.html"
    drawn = [fid for fid, p in figures.items() if p is not None]
    sections = gallery_sections(out_dir, page, drawn)
    tbl = _tables_section(tables, page)
    if tbl is not None:
        sections.append(tbl)
    missing = {fid: reasons.get(fid, "no output") for fid, p in figures.items() if p is None}
    if missing:
        items = "".join(
            f"<li><code>{esc(f)}</code>: {esc(r)}</li>" for f, r in sorted(missing.items())
        )
        sections.append(
            Section(
                "not-drawn",
                "Not drawn",
                f"<ul>{items}</ul>",
                note="Registered figures with no example, or whose example failed.",
            )
        )
    prov = {
        "Figures": f"{len(drawn)} of {len(figures)}",
        "Tables": sum(1 for t in tables.values() if t),
    }
    return render_page(
        page,
        title="Figure catalog",
        lede=LEDE,
        sections=sections,
        crumbs=[*crumbs, ("Figure catalog", None)],
        provenance=prov,
    )


def catalog_page_from_disk(
    out_dir: str | Path, *, crumbs: Sequence[tuple[str, str | None]] = ()
) -> Path:
    """Re-render ``catalog.html`` from the figures and tables an earlier catalog run wrote."""
    out_dir = Path(out_dir)
    figdir, tabdir = out_dir / "figures", out_dir / "tables"
    figures = {p.stem: p for p in sorted(figdir.glob("*")) if p.suffix in (".png", ".pdf", ".svg")}
    tables: dict[str, dict[str, Path]] = {}
    for p in sorted(tabdir.glob("*")) if tabdir.is_dir() else []:
        tables.setdefault(p.stem, {})[p.suffix[1:]] = p
    return catalog_page(out_dir, figures=figures, reasons={}, tables=tables, crumbs=crumbs)
