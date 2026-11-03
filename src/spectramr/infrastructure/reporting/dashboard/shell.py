"""The one HTML page shell every reporting surface renders through.

A page is a title, an optional lede, provenance, and an ordered list of sections; the
shell adds the breadcrumb bar, the light/dark switch, the section nav and the footer.
Figures are linked, not base64-embedded, and plotly.js is written once per site as a
sibling file, so a 15-arm comparison is kilobytes of HTML instead of tens of megabytes.
"""

from __future__ import annotations

import html
import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


from .theme import script, stylesheet

logger = logging.getLogger(__name__)

PLOTLY_ASSET = "plotly.min.js"


@dataclass(frozen=True)
class Section:
    """One navigable block of a page; ``body`` is trusted, already-escaped HTML."""

    id: str
    title: str
    body: str
    note: str | None = None


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def rel(target: str | Path, page: str | Path) -> str:
    """``target`` as a URL relative to the directory holding ``page``."""
    return Path(os.path.relpath(Path(target), Path(page).parent)).as_posix()


def write_plotly_asset(assets_dir: str | Path) -> Path | None:
    """Write ``plotly.min.js`` into ``assets_dir`` once; None when plotly is absent."""
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        logger.info("plotly not installed; interactive charts fall back to static figures")
        return None
    path = Path(assets_dir) / PLOTLY_ASSET
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(get_plotlyjs(), encoding="utf-8")
    return path


# The authors' own WOFF2 releases (IBM's Latin-1/Greek split files, Adobe's Source Serif 4
# Semibold), unmodified: a self-made subset is a Modified Version under the OFL, which may
# not keep the Reserved Font Names "Plex" and "Source".
_FONT_DIR = Path(__file__).with_name("fonts")
_LATIN1 = (
    "U+0000,U+000D,U+0020-007E,U+00A0-00FF,U+0131,U+0152-0153,U+02C6,U+02DA,U+02DC,"
    "U+2013-2014,U+2018-201A,U+201C-201E,U+2020-2022,U+2026,U+2030,U+2039-203A,U+2044,"
    "U+2074,U+20AC,U+2122,U+2212,U+FB01-FB02"
)
_GREEK = "U+0384-038A,U+038C,U+038E-03A1,U+03A3-03CE"
_FACES = (
    ("IBM Plex Sans", 400, "IBMPlexSans-Regular-Latin1.woff2", _LATIN1),
    ("IBM Plex Sans", 400, "IBMPlexSans-Regular-Greek.woff2", _GREEK),
    ("IBM Plex Sans", 500, "IBMPlexSans-Medium-Latin1.woff2", _LATIN1),
    ("IBM Plex Sans", 500, "IBMPlexSans-Medium-Greek.woff2", _GREEK),
    ("IBM Plex Sans", 600, "IBMPlexSans-SemiBold-Latin1.woff2", _LATIN1),
    ("IBM Plex Sans", 600, "IBMPlexSans-SemiBold-Greek.woff2", _GREEK),
    ("IBM Plex Mono", 400, "IBMPlexMono-Regular-Latin1.woff2", _LATIN1),
    ("Source Serif 4", 600, "SourceSerif4-Semibold.otf.woff2", None),
)
FONTS_CSS = "fonts/fonts.css"


def write_font_assets(assets_dir: str | Path) -> Path:
    """Copy the bundled typefaces and their licences into ``assets_dir/fonts`` once.

    Returns the ``fonts.css`` to link. Pages then render in the house faces offline and
    without a request to a third-party font host.
    """
    css = Path(assets_dir) / FONTS_CSS
    if css.exists():
        return css
    css.parent.mkdir(parents=True, exist_ok=True)
    for src in _FONT_DIR.iterdir():
        shutil.copyfile(src, css.parent / src.name)
    css.write_text(
        "".join(
            f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
            f"font-display:swap;src:url('{name}') format('woff2')"
            + (f";unicode-range:{rng}" if rng else "")
            + "}\n"
            for family, weight, name, rng in _FACES
        ),
        encoding="utf-8",
    )
    return css


PREVIEW_WIDTH = 1200  # px: a ~420 px tile at 2-3x density


def preview(png: Path) -> Path:
    """A downscaled WebP of ``png`` in ``_preview/`` beside it, rebuilt when the PNG is newer.

    Report PNGs are drawn at 600 dpi (~4000 px, ~2 MB each); a gallery tile needs a
    tenth of that. The full-resolution file stays one click away.
    """
    from PIL import Image, UnidentifiedImageError

    out = png.parent / "_preview" / f"{png.stem}.webp"
    if out.exists() and out.stat().st_mtime >= png.stat().st_mtime:
        return out
    try:
        with Image.open(png) as im:
            if im.width <= PREVIEW_WIDTH:
                return png
            out.parent.mkdir(exist_ok=True)
            small = im.convert("RGBA" if "A" in im.getbands() else "RGB")
            small.thumbnail((PREVIEW_WIDTH, PREVIEW_WIDTH * 4))
            small.save(out, "WEBP", quality=88, method=4)
    except (OSError, UnidentifiedImageError) as exc:  # a truncated download must not sink the page
        logger.warning("no preview for %s (%s); the tile links the original", png, exc)
        return png
    return out


def figure_html(
    path: str | Path, page: str | Path, *, caption: str, wide: bool = False
) -> str | None:
    """A linked figure tile: PNG preview, and links to every format written beside it."""
    p = Path(path)
    png = p.with_suffix(".png")
    formats = [f for f in (".png", ".pdf", ".svg") if p.with_suffix(f).exists()]
    if not formats:
        return None
    links = "".join(
        f"<a href='{esc(rel(p.with_suffix(f), page))}'>{f[1:].upper()}</a>" for f in formats
    )
    if png.exists():
        img = (
            f"<a href='{esc(rel(png, page))}'><img loading='lazy' decoding='async' "
            f"src='{esc(rel(preview(png), page))}' alt='{esc(caption)}'></a>"
        )
    else:
        img = f"<p class='empty'>No PNG preview for {esc(caption)}; open the file.</p>"
    cls = " class='wide'" if wide else ""
    return (
        f"<figure{cls}>{img}<figcaption><span>{esc(caption)}</span>"
        f"<span class='fmt'>{links}</span></figcaption></figure>"
    )


def gallery(tiles: Sequence[str | None]) -> str:
    kept = [t for t in tiles if t]
    return f"<div class='gallery'>{''.join(kept)}</div>" if kept else ""


def table_html(
    df: pd.DataFrame,
    *,
    best: Mapping[str, object] | None = None,
    baseline: str | None = None,
    digits: int = 4,
    max_rows: int = 200,
) -> str:
    """A scrollable table; ``best`` maps column -> the row label holding its best value."""
    if df.empty:
        return "<p class='empty'>No rows.</p>"
    best = best or {}
    cols = list(df.columns)
    numeric = {c for c in cols if pd.api.types.is_numeric_dtype(df[c])}
    head = "".join(
        f"<th scope='col'{' class=num' if c in numeric else ''}>{esc(c)}</th>" for c in cols
    )
    rows = []
    for _, r in df.head(max_rows).iterrows():
        label = r[cols[0]]
        cells = []
        for c in cols:
            v = r[c]
            txt = f"{v:.{digits}g}" if isinstance(v, float) else ("" if pd.isna(v) else str(v))
            flags = [
                k
                for k, on in (("num", c in numeric), ("best", c in best and best[c] == label))
                if on
            ]
            cells.append(
                f"<td class='{' '.join(flags)}'>{esc(txt)}</td>"
                if flags
                else f"<td>{esc(txt)}</td>"
            )
        tr = " class='base'" if baseline is not None and label == baseline else ""
        rows.append(f"<tr{tr}>{''.join(cells)}</tr>")
    more = (
        f"<p class='meta'>Showing {max_rows} of {len(df)} rows.</p>" if len(df) > max_rows else ""
    )
    return (
        f"<div class='tbl-wrap'><table class='tbl'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>{more}"
    )


def cards(items: Sequence[tuple[str, str, str]]) -> str:
    """Link cards from ``(href, title, detail)`` triples (already relative hrefs)."""
    return (
        "<div class='cards'>"
        + "".join(
            f"<a class='card' href='{esc(h)}'><b>{esc(t)}</b><span>{esc(d)}</span></a>"
            for h, t, d in items
        )
        + "</div>"
    )


def render_page(
    out_path: str | Path,
    *,
    title: str,
    sections: Sequence[Section],
    lede: str | None = None,
    crumbs: Sequence[tuple[str, str | None]] = (),
    provenance: Mapping[str, object] | None = None,
    plotly_js: str | Path | None = None,
    assets: str | Path | None = None,
) -> Path:
    """Write one page to ``out_path`` and return it.

    Args:
        crumbs: ``(label, href or None)`` from the site root to this page.
        provenance: label -> value pairs shown under the title (git, data, generated ...).
        plotly_js: path to the shared ``plotly.min.js``; linked only when a section needs it.
        assets: the shared assets directory the fonts are written to; defaults to the one
            holding ``plotly_js``, else ``_assets`` beside the page.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    crumb_html = "".join(
        f"<a href='{esc(h)}'>{esc(lbl)}</a>" if h else f"<span>{esc(lbl)}</span>"
        for lbl, h in crumbs
    )
    prov = "".join(
        f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>"
        for k, v in (provenance or {}).items()
        if v not in (None, "")
    )
    toc = "".join(f"<li><a href='#{esc(s.id)}'>{esc(s.title)}</a></li>" for s in sections)
    body = "".join(
        f"<section id='{esc(s.id)}'><h2>{esc(s.title)}</h2>"
        + (f"<p class='note'>{esc(s.note)}</p>" if s.note else "")
        + s.body
        + "</section>"
        for s in sections
    )
    needs_plotly = plotly_js is not None and "plotly-graph-div" in body
    js_tag = f"<script src='{esc(rel(plotly_js, out))}'></script>" if needs_plotly else ""
    if assets is None:
        assets = Path(plotly_js).parent if plotly_js is not None else out.parent / "_assets"
    fonts = f"<link rel='stylesheet' href='{esc(rel(write_font_assets(assets), out))}'>"
    doc = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{esc(title)}</title>{fonts}<style>{stylesheet()}</style>"
        f"<script>{script()}</script>{js_tag}</head><body>"
        f"<div class='bar'><nav class='crumbs' aria-label='Breadcrumb'>{crumb_html}</nav>"
        "<div class='mode' role='group' aria-label='Colour theme'>"
        "<button type='button' data-mode='light' aria-pressed='true'>Light</button>"
        "<button type='button' data-mode='dark' aria-pressed='false'>Dark</button></div></div>"
        f"<header class='head'><h1>{esc(title)}</h1>"
        + (f"<p class='lede'>{esc(lede)}</p>" if lede else "")
        + (f"<dl class='prov'>{prov}</dl>" if prov else "")
        + "</header><div class='frame'>"
        + (
            f"<nav class='toc' aria-label='Sections'><ol>{toc}</ol></nav>"
            if len(sections) > 1
            else "<div></div>"
        )
        + f"<main>{body}</main></div><footer>Generated by spectramr report.</footer></body></html>"
    )
    out.write_text(doc, encoding="utf-8")
    return out
