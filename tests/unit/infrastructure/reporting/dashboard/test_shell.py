"""Tests for :mod:`spectramr.infrastructure.reporting.dashboard.shell`."""

from __future__ import annotations

import pandas as pd
import pytest

from spectramr.infrastructure.reporting.dashboard.shell import (
    PREVIEW_WIDTH,
    Section,
    figure_html,
    render_page,
    table_html,
    write_font_assets,
    write_plotly_asset,
)


def _png(path, width):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, width // 2), "white").save(path)
    return path


def test_figure_links_every_format_and_shows_a_small_png_as_is(tmp_path):
    _png(tmp_path / "figures" / "f.png", 400)
    (tmp_path / "figures" / "f.pdf").write_bytes(b"x")
    tile = figure_html(tmp_path / "figures" / "f.pdf", tmp_path / "page.html", caption="F")
    assert "src='figures/f.png'" in tile and "href='figures/f.pdf'" in tile
    assert "base64" not in tile


def test_large_png_tile_loads_a_cached_preview_and_links_the_original(tmp_path):
    png = _png(tmp_path / "figures" / "big.png", 4000)
    tile = figure_html(png, tmp_path / "page.html", caption="Big")
    small = tmp_path / "figures" / "_preview" / "big.webp"
    assert "src='figures/_preview/big.webp'" in tile and "href='figures/big.png'" in tile
    from PIL import Image

    with Image.open(small) as im:
        assert im.width == PREVIEW_WIDTH
    mtime = small.stat().st_mtime_ns
    figure_html(png, tmp_path / "page.html", caption="Big")
    assert small.stat().st_mtime_ns == mtime  # reused, not re-encoded


def test_pdf_only_figure_is_linked_not_dropped(tmp_path):
    (tmp_path / "f.pdf").write_bytes(b"x")
    tile = figure_html(tmp_path / "f.pdf", tmp_path / "page.html", caption="F")
    assert tile is not None and "href='f.pdf'" in tile and "<img" not in tile


def test_missing_figure_yields_none(tmp_path):
    assert figure_html(tmp_path / "nope.pdf", tmp_path / "p.html", caption="x") is None


def test_table_marks_best_cell_baseline_row_and_numeric_columns():
    df = pd.DataFrame({"arm": ["a", "b"], "psnr": [30.0, 31.0]})
    doc = table_html(df, best={"psnr": "b"}, baseline="a")
    assert "<tr class='base'><td>a</td>" in doc
    assert "<td class='num best'>31</td>" in doc


def test_page_escapes_text_and_links_plotly_only_when_used(tmp_path):
    js = tmp_path / "_assets" / "plotly.min.js"
    js.parent.mkdir()
    js.write_text("//")
    plain = render_page(
        tmp_path / "a.html", title="<x>", sections=[Section("s", "S", "<p>hi</p>")], plotly_js=js
    ).read_text()
    assert "&lt;x&gt;" in plain and "plotly.min.js" not in plain
    used = render_page(
        tmp_path / "b.html",
        title="t",
        sections=[Section("s", "S", "<div class='plotly-graph-div'></div>")],
        plotly_js=js,
    ).read_text()
    assert "src='_assets/plotly.min.js'" in used


def test_page_carries_both_colour_modes(tmp_path):
    doc = render_page(tmp_path / "p.html", title="t", sections=[]).read_text()
    assert ":root[data-theme=dark]" in doc and "prefers-color-scheme:dark" in doc
    assert "data-mode='dark'" in doc


def test_plotly_asset_written_once(tmp_path):
    pytest.importorskip("plotly")
    p = write_plotly_asset(tmp_path)
    mtime = p.stat().st_mtime_ns
    assert write_plotly_asset(tmp_path).stat().st_mtime_ns == mtime


def test_page_links_bundled_fonts_and_no_font_host(tmp_path):
    doc = render_page(tmp_path / "run" / "p.html", title="t", sections=[]).read_text()
    assert "href='_assets/fonts/fonts.css'" in doc and "googleapis" not in doc
    css = (tmp_path / "run" / "_assets" / "fonts" / "fonts.css").read_text()
    for name in ("IBMPlexSans-SemiBold-Latin1.woff2", "SourceSerif4-Semibold.otf.woff2"):
        assert name in css and (tmp_path / "run" / "_assets" / "fonts" / name).is_file()


def test_font_assets_ship_their_licences_and_are_written_once(tmp_path):
    css = write_font_assets(tmp_path)
    assert {"LICENSE-IBM-Plex.txt", "LICENSE-Source-Serif.md"} <= {p.name for p in css.parent.iterdir()}
    mtime = css.stat().st_mtime_ns
    assert write_font_assets(tmp_path).stat().st_mtime_ns == mtime
