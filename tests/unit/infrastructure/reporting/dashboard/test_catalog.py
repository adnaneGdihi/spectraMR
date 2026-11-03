"""Tests for :mod:`spectramr.infrastructure.reporting.dashboard.catalog`."""

from __future__ import annotations

import pandas as pd
from PIL import Image

from spectramr.infrastructure.reporting.dashboard.catalog import (
    catalog_page,
    catalog_page_from_disk,
)


def _tree(root):
    (root / "figures").mkdir(parents=True)
    (root / "tables").mkdir()
    for fid in ("cmp_mosaic", "fig_1_2_learning_curves", "ksd_certificate"):
        Image.new("RGB", (64, 32)).save(root / "figures" / f"{fid}.png")
    pd.DataFrame({"arm": ["a"], "psnr": [30.0]}).to_csv(
        root / "tables" / "tab_2_1_main_results.csv", index=False
    )
    return root


def test_catalog_groups_figures_inlines_tables_and_lists_failures(tmp_path):
    root = _tree(tmp_path)
    figs = {p.stem: p for p in (root / "figures").glob("*.png")} | {"fig_x": None}
    doc = catalog_page(
        root,
        figures=figs,
        reasons={"fig_x": "no_example"},
        tables={"tab_2_1_main_results": {"csv": root / "tables" / "tab_2_1_main_results.csv"}},
    ).read_text()
    for heading in ("Images", "Training", "Physics and certificates", "Tables", "Not drawn"):
        assert heading in doc
    assert "<code>fig_x</code>: no_example" in doc and "<td class='num'>30</td>" in doc


def test_catalog_rebuilds_from_disk_with_a_way_home(tmp_path):
    doc = catalog_page_from_disk(
        _tree(tmp_path), crumbs=[("Campaign x", "../index.html")]
    ).read_text()
    assert "href='../index.html'" in doc and "Reconstructions side by side" in doc
