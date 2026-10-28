"""The catalog renders every registered figure and table from its example."""

import matplotlib

matplotlib.use("Agg")

from spectramr.infrastructure.reporting import plotters, tables
from spectramr.infrastructure.reporting.catalog import EXAMPLES, TABLE_EXAMPLES, render_catalog


def test_every_registered_figure_has_an_example():
    assert set(plotters.list_available()) <= set(EXAMPLES)


def test_every_registered_table_has_an_example():
    assert set(tables.list_available()) <= set(TABLE_EXAMPLES)


def test_the_catalog_renders_everything(tmp_path):
    res = render_catalog(tmp_path, formats=("png",), dpi=40)
    missing = {fid: res["reasons"].get(fid) for fid, p in res["figures"].items() if p is None}
    assert not missing, missing
    assert all(v for v in res["tables"].values()), res["tables"]
    assert len(list((tmp_path / "figures").glob("*.png"))) == len(plotters.list_available())
