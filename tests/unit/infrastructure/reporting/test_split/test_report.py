"""``test/report.html`` holds the summary tables, the distributions and every slice."""

from __future__ import annotations

import json

import pytest

from spectramr.infrastructure.reporting.test_split.report import (
    DISTRIBUTIONS_PNG,
    write_test_report,
)

from .test_summary import _rows

pytestmark = pytest.mark.unit


def test_the_page_carries_tables_figure_and_gallery(tmp_path) -> None:
    _rows(tmp_path)
    meta = tmp_path / "test_metrics.json"
    meta.write_text(json.dumps({"manifest": "m.json", "n_cases": 4}))
    page = write_test_report(tmp_path, meta).read_text()
    assert (tmp_path / DISTRIBUTIONS_PNG).is_file()
    assert "m.json" in page and "<h3>contrast</h3>" in page
    assert page.count("<img loading='lazy'") == 4


def test_no_rows_no_page(tmp_path) -> None:
    assert write_test_report(tmp_path) is None
