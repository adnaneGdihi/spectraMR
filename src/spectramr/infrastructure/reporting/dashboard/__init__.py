"""The reporting dashboard: one page shell and theme for every HTML report surface."""

from .pages import comparison_page
from .run_page import build_html_report
from .shell import Section, render_page, write_plotly_asset
from .site import build_site

__all__ = [
    "Section",
    "build_html_report",
    "build_site",
    "comparison_page",
    "render_page",
    "write_plotly_asset",
]
