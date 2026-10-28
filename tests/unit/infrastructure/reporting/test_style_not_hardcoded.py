"""No plotter may pin a house style.

``use_default_style("nature")`` inside a plotter also *resets* the active style,
so under ``reporting.style: ieee`` every figure rendered after the first such
plotter came out Nature -- 13 plotters did this. Plotters call the bare
``use_default_style()``, which re-applies whatever ``generate_report`` selected.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_PLOTTERS = Path("src/spectramr/infrastructure/reporting/plotters")
#: A literal string as the first positional argument, or a ``style=`` keyword.
_PINNED = re.compile(r"use_default_style\(\s*(?:style\s*=\s*)?[\"']")


def _offenders(text: str) -> list[str]:
    return [m.group(0) for m in _PINNED.finditer(text)]


@pytest.mark.parametrize(
    "planted",
    [
        'use_default_style("nature")',
        "use_default_style('ieee')",
        'use_default_style(style="nature")',
        'use_default_style(\n    "poster"\n)',
    ],
    ids=["double-quoted", "single-quoted", "keyword", "multiline"],
)
def test_the_detector_sees_every_pinning_shape(planted):
    assert _offenders(f"def make():\n    {planted}\n")


@pytest.mark.parametrize("clean", ["use_default_style()", "use_default_style(style)"])
def test_the_detector_passes_inherited_styles(clean):
    assert not _offenders(clean)


def test_no_plotter_pins_a_style():
    hits = {
        str(p): _offenders(p.read_text(encoding="utf-8"))
        for p in _PLOTTERS.rglob("*.py")
        if _offenders(p.read_text(encoding="utf-8"))
    }
    assert not hits, f"plotters pinning a house style: {hits}"


def test_every_style_the_schema_offers_has_a_file():
    from spectramr.config.schemas.enums import ReportStyle
    from spectramr.infrastructure.reporting import style

    for member in ReportStyle:
        assert style._STYLE_FILES[member.value].exists(), member
        style.use_default_style(member.value)
    style.use_default_style("nature")
