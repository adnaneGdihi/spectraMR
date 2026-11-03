"""Regression test pinning the METHOD_COLOURS coverage.

Locks beautify-backlog item 2.3
(``TODO/backlog_beautify_logging_and_reporting.md``): every registered
strategy in ``STRATEGY_CLASS_PATHS`` must have a deterministic colour
in :data:`spectramr.infrastructure.reporting.style.METHOD_COLOURS`. Without
this, headline figures fall back to the matplotlib cycler order, which
gives the same method two different colours in different plots.

When a new strategy is registered:

1. Add a colour to ``METHOD_COLOURS`` (group it with its paradigm family).
2. This test passes again automatically.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def _read_strategy_keys() -> set[str]:
    # post-2026-05 src→spectramr refactor layout + import prefix
    factory = (REPO_ROOT / "src" / "spectramr" / "infrastructure" / "training"
               / "strategy_factory.py").read_text()
    return set(re.findall(
        r'^\s+"([a-z_]+)":\s+"(?:src|spectramr)\.infrastructure', factory, re.M))


def _read_method_colour_keys() -> set[str]:
    style = (REPO_ROOT / "src" / "spectramr" / "infrastructure" / "reporting" / "style.py").read_text()
    m = re.search(r"METHOD_COLOURS:[^=]*=\s*\{(.+?)\n\}", style, re.S)
    assert m, "Could not locate METHOD_COLOURS in style.py"
    return set(re.findall(r'"([a-z_]+)":', m.group(1)))


def test_every_strategy_has_a_method_colour() -> None:
    """All registered strategies have a colour assignment."""
    strategy_keys = _read_strategy_keys()
    colour_keys = _read_method_colour_keys()
    missing = sorted(strategy_keys - colour_keys)
    assert not missing, (
        f"{len(missing)} registered strategies lack a METHOD_COLOURS entry. "
        f"Add them to src/spectramr/infrastructure/reporting/style.py grouped by "
        f"paradigm family. Missing: {missing}"
    )


def test_colour_for_a_known_strategy_returns_registered_value() -> None:
    """``colour_for`` resolves a deterministic value (not cycler fallback)."""
    from spectramr.infrastructure.reporting.style import (
        METHOD_COLOURS,
        colour_for,
    )
    from spectramr.core.design_tokens import CATEGORICAL

    # Pick a known key.
    assert "diffusion" in METHOD_COLOURS
    expected = METHOD_COLOURS["diffusion"]
    assert colour_for("diffusion") == expected
    assert colour_for("DIFFUSION") == expected  # case-insensitive
    # An unknown key takes one of the eight validated categorical hues.
    assert colour_for("totally_unknown") in CATEGORICAL
