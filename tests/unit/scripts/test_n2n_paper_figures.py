"""``scripts/n2n_paper_figures.py``: each figure renders from the synthetic cohort.

The figures are read by eye, so the tests pin only what a wrong figure would get
wrong silently: the Rep-BV panel skips when no arm scored the decomposition, the
theta sweep needs at least two angles, and every rendered figure lands as PDF
and PNG under the stem the paper links.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.unit.scripts.n2n_paper_fixture import build_runs

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = REPO_ROOT / "scripts"


def _module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def figures() -> Any:
    return _module("n2n_paper_figures")


@pytest.fixture(scope="module")
def common() -> Any:
    return _module("n2n_paper_common")


def test_every_figure_lands_as_pdf_and_png(figures: Any, common: Any, tmp_path: Path) -> None:
    arms = common.discover_arms(build_runs(tmp_path / "runs"))
    shared = common.common_iteration(arms)
    out = tmp_path / "figures"
    for stem in (
        figures.fig_curves(arms, out),
        figures.fig_repbv(arms, shared, out),
        figures.fig_theta(arms, shared, out),
    ):
        assert stem is not None and stem.exists() and stem.with_suffix(".png").exists()


def test_panels_that_have_nothing_to_show_are_skipped(
    figures: Any, common: Any, tmp_path: Path
) -> None:
    arms = common.discover_arms(build_runs(tmp_path / "runs"))
    shared = common.common_iteration(arms)
    poor = [a for a in arms if a.letter in ("A", "B")]
    assert figures.fig_repbv(poor, shared, tmp_path / "poor") is None
    assert figures.fig_theta(poor, shared, tmp_path / "poor") is None
    assert not (tmp_path / "poor").exists()
