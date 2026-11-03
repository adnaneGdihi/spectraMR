"""Tests for :mod:`spectramr.infrastructure.reporting.comparison.figure_set`."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.reporting import plotters
from spectramr.infrastructure.reporting.comparison.figure_set import (
    COMPARISON_FIGURES,
    DRAWN_ELSEWHERE,
    comparison_figure_ids,
)


def test_default_is_the_comparison_set():
    assert comparison_figure_ids(None) == COMPARISON_FIGURES


def test_all_attempts_every_registered_figure_once():
    ids = comparison_figure_ids("all")
    assert len(ids) == len(set(ids))
    assert set(ids) | DRAWN_ELSEWHERE == set(plotters.list_available())
    assert comparison_figure_ids(["all"]) == ids


def test_ids_drawn_by_their_own_block_are_registered_and_not_redispatched():
    assert set(plotters.list_available()) >= DRAWN_ELSEWHERE
    assert comparison_figure_ids(["cmp_phase_bars", "cmp_mosaic"]) == ("cmp_mosaic",)


def test_an_unknown_id_raises():
    with pytest.raises(ValueError, match="unknown figure id"):
        comparison_figure_ids(["cmp_mosiac"])
