"""Tests for :mod:`spectramr.infrastructure.reporting.dashboard.figure_index`."""

from __future__ import annotations

from spectramr.infrastructure.reporting import plotters
from spectramr.infrastructure.reporting.dashboard.figure_index import OTHER, TITLES, caption, family


def test_every_registered_figure_has_a_family_and_a_title():
    ids = plotters.list_available()
    assert [i for i in ids if family(i) == OTHER] == []
    assert [i for i in ids if i not in TITLES] == []


def test_fig_1_1_does_not_swallow_fig_1_11_to_1_18():
    assert family("fig_1_1_headline_pareto") == "Headline metrics"
    assert family("fig_1_17_metric_correlation") == "Per-case distributions"
    assert family("fig_1_18_train_val_gap") == "Training"


def test_caption_names_known_figures_and_numbers_repeats():
    assert caption("cmp_metric_bars") == "Metric means per arm"
    assert caption("cmp_metric_bars_2") == "Metric means per arm (2)"
    assert caption("cmp_new_thing") == "New thing"
