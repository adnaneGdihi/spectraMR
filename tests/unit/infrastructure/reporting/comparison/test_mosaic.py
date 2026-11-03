"""Tests for :mod:`spectramr.infrastructure.reporting.comparison.mosaic`."""

from __future__ import annotations

import dataclasses
from itertools import pairwise

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_arms
from spectramr.infrastructure.reporting.comparison import mosaic


def test_sixteen_long_arm_titles_do_not_overlap(tmp_path, monkeypatch):
    base = make_arms()
    names = [f"{c}_nullspace_r2r_single_ex_theta20" for c in "abcdefghijklmnop"]
    arms = dataclasses.replace(
        base,
        labels=names,
        cases=dict.fromkeys(names, base.cases["a"]),
        steps=dict.fromkeys(names, 10),
    )
    seen = []
    monkeypatch.setattr(plt, "close", seen.append)
    mosaic.make(pd.DataFrame(), tmp_path / "m.png", arms=arms, formats=("png",), dpi=40)
    fig = seen[0]
    renderer = fig.canvas.get_renderer()
    boxes = [ax.title.get_window_extent(renderer) for ax in fig.axes[:17] if ax.get_title()]
    assert len(boxes) == 17
    for left, right in pairwise(boxes):
        assert left.x1 <= right.x0 + 1, (left, right)
