"""``comparison/_bars.py``: layout, zero baseline and whiskers shared by the grouped bar figures."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spectramr.infrastructure.reporting.comparison._bars import (
    GAP,
    arm_groups,
    decorate,
    draw_bar,
    grouped_positions,
    include_zero,
    wrap_label,
)


def starts_at_zero(ax) -> bool:
    """True when every bar is drawn from zero and the axis contains zero."""
    lo, hi = ax.get_ylim()
    return lo <= 0.0 <= hi and all(p.get_y() == 0.0 for p in ax.patches)


def test_arms_without_a_group_share_the_group_arms_and_order_is_kept():
    layout = arm_groups(["a", "b", "c", "d"], {"b": "x", "d": "x", "c": "y"})
    assert layout == {"arms": ["a"], "x": ["b", "d"], "y": ["c"]}  # first appearance, arm order
    assert arm_groups(["a", "b"], None) == {"arms": ["a", "b"]}


def test_slots_are_one_apart_and_groups_are_separated_by_the_gap():
    positions, spans = grouped_positions({"g1": ["a", "b"], "g2": ["c"]})
    assert positions[("g1", "a")] == 0.0 and positions[("g1", "b")] == 1.0
    assert positions[("g2", "c")] == 1.0 + 1.0 + GAP
    assert spans == {"g1": (0.0, 1.0), "g2": (2.0 + GAP, 2.0 + GAP)}


def test_a_bar_is_drawn_from_zero_and_a_planted_truncated_axis_is_detected():
    fig, ax = plt.subplots()
    draw_bar(ax, 0, 27.0, "C0")
    draw_bar(ax, 1, 28.0, "C1")
    include_zero(ax)
    assert starts_at_zero(ax)
    ax.set_ylim(26.0, 29.0)  # the violation the helper exists to prevent
    assert not starts_at_zero(ax)
    plt.close(fig)


def test_include_zero_widens_for_all_negative_and_all_positive_bars():
    fig, ax = plt.subplots()
    draw_bar(ax, 0, -3.0, "C0")
    include_zero(ax)
    assert ax.get_ylim()[1] >= 0.0
    plt.close(fig)


def test_a_whisker_needs_both_ends_and_a_non_finite_value_draws_nothing():
    fig, ax = plt.subplots()
    draw_bar(ax, 0, 1.0, "C0", lo=0.5, hi=1.5)
    draw_bar(ax, 1, 1.0, "C1", lo=np.nan, hi=np.nan)
    draw_bar(ax, 2, np.nan, "C2")
    assert len(ax.containers) == 3 and len(ax.patches) == 2  # two bars, one error bar container
    assert sum(type(c).__name__ == "ErrorbarContainer" for c in ax.containers) == 1
    plt.close(fig)


def test_group_headers_appear_only_with_several_groups_and_can_be_switched_off():
    def texts(layout, **kw):
        fig, ax = plt.subplots()
        positions, spans = grouped_positions(layout)
        decorate(ax, positions, spans, layout, **kw)
        out = [t.get_text() for t in ax.texts]
        plt.close(fig)
        return out

    assert texts({"only": ["a", "b"]}) == []
    assert texts({"g1": ["a"], "g2": ["b"]}) == ["g1", "g2"]
    assert texts({"g1": ["a"], "g2": ["b"]}, headers=False) == []


def test_wrap_label_breaks_a_long_metric_name():
    assert wrap_label("reference noise corrected mse", 16) == "reference noise\ncorrected mse"
    assert wrap_label("psnr") == "psnr"
    with pytest.raises(ValueError):
        wrap_label("x", 0)
