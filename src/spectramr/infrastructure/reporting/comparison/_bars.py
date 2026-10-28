r"""Layout and drawing shared by the grouped bar figures.

Bars are laid out in groups: each arm owns a slot of width 1, bars fill 0.8 of it, and
groups are separated by ``GAP`` empty slots. A group header is written above the axes
(skipped when there is only one group), and a faint line separates neighbouring groups.

Two rules are enforced here so no figure can break them. A bar always starts at zero,
and the axis always includes zero: a truncated bar turns a small difference into a large
one. A bar's whisker is passed in by the caller, who says what it is in the caption.
"""

from __future__ import annotations

import textwrap
from itertools import pairwise

import numpy as np

from ..style import colour_for

GAP = 0.8
BAR_WIDTH = 0.8
HATCH = "////"
#: Neutral greys; arm colours always come from ``colour_for``.
EDGE = "#555555"
GUIDE = "#aaaaaa"
UNGROUPED = "arms"


def arm_groups(arms: list[str], groups: dict[str, str] | None) -> dict[str, list[str]]:
    """Group name to its arms, groups in order of first appearance, arms in ``arms`` order."""
    out: dict[str, list[str]] = {}
    for arm in arms:
        out.setdefault((groups or {}).get(arm) or UNGROUPED, []).append(arm)
    return out


def grouped_positions(groups: dict[str, list[str]]) -> tuple[dict[tuple[str, str], float], dict]:
    """``({(group, arm): x}, {group: (first x, last x)})`` with ``GAP`` slots between groups."""
    positions: dict[tuple[str, str], float] = {}
    spans: dict[str, tuple[float, float]] = {}
    x = 0.0
    for name, members in groups.items():
        start = x
        for arm in members:
            positions[(name, arm)] = x
            x += 1.0
        spans[name] = (start, x - 1.0)
        x += GAP
    return positions, spans


def draw_bar(ax, x, value, colour, *, lo=np.nan, hi=np.nan, hatch=None, width=BAR_WIDTH, alpha=1.0):
    """One bar from zero to ``value`` with an optional whisker from ``lo`` to ``hi``."""
    if not np.isfinite(value):
        return
    ax.bar(x, value, width, color=colour, alpha=alpha, hatch=hatch, edgecolor=EDGE, linewidth=0.4)
    if np.isfinite(lo) and np.isfinite(hi):
        ax.errorbar(
            x,
            value,
            yerr=[[max(value - lo, 0.0)], [max(hi - value, 0.0)]],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
        )


def include_zero(ax) -> None:
    """Widen the y axis to contain zero, so no bar is drawn from a truncated baseline."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0.0), max(hi, 0.0))


def decorate(
    ax, positions, spans, groups, *, tick_labels: bool = True, headers: bool = True
) -> None:
    """Zero line, group separators and headers, and arm tick labels."""
    ax.axhline(0.0, color="#333333", lw=0.6, zorder=1)
    names = list(spans)
    for left, right in pairwise(names):
        ax.axvline((spans[left][1] + spans[right][0]) / 2, color=GUIDE, lw=0.5, ls="--")
    if headers and len(names) > 1:
        for name in names:
            ax.text(
                sum(spans[name]) / 2,
                1.01,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=6,
            )
    if tick_labels:
        ax.set_xticks([positions[(g, a)] for g, members in groups.items() for a in members])
        ax.set_xticklabels(
            [a for members in groups.values() for a in members], rotation=45, ha="right", fontsize=6
        )
    include_zero(ax)


def wrap_label(text: str, width: int = 16) -> str:
    """``text`` broken onto lines of at most ``width`` characters, so a long metric name fits."""
    return textwrap.fill(text, width)


def arm_colours(arms: list[str]) -> dict[str, str]:
    return {arm: colour_for(arm, i) for i, arm in enumerate(arms)}


__all__ = [
    "BAR_WIDTH",
    "GAP",
    "HATCH",
    "UNGROUPED",
    "arm_colours",
    "arm_groups",
    "decorate",
    "draw_bar",
    "grouped_positions",
    "include_zero",
    "wrap_label",
]
