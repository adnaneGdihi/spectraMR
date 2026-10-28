"""One legend, one palette: the clutter policy every report figure shares.

A figure carries its axes, panel letters and at most one legend; counts, tests and
provenance go to the sidecar or a table. ``arm_palette`` keeps an arm's colour fixed
across figures and ``shared_legend`` caps the entries so a many-arm figure stays legible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from .style import METHOD_COLOURS, OKABE_ITO, colour_for

MAX_LEGEND_ENTRIES = 12
LEGEND_COLUMNS = 6

# beyond the eight Okabe-Ito hues: distinct, colour-blind-tolerant extras so 12 arms never share one
_EXTRA_HUES = ("#7F7F7F", "#8C564B", "#17BECF", "#7B3294", "#A6761D")


def arm_palette(arms: Sequence[str]) -> dict[str, str]:
    """Colour per arm, stable across figures (registered methods keep their own colour)."""
    pool = [c for c in OKABE_ITO if c.lower() != "#000000"] + list(_EXTRA_HUES)
    out: dict[str, str] = {}
    for i, arm in enumerate(arms):
        registered = str(arm).lower() in METHOD_COLOURS
        out[str(arm)] = colour_for(str(arm), i) if registered else pool[i % len(pool)]
    return out


def collect_handles(axes: Iterable) -> tuple[list, list[str]]:
    """Unique ``(handle, label)`` pairs over ``axes`` in first-seen order, hidden labels dropped."""
    handles: list = []
    labels: list[str] = []
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=False):
            if label and not label.startswith("_") and label not in labels:
                handles.append(handle)
                labels.append(label)
    return handles, labels


def legend_columns(labels: Sequence[str]) -> int:
    """Columns that keep the legend inside the figure: fewer for longer labels.

    A bottom legend wider than the axes widens the saved figure (``bbox: tight``) and shrinks
    the data, so the column count follows the longest label, not the entry count alone.
    """
    longest = max((len(str(label)) for label in labels), default=0)
    fit = 6 if longest <= 14 else 4 if longest <= 22 else 3 if longest <= 32 else 2
    return max(1, min(len(labels), fit, LEGEND_COLUMNS))


def shared_legend(
    fig,
    handles: Sequence,
    labels: Sequence[str],
    *,
    max_entries: int = MAX_LEGEND_ENTRIES,
    anchor_y: float = -0.02,
) -> Legend | None:
    """One frameless legend under the panels, capped at ``max_entries``.

    Beyond the cap the tail collapses into a single ``+k more`` entry, so the legend
    never grows into the data region.
    """
    handles, labels = list(handles), list(labels)
    if not labels:
        return None
    if len(labels) > max_entries:
        hidden = len(labels) - (max_entries - 1)
        handles = [*handles[: max_entries - 1], Line2D([], [], linestyle="none")]
        labels = [*labels[: max_entries - 1], f"+{hidden} more"]
    return fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_columns(labels),
        frameon=False,
        bbox_to_anchor=(0.5, anchor_y),
    )


def reserve_legend_space(fig, row_inches: float = 0.19, pad_inches: float = 0.12) -> float:
    """Grow ``fig`` by its bottom legend's height; return the fraction to leave free below the axes.

    Reserving a fixed fraction squeezed the data when the legend had many rows on a short
    figure; adding the legend's own height keeps the axes at their designed size. Use it as
    ``fig.tight_layout(rect=(0, reserve_legend_space(fig), 1, 1))``.
    """
    if not fig.legends:
        return 0.0
    legend = fig.legends[-1]
    rows = -(-len(legend.get_texts()) // max(legend._ncols, 1))
    extra = rows * row_inches + pad_inches
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height + extra)
    return extra / (height + extra)


def sparse_ticks(ax, n: int = 4, *, axes: str = "xy") -> None:
    """At most ``n`` major ticks on each of ``axes`` (the default thicket reads as clutter).

    A categorical axis (fixed tick labels) must be left out: ``axes="y"``.
    """
    if "x" in axes and ax.get_xscale() == "linear":
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=2))
    if "y" in axes and ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=2))


def quiet_axes(ax) -> None:
    """No grid, no top/right spines, whatever the active style says."""
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
