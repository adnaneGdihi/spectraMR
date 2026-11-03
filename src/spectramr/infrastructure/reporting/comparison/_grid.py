r"""Panel grids and numbered continuation files shared by the cross-arm figures.

A figure of many panels is split into chunks of at most ``MAX_PANELS`` and written
as ``<stem>``, ``<stem>_2``, ``<stem>_3``: one grid of dozens of panels is
unreadable, and the panel letters ``a``, ``b``, ... are only defined up to ``z``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from ..metadata import RunMetadata, stamp_figure, write_sidecar
from ..style import column_width, panel_label, panel_letter, save_figure, use_default_style

#: Panels per figure and per row.
MAX_PANELS = 12
PANEL_COLUMNS = 4


def chunks(names: Sequence, size: int) -> list[list]:
    """``names`` in consecutive groups of at most ``size``."""
    if size < 1:
        raise ValueError(f"chunk size must be at least 1, got {size}")
    return [list(names[i : i + size]) for i in range(0, len(names), size)]


#: Below this many inches per bar slot, arm labels collide and bars blur into each other.
SLOT_INCHES = 0.2
#: Never wider than two page columns side by side.
MAX_WIDTH_PAGES = 2.0


def new_grid(
    n_panels: int,
    *,
    row_height: float = 0.36,
    columns: int = PANEL_COLUMNS,
    slots: float | None = None,
    label_chars: int = 0,
):
    """A figure with ``n_panels`` axes in rows of ``columns``; returns ``(fig, axes)``.

    ``slots`` (bar slots per panel) and ``label_chars`` (longest rotated tick label) size the
    grid to its content: fewer columns, or a wider figure up to two page widths, until each
    bar has ``SLOT_INCHES``, and rows tall enough for the labels. Without them the grid
    keeps the fixed page-width layout. Axes beyond ``n_panels`` are hidden, and each used
    axis carries its panel letter when there is more than one.
    """
    use_default_style()
    page = column_width("double")
    cols = min(n_panels, columns)
    width, row_inches = page, page * row_height
    if slots is not None:
        panel = max(1.6, 0.6 + SLOT_INCHES * slots)
        cols = max(1, min(cols, int(page // panel)))
        width = min(max(page, panel * cols), MAX_WIDTH_PAGES * page)
        row_inches = 1.7 + 0.045 * label_chars * 0.71  # 45-degree labels hang below the axes
    rows = -(-n_panels // cols)
    fig, grid = plt.subplots(rows, cols, figsize=(width, row_inches * rows), squeeze=False)
    axes = list(grid.ravel())
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    if n_panels > 1:
        for i, ax in enumerate(axes[:n_panels]):
            panel_label(ax, panel_letter(i))
    return fig, axes[:n_panels]


def write_chunks(
    out_path: str | Path,
    names: Sequence,
    size: int,
    draw: Callable[[list], plt.Figure],
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Draw and save each chunk of ``names``; returns the first file, ``None`` if no names."""
    out_path, primary = Path(out_path), None
    for k, chunk in enumerate(chunks(list(names), size), start=1):
        target = out_path if k == 1 else out_path.with_name(f"{out_path.stem}_{k}{out_path.suffix}")
        fig = draw(chunk)
        if metadata is not None:
            stamp_figure(fig, metadata)
        written = save_figure(fig, target.parent, target.stem, formats=formats, dpi=dpi)
        plt.close(fig)
        path = written.get(formats[0])
        if metadata is not None and path is not None:
            write_sidecar(path, metadata)
        primary = primary or path
    return primary


__all__ = ["MAX_PANELS", "PANEL_COLUMNS", "SLOT_INCHES", "chunks", "new_grid", "write_chunks"]
