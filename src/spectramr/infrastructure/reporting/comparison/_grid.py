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
from ..style import column_width, panel_label, save_figure, use_default_style

#: Panels per figure and per row.
MAX_PANELS = 12
PANEL_COLUMNS = 4


def chunks(names: Sequence, size: int) -> list[list]:
    """``names`` in consecutive groups of at most ``size``."""
    if size < 1:
        raise ValueError(f"chunk size must be at least 1, got {size}")
    return [list(names[i : i + size]) for i in range(0, len(names), size)]


def new_grid(n_panels: int, *, row_height: float = 0.36, columns: int = PANEL_COLUMNS):
    """A figure with ``n_panels`` axes in rows of ``columns``; returns ``(fig, axes)``.

    Axes beyond ``n_panels`` are hidden, and each used axis carries its panel letter
    when there is more than one.
    """
    use_default_style()
    cols = min(n_panels, columns)
    rows = -(-n_panels // cols)
    width = column_width("double")
    fig, grid = plt.subplots(rows, cols, figsize=(width, width * row_height * rows), squeeze=False)
    axes = list(grid.ravel())
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    if n_panels > 1:
        for i, ax in enumerate(axes[:n_panels]):
            panel_label(ax, chr(ord("a") + i))
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


__all__ = ["MAX_PANELS", "PANEL_COLUMNS", "chunks", "new_grid", "write_chunks"]
