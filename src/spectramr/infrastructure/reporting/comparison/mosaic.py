r"""Cross-arm mosaic: every arm beside the reference, and every arm's error.

Two rows per reference slice: the reconstructions of each arm next to the
target, then each arm's ``|error|`` on *one* scale shared by the whole figure.
Slices are aligned across arms by target content (:func:`.arms.align_cases`),
so a column never compares two arms on different anatomy. Each slice zooms
where the arms' mean error is largest, and its error maps share one scale with
its own colour bar -- arms are compared within a slice, and a figure-wide scale
set by the highest R would blank the error maps of the lowest.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..metadata import RunMetadata, stamp_figure, write_sidecar
from ..plotters._mosaic import (
    ERROR_CMAP,
    draw_tile,
    error_map,
    shared_vmax,
    to_display,
    window,
    zoom_box,
)
from ..style import column_width, save_figure, use_default_style
from .arms import Arms, _target_key, align_cases


def _rows(arms: Arms, keys: list[str]) -> list[dict[str, dict]]:
    """``[{label: case}]`` per aligned slice."""
    by_key = {lab: {_target_key(c): c for c in cs} for lab, cs in arms.cases.items()}
    return [{lab: by_key[lab][k] for lab in arms.labels} for k in keys]


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    arms: Arms | None = None,
    max_rows: int | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if arms is None or not arms.cases or len(arms.cases) < 2:
        return None
    if max_rows is None:
        n_r = len({c.get("acceleration") for c in next(iter(arms.cases.values()))})
        max_rows = n_r if n_r > 1 else 3  # one slice per R for a cascade
    keys = align_cases(arms.cases, max_rows)
    if not keys:
        return None
    rows = _rows(arms, keys)
    labels = arms.labels
    k = len(labels) + 1
    use_default_style()
    wins = [window(next(iter(r.values()))["target"]) for r in rows]
    errs = [
        {lab: error_map(r[lab]["prediction"], r[lab]["target"], w) for lab in labels}
        for r, w in zip(rows, wins, strict=True)
    ]
    # One scale per slice, shared by every arm on it: a single figure-wide scale
    # is set by the hardest R and turns the low-R error maps black.
    vmaxes = [shared_vmax(list(row.values())) for row in errs]
    width = column_width("double")
    fig, axes = plt.subplots(
        2 * len(rows),
        k,
        figsize=(width, width / k * 2 * len(rows) * 0.92 + 0.4),
        squeeze=False,
        layout="compressed",
    )
    artist = None
    for i, (row, win, err) in enumerate(zip(rows, wins, errs, strict=True)):
        first = next(iter(row.values()))
        target = to_display(first["target"], win)
        box = zoom_box(list(err.values()), target.shape, target)
        draw_tile(axes[2 * i][0], target, box=box, inset=True)
        axes[2 * i + 1][0].set_axis_off()
        for j, lab in enumerate(labels, start=1):
            pred = to_display(row[lab]["prediction"], win)
            draw_tile(axes[2 * i][j], pred, box=box, inset=True)
            artist = draw_tile(
                axes[2 * i + 1][j], err[lab], cmap=ERROR_CMAP, vmax=vmaxes[i], box=box
            )
        accel = first.get("acceleration")
        tag = f"slice {i + 1}" + (f"\nR = {accel:g}" if accel is not None else "")
        axes[2 * i][0].set_ylabel(tag, fontsize="x-small")
        axes[2 * i + 1][1].set_ylabel("|error|", fontsize="x-small")
        fig.colorbar(artist, ax=axes[2 * i + 1, 1:].tolist(), shrink=0.9, pad=0.01, aspect=12)
    size = "small" if k <= 7 else "x-small"
    for j, lab in enumerate(["reference", *labels]):
        axes[0][j].set_title(textwrap.fill(lab.replace("_", " "), 14), fontsize=size)
    steps = sorted(set(arms.steps.values()))
    where = (
        f"validation step {steps[0]}" if len(steps) == 1 else f"each arm at its {arms.view} step"
    )
    note = f"{where}; |error| as a fraction of the reference window"
    fig.suptitle(note, fontsize="small")
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    plt.close(fig)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
