r"""Headline qualitative reconstruction panel.

Rows = recorded cases; columns = [input, prediction, target, |error|]. The input
column appears only when the run recorded a real input -- a target copied into
it would show the answer under the label "input". Prediction and target share
the target's display window; the error column shares one scale across rows,
stated on the colour bar; each row zooms where its error is largest
(:mod:`.._mosaic`).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters._mosaic import (
    ERROR_CMAP,
    draw_tile,
    error_map,
    shared_vmax,
    to_display,
    window,
    zoom_box,
)
from spectramr.infrastructure.reporting.style import column_width, save_figure, use_default_style


def _row_label(case: dict, r: int) -> str:
    rank = case.get("rank")
    cid = str(case.get("case_id", f"case {r}"))
    return f"{rank}\n{cid}" if rank and rank != "case" else cid


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases=None,
    max_rows: int = 6,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    cases = [c for c in (cases or []) if "prediction" in c and "target" in c][:max_rows]
    if not cases:
        return None
    use_default_style()
    has_input = all("input" in c for c in cases)
    titles = (["input"] if has_input else []) + ["prediction", "target", "|error|"]
    wins = [window(c["target"]) for c in cases]
    errors = [error_map(c["prediction"], c["target"], w) for c, w in zip(cases, wins, strict=True)]
    vmax = shared_vmax(errors)
    n, k = len(cases), len(titles)
    width = column_width("double")
    # "compressed" packs fixed-aspect image axes and the colour bar without the
    # slack a gridspec leaves around them.
    fig, axes = plt.subplots(
        n, k, figsize=(width, width / k * n * 0.86 + 0.3), squeeze=False, layout="compressed"
    )
    artist = None
    for r, (case, win, err) in enumerate(zip(cases, wins, errors, strict=True)):
        box = zoom_box([err], err.shape, to_display(case["target"], win))
        pred = to_display(case["prediction"], win)
        target = to_display(case["target"], win)
        tiles = []
        if has_input:
            tiles.append((to_display(case["input"], window(case["input"])), {}))
        tiles.append((pred, {"box": box, "inset": True}))
        tiles.append((target, {"box": box, "inset": True}))
        for c, (img, kw) in enumerate(tiles):
            draw_tile(axes[r][c], img, **kw)
        artist = draw_tile(axes[r][k - 1], err, cmap=ERROR_CMAP, vmax=vmax, box=box)
        axes[r][0].set_ylabel(_row_label(case, r), fontsize="x-small", rotation=0,
                              ha="right", va="center")
    for c, title in enumerate(titles):
        axes[0][c].set_title(title)
    fig.colorbar(artist, ax=axes[:, k - 1].tolist(), shrink=0.8, pad=0.02,
                 label="|error| / target window")
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    plt.close(fig)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
