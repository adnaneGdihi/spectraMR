"""Figure 4.3 — the validation sample at evenly spaced passes of training.

For each pass: the target, the prediction and its error, from the PNG pair that pass writes.
The validation loader shuffles, so a pass's ``_000`` image is a different slice from the
previous pass's unless the run fixes it: each column therefore carries its own target and the
error is always against that column's target. Needs no checkpoint and no model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.cases.loader import _STEP_RE, _find_image_dirs, _read_gray
from spectramr.infrastructure.reporting.metadata import RunMetadata
from spectramr.infrastructure.reporting.plotters import _mosaic
from spectramr.infrastructure.reporting.style import column_width, save_plot, use_default_style

MAX_PASSES = 6


def pass_pairs(run_dir: str | Path) -> list[tuple[int, Path, Path]]:
    """``(step, target png, prediction png)`` of the first slice of each pass, in step order."""
    real_dir, fake_dir = _find_image_dirs(Path(run_dir))
    if real_dir is None or fake_dir is None:
        return []
    first: dict[int, Path] = {}
    for path in sorted(real_dir.glob("*_real_000.png")):
        match = _STEP_RE.search(path.name)
        if match:
            first[int(match.group(2))] = path  # a replayed pass keeps its later set
    pairs = []
    for step in sorted(first):
        fake = fake_dir / first[step].name.replace("_real_", "_fake_")
        if fake.exists():
            pairs.append((step, first[step], fake))
    return pairs


def thin(items: list, limit: int) -> list:
    """At most ``limit`` evenly spaced items, always keeping the first and the last."""
    if len(items) <= limit:
        return items
    keep = np.unique(np.linspace(0, len(items) - 1, limit).round().astype(int))
    return [items[i] for i in keep]


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if run_dir is None:
        return None
    pairs = thin(pass_pairs(run_dir), MAX_PASSES)
    if len(pairs) < 2:
        return None
    targets = [_read_gray(real) for _, real, _ in pairs]
    preds = [_read_gray(fake) for _, _, fake in pairs]
    if any(p.shape != t.shape for t, p in zip(targets, preds, strict=True)):
        return None
    errors = [np.abs(p - t) for t, p in zip(targets, preds, strict=True)]
    vmax = _mosaic.shared_vmax(errors)
    use_default_style()
    import matplotlib.pyplot as plt

    n = len(pairs)
    width = column_width("double")
    fig, axes = plt.subplots(
        3, n, figsize=(width, width / n * 3.2), squeeze=False, layout="compressed"
    )
    artist = None
    for j, ((step, _, _), t, p, e) in enumerate(zip(pairs, targets, preds, errors, strict=True)):
        _mosaic.draw_tile(axes[0][j], t)
        _mosaic.draw_tile(axes[1][j], p)
        artist = _mosaic.draw_tile(axes[2][j], e, cmap=_mosaic.ERROR_CMAP, vmax=vmax)
        axes[0][j].set_title(f"{step // 1000}k" if step >= 1000 else str(step), fontsize="x-small")
    for row, name in enumerate(("target", "prediction", "|error|")):
        axes[row][0].set_ylabel(name, fontsize="x-small")
    fig.colorbar(artist, ax=axes[2, :].tolist(), shrink=0.9, pad=0.01, aspect=12)
    primary = save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)
    plt.close(fig)
    return primary
