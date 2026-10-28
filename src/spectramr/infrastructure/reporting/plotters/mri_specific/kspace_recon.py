r"""Figure A.7 — k-space → image reconstruction.

Per-row layout:

    zero-filled | [baseline] | proposed | fully-sampled |
    residual (proposed − FS) + colorbar | residual histogram | radial k-error

The baseline column renders only when the case carries a recorded ``baseline``.

Each magnitude image annotated with PSNR / SSIM / L1 versus FS reference.
The radial k-error panel is on a log-y axis with bin-width-normalised
mean absolute k-space error per |k| bin.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.plotters._comparison_helpers import (
    _format_metric_str,
    add_image_panel,
    add_residual_histogram,
    add_residual_panel,
)
from spectramr.infrastructure.reporting.style import save_figure, use_default_style


def _radial_error(pred: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    from spectramr.infrastructure.physics.fft_ops import fft2c

    diff = torch.from_numpy(np.asarray(pred - ref, dtype=np.complex64))
    err = fft2c(diff).abs().numpy()
    h, w = err.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(err.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    counts = np.bincount(r.ravel()).clip(min=1)
    binned = np.bincount(r.ravel(), err.ravel()) / counts
    return np.arange(len(binned)), binned


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases: Sequence[dict] | None = None,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """Build the k-space reconstruction figure.

    Args:
        cases: dicts with arrays ``zero_filled``, ``pred``, ``fs_ref`` and
            optionally ``baseline`` (all magnitude images, same H x W).
    """
    cases = [c for c in (cases or []) if all(k in c for k in ("zero_filled", "pred", "fs_ref"))]
    if not cases:
        return None
    out_path = Path(out_path)

    use_default_style()
    # The comparison column exists only when the run recorded one. The loader used
    # to alias ``baseline`` onto the zero-filled input, so a column titled
    # "CG-SENSE" showed the input image for a baseline that never ran.
    has_baseline = all("baseline" in c for c in cases)
    images = [("zero_filled", "zero-filled", "#888888")]
    if has_baseline:
        images.append(("baseline", "baseline", "#0072B2"))
    images += [("pred", "proposed", "#D55E00"), ("fs_ref", "fully-sampled", None)]
    titles = [t for _, t, _ in images] + ["residual (ours \u2212 FS)", "residual histogram",
                                           "radial k-error"]
    n_rows, n_cols = len(cases), len(titles)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.6), squeeze=False,
        gridspec_kw={"wspace": 0.22, "hspace": 0.20},
    )
    for r, c in enumerate(cases):
        arrays = {key: np.asarray(c[key]).squeeze().astype(float) for key, _, _ in images}
        ref, pr = arrays["fs_ref"], arrays["pred"]
        for col, (key, title, _) in enumerate(images):
            add_image_panel(
                axes[r][col],
                arrays[key],
                title=title if r == 0 else None,
                row_label=c.get("case_id") if col == 0 else None,
                metric_str=None if key == "fs_ref" else _format_metric_str(arrays[key], ref),
            )
        base = len(images)
        abs_max, residual = add_residual_panel(
            axes[r][base], pr, ref, title=titles[base] if r == 0 else None, fig=fig
        )
        add_residual_histogram(
            axes[r][base + 1], residual, abs_max=abs_max,
            title=titles[base + 1] if r == 0 else None,
        )
        ax = axes[r][base + 2]
        for key, label, colour in images:
            if colour is None:
                continue
            k, e = _radial_error(arrays[key], ref)
            ax.semilogy(k[1:], e[1:].clip(min=1e-12), color=colour, linewidth=0.9, label=label)
        if r == 0:
            ax.set_title(titles[-1])
            ax.legend(frameon=False, loc="upper right")
        ax.set_xlabel("|k|")
        ax.set_ylabel("|Δk|")

    if metadata is not None:
        stamp_figure(fig, metadata)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
