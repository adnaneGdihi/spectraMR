r"""Radial k-space error spectrum.

Azimuthally-averaged |F(pred) − F(target)| as a function of normalised
spatial frequency, log-y. Shows *where in frequency* a method fails —
low-frequency (contrast) vs high-frequency (detail). Uses the physics
SSOT centered FFT (``fft_ops.fft2c``), never raw ``torch.fft``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectramr.infrastructure.reporting.metadata import RunMetadata, stamp_figure, write_sidecar
from spectramr.infrastructure.reporting.style import (
    colour_for,
    column_width,
    save_figure,
    use_default_style,
)


def _radial_profile(err2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = err2d.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), err2d.ravel())
    nbin = np.bincount(r.ravel())
    radial = tbin / np.maximum(nbin, 1)
    # beyond min(h, w) / 2 an annulus is clipped to the corners and holds few samples
    full = int(min(h, w) // 2) + 1
    return np.arange(full) / (full - 1), radial[:full]


def _fft_mag(img: np.ndarray) -> np.ndarray:
    # Centered FFT magnitude via the physics SSOT; fall back to numpy fftshift
    # if torch is unavailable in the report environment.
    try:
        import torch

        from spectramr.infrastructure.physics.fft_ops import fft2c

        t = torch.from_numpy(np.asarray(img, dtype=np.float32))[None, None]
        k = fft2c(t)
        return torch.abs(k)[0, 0].cpu().numpy()
    except Exception:
        k = np.fft.fftshift(np.fft.fft2(np.asarray(img, dtype=np.float64)))
        return np.abs(k)


def make(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    cases=None,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    if not cases:
        return None
    use_default_style()
    fig, ax = plt.subplots(figsize=(column_width("single"), column_width("single") * 0.75))
    for i, case in enumerate(cases):
        err = np.abs(_fft_mag(case["prediction"]) - _fft_mag(case["target"]))
        freq, radial = _radial_profile(err)
        ax.plot(
            freq,
            radial + 1e-8,
            color=colour_for(case.get("rank", str(i)), i),
            label=case.get("rank", f"case {i}"),
        )
    ax.set_yscale("log")
    ax.set_xlabel("normalised spatial frequency")
    ax.set_ylabel("mean |Δ k-space|")
    if len(cases) > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    primary = written.get(formats[0])
    if metadata is not None and primary is not None:
        write_sidecar(primary, metadata)
    return primary
