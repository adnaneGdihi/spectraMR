r"""Shared machinery for image mosaics: windows, error maps, zoom, annotation.

Every qualitative figure -- the per-run reconstruction panel and the cross-run
comparison mosaic -- needs the same four decisions made the same way, or two
figures of one run disagree about what an error of 0.1 looks like:

* **Display window.** Prediction and target share the *target's* window, so the
  grey levels are comparable; normalising each tile to its own percentiles
  hides a global gain error entirely.
* **Error scale.** ``|pred - target|`` as a fraction of that window, with one
  ``vmax`` shared by every tile in the figure and stated on the colour bar.
* **Zoom.** One region per row, placed where the row's error is largest, boxed
  on the full image and magnified in an inset.
* **Annotation.** PSNR/SSIM per tile, read from the recorded metrics when the
  run logged them, never re-derived with a different formula.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage


#: Percentiles of the target that set the display window.
WINDOW_PCT: tuple[float, float] = (0.5, 99.5)
ERROR_CMAP = "magma"
ZOOM_FRACTION = 0.25
TISSUE_FLOOR = 0.15
INSET_BOUNDS = (0.56, 0.02, 0.42, 0.42)
_ZOOM_COLOUR = "#E69F00"


def window(reference: np.ndarray) -> tuple[float, float]:
    """``(lo, hi)`` display window from the reference image's percentiles."""
    ref = np.asarray(reference, dtype=float)
    lo, hi = np.percentile(ref, WINDOW_PCT)
    if not np.isfinite(hi - lo) or hi <= lo:
        lo, hi = float(ref.min()), float(ref.max()) + 1e-8
    return float(lo), float(hi)


def to_display(image: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    lo, hi = win
    return np.clip((np.asarray(image, dtype=float) - lo) / (hi - lo), 0.0, 1.0)


def error_map(pred: np.ndarray, ref: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    """``|pred - ref|`` in units of the display window (1.0 = full window)."""
    lo, hi = win
    return np.abs(np.asarray(pred, dtype=float) - np.asarray(ref, dtype=float)) / (hi - lo)


def shared_vmax(errors: list[np.ndarray], pct: float = 99.0) -> float:
    """One error ceiling for the whole figure, rounded up to a readable value."""
    peaks = [float(np.percentile(e, pct)) for e in errors if e.size]
    top = max(peaks) if peaks else 1.0
    if top <= 0:
        return 1.0
    for nice in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0):
        if top <= nice:
            return nice
    return float(np.ceil(top))


def zoom_box(
    errors: list[np.ndarray],
    shape: tuple[int, int],
    reference: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """``(r0, r1, c0, c1)`` of the square where the row's mean error peaks.

    With ``reference`` (display units) the square must be mostly tissue: a noisy
    reference leaves a uniform error floor over the background that would
    otherwise win the argmax and zoom into empty space.
    """
    h, w = shape
    side = max(8, round(min(h, w) * ZOOM_FRACTION))
    if not errors:
        r0, c0 = (h - side) // 2, (w - side) // 2
        return r0, r0 + side, c0, c0 + side
    mean_err = np.mean([e for e in errors if e.shape == shape], axis=0)
    density = ndimage.uniform_filter(mean_err, size=side, mode="constant")
    if reference is not None and np.shape(reference) == shape:
        covered = ndimage.uniform_filter(
            (np.asarray(reference) > TISSUE_FLOOR).astype(float), size=side, mode="constant"
        )
        for need in (0.8, 0.5):
            if np.any(covered >= need):
                density = np.where(covered >= need, density, -1.0)
                break
    half = side // 2
    density[:half, :] = density[h - half :, :] = -1
    density[:, :half] = density[:, w - half :] = -1
    rc, cc = np.unravel_index(np.argmax(density), density.shape)
    r0, c0 = int(rc) - half, int(cc) - half
    return r0, r0 + side, c0, c0 + side


def draw_tile(
    ax,
    image: np.ndarray,
    *,
    cmap: str = "gray",
    vmax: float = 1.0,
    box: tuple[int, int, int, int] | None = None,
    inset: bool = False,
):
    """Draw one tile; returns the image artist (for a shared colour bar)."""
    artist = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if box is not None:
        r0, r1, c0, c1 = box
        ax.add_patch(
            Rectangle(
                (c0 - 0.5, r0 - 0.5),
                c1 - c0,
                r1 - r0,
                fill=False,
                edgecolor=_ZOOM_COLOUR,
                linewidth=0.6,
            )
        )
        if inset:
            axin = ax.inset_axes(INSET_BOUNDS)
            axin.imshow(
                image[r0:r1, c0:c1], cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest"
            )
            axin.set_xticks([])
            axin.set_yticks([])
            for spine in axin.spines.values():
                spine.set_edgecolor(_ZOOM_COLOUR)
                spine.set_linewidth(0.8)
    return artist


__all__ = [
    "ERROR_CMAP",
    "draw_tile",
    "error_map",
    "shared_vmax",
    "to_display",
    "window",
    "zoom_box",
]
