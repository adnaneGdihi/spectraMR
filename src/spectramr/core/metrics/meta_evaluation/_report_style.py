"""Package-local figure-styling SSOT for the meta-evaluation report.

This module exists because the layer-direction rule (CLAUDE.md
non-negotiable #5) forbids ``src/`` from importing
``scripts/sim2rank/style.py``. Rather than re-inlining a palette and a
top-3/bottom-3 helper into every renderer in :mod:`figures`, the shared
look-and-feel lives here.

Two design choices worth knowing:

* **Constrained layout, not ``tight_layout``.** Every figure is built via
  :func:`figure`, which sets ``layout="constrained"``. Constrained layout
  reconciles colorbars, ``suptitle``, and external legends on its own —
  so we never call ``fig.tight_layout()`` after adding a colorbar, which
  was the source of the "Axes that are not compatible with tight_layout"
  warnings in the old code.
* **One save site.** :func:`save_figure` is the only place that calls
  ``savefig``, so DPI and the bbox policy are set in exactly one location.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

# Probed without importing: pyplot costs ~0.5 s and is needed only to draw.
MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None


def pyplot() -> Any:
    """Import ``matplotlib.pyplot`` on the headless Agg backend, on first use."""
    _require_matplotlib()
    import matplotlib

    matplotlib.use("Agg")  # must precede the pyplot import
    import matplotlib.pyplot as plt

    return plt


def _require_matplotlib() -> None:
    """Raise a clear error when a figure is built without matplotlib.

    matplotlib is a REQUIRED dependency in pyproject; this guard only fires
    in a degraded cluster env. Raising at figure-construction time (not
    import time) keeps the module importable so the rest of
    ``meta_evaluation`` — and walk-discovery — is unaffected.
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "The meta-evaluation report figures require matplotlib (pip install matplotlib)."
        )


# ---------------------------------------------------------------------------
# Palette — editorial, matches scripts/sim2rank/style.py by value so the
# two reporting surfaces look identical without a forbidden import.
# ---------------------------------------------------------------------------

ACCENT = "#D97A3A"  # top-k emphasis (editorial orange)
BAR = "#3B6E8F"  # default / bottom-k bars (editorial blue)
DIM_GREY = "#C0C4C9"  # de-emphasised bulk
DEFENSIBLE_GREEN = "#2C7A3D"  # passes the defensibility rule
FAIL_RED = "#A93030"  # fails the defensibility rule

DIVERGING_CMAP = "RdBu_r"  # correlation / signed heatmaps
SEQUENTIAL_CMAP = "viridis"  # magnitude heatmaps

DEFAULT_DPI = 150

# Curated display names for metrics whose title-cased form would be wrong
# (acronyms, hyphenation). Anything not here falls back to a cleaned-up
# title case, so the map only needs the exceptions.
_DISPLAY_NAMES: dict[str, str] = {
    "psnr": "PSNR",
    "robust_mri_psnr": "Robust-MRI-PSNR",
    "ssim": "SSIM",
    "ms_ssim": "MS-SSIM",
    "cw_ssim": "CW-SSIM",
    "clinical_ssim": "Clinical-SSIM",
    "fsim": "FSIM",
    "gmsd": "GMSD",
    "ms_gmsd": "MS-GMSD",
    "lpips": "LPIPS",
    "dists": "DISTS",
    "vif": "VIF",
    "vifp": "VIFp",
    "vsi": "VSI",
    "mad": "MAD",
    "mse": "MSE",
    "mae": "MAE",
    "rmse": "RMSE",
    "nmse": "NMSE",
    "nrmse": "NRMSE",
    "hfen": "HFEN",
    "complex_hfen": "Complex-HFEN",
    "fid": "FID",
    "kid": "KID",
    "med_fid": "MedFID",
    "frd": "FRD",
    "rfs": "RFS",
    "niqe": "NIQE",
    "brisque": "BRISQUE",
    "nema_snr": "NEMA-SNR",
    "cc_snr": "CC-SNR",
    "g_factor": "g-factor",
    "snr": "SNR",
    "cnr": "CNR",
    "cjv": "CJV",
    "iou": "IoU",
    "hd95": "HD95",
    "phase_mse": "Phase-MSE",
    "ipen": "IPEN",
    "auroc": "AUROC",
    "auprc": "AUPRC",
}


def human_label(metric_key: str) -> str:
    """Registry key -> presentable display name.

    Known acronyms are mapped exactly; everything else has underscores
    replaced by spaces and is title-cased. A metric is never dropped — an
    unknown key still produces a readable label.
    """
    key = metric_key.strip()
    if key in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[key]
    return key.replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# Top-3 / bottom-3 selection (was inlined in figures.py)
# ---------------------------------------------------------------------------


def top_bottom_indices(values: list[float], k: int = 3) -> tuple[list[int], list[int]]:
    """Indices of the top-``k`` and bottom-``k`` finite values.

    NaN entries are ignored. ``top`` is ordered best-first; ``bottom`` is
    ordered worst-first.
    """
    finite = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
    if not finite:
        return [], []
    ordered = sorted(finite, key=lambda iv: iv[1])
    bottom = [i for i, _ in ordered[: min(k, len(ordered))]]
    top = [i for i, _ in ordered[max(0, len(ordered) - k) :][::-1]]
    return top, bottom


def annotate_top_bottom(
    ax,
    values: list[float],
    *,
    k: int = 3,
    orientation: str = "h",
    fmt: str = "{:.2f}",
    fontsize: int = 8,
) -> tuple[set[int], set[int]]:
    """Label only the top-``k`` / bottom-``k`` bars; return the index sets.

    ``orientation="h"`` writes the value to the right of a horizontal bar at
    height ``i``; ``orientation="v"`` writes it above/below a vertical bar at
    position ``i``. The caller owns bar coloring (it varies per figure); this
    helper only places the numeric labels so the rule is applied identically
    everywhere.
    """
    top, bottom = top_bottom_indices(values, k=k)
    top_set, bottom_set = set(top), set(bottom)
    finite = [v for v in values if math.isfinite(v)]
    span = (max(finite) - min(finite)) if len(finite) >= 2 else 1.0
    pad = max(span, 1e-9) * 0.015
    for i in sorted(top_set | bottom_set):
        v = values[i]
        if not math.isfinite(v):
            continue
        weight = "bold" if i in top_set else "normal"
        if orientation == "h":
            ax.text(
                v + pad,
                i,
                fmt.format(v),
                va="center",
                fontsize=fontsize,
                fontweight=weight,
                color="black",
            )
        else:
            ax.text(
                i,
                v + pad if v >= 0 else v - pad,
                fmt.format(v),
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=fontsize,
                fontweight=weight,
                color="black",
            )
    return top_set, bottom_set


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def barh_figsize(
    n_items: int,
    *,
    width: float = 7.0,
    min_h: float = 2.5,
    per_item: float = 0.34,
    pad: float = 1.0,
) -> tuple[float, float]:
    """Figure size for a horizontal-bar chart of ``n_items`` rows."""
    height = max(min_h, per_item * max(n_items, 1) + pad)
    return (width, height)


# ---------------------------------------------------------------------------
# Figure factory + single save site
# ---------------------------------------------------------------------------


def figure(*, figsize: tuple[float, float] = (7.0, 4.5), polar: bool = False):
    """Create a constrained-layout figure + axis.

    Constrained layout handles colorbars / suptitles / external legends
    without the ``tight_layout`` warning class. Use this everywhere instead
    of a bare ``plt.subplots`` + ``fig.tight_layout()``.
    """
    subplot_kw = {"projection": "polar"} if polar else None
    fig, ax = pyplot().subplots(figsize=figsize, layout="constrained", subplot_kw=subplot_kw)
    return fig, ax


def figure_grid(rows: int, cols: int, *, figsize: tuple[float, float]):
    """Constrained-layout subplot grid (always 2-D ``axes``)."""
    fig, axes = pyplot().subplots(rows, cols, figsize=figsize, layout="constrained", squeeze=False)
    return fig, axes


def save_figure(fig, path, *, dpi: int = DEFAULT_DPI) -> Path:
    """The single ``savefig`` site. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # No bbox_inches="tight": constrained layout already trims; combining the
    # two re-introduces the very clipping/overlap we removed.
    fig.savefig(path, dpi=dpi)
    pyplot().close(fig)
    return path
