r"""Shared helpers for image-comparison figures (SR / C2C / kspace recon /
failure gallery / etc.).

Three building blocks are exposed:

* :func:`add_image_panel` — show an image with consistent fontsize and
  optional per-image PSNR / SSIM / L1 annotation.
* :func:`add_residual_panel` — show a residual map with a *symmetric*
  colormap, an attached colorbar, and a per-image L1 annotation.
* :func:`add_residual_histogram` — show the residual-value distribution
  alongside the residual map; reports L1 mean, residual std, and
  signed bias.

Design rationale
----------------
The previous reporting plotters used 5–7 pt fonts, no colorbars on
residual columns, an arbitrary residual-scale multiplier, and no
per-row metric annotations. Reviewers cannot judge the magnitude of a
predicted residual from a colour map alone — they need either (a) a
colorbar tied to the absolute residual range, or (b) the residual
histogram with the L1 mean called out. This module supplies both.
"""

from __future__ import annotations

import numpy as np

# Per-IEEE-style baseline — readable on a single-column figure.
_FS_TITLE = 8  # column titles
_FS_ROW = 7  # row labels
_FS_NOTE = 7  # in-axes annotation (e.g. PSNR=…)
_FS_TICK = 6  # colorbar ticks


# ──────────────────────────────────────────────────────────────────────
# Numerical helpers (paradigm-agnostic, work on any 2-D arrays)
# ──────────────────────────────────────────────────────────────────────


def l1(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.abs(pred - ref).mean())


def psnr(pred: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((pred - ref) ** 2))
    if mse < 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim_simple(pred: np.ndarray, ref: np.ndarray) -> float:
    """Single-value SSIM approximation (no sliding window).

    Sufficient for in-figure annotation; full SSIM should use the
    :mod:`spectramr.core.metrics` registry. Returns the standard luminance ×
    contrast × structure product on the entire image.
    """
    mu_x, mu_y = float(pred.mean()), float(ref.mean())
    sx, sy = float(pred.var()), float(ref.var())
    sxy = float(((pred - mu_x) * (ref - mu_y)).mean())
    L = 1.0
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sxy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sx + sy + c2)
    if den < 1e-12:
        return 0.0
    return float(num / den)


def _format_metric_str(
    pred: np.ndarray | None,
    ref: np.ndarray | None,
    *,
    show: tuple[str, ...] = ("psnr", "ssim", "l1"),
    multiline: bool = True,
) -> str:
    """Compose a compact metric string for in-axes annotation.

    With ``multiline=True`` (default) the metrics stack on separate
    lines so the box stays narrow enough for column-bound figures.
    """
    if pred is None or ref is None:
        return ""
    parts: list[str] = []
    if "psnr" in show:
        try:
            # Peak = max|ref|, the convention fiducial_check uses; a fixed 1.0
            # printed negative dB on every magnitude image outside [0, 1].
            peak = float(np.abs(ref).max()) or 1.0
            parts.append(f"PSNR {psnr(pred, ref, data_range=peak):.1f}")
        except Exception:
            pass
    if "ssim" in show:
        try:
            parts.append(f"SSIM {ssim_simple(pred, ref):.3f}")
        except Exception:
            pass
    if "l1" in show:
        parts.append(f"L1 {l1(pred, ref):.4f}")
    return ("\n" if multiline else "  ").join(parts)


# ──────────────────────────────────────────────────────────────────────
# Panel builders
# ──────────────────────────────────────────────────────────────────────


def add_image_panel(
    ax,
    image: np.ndarray,
    *,
    title: str | None = None,
    row_label: str | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    metric_str: str = "",
) -> None:
    """Render an image with consistent style + optional per-image metric.

    When ``vmin`` / ``vmax`` are not provided, the panel uses per-image
    [0.5, 99.5] percentile windowing. The previous defaults of
    ``vmin=0.0, vmax=1.0`` produced black panels for any caller that
    forgot to pre-normalize (e.g. raw IFFT-magnitude tensors with values
    in the hundreds — the May 2026 black-image regression).
    """
    if vmin is None or vmax is None:
        finite = image[np.isfinite(image)] if image.size else image
        if finite.size:
            lo = float(np.quantile(finite, 0.005))
            hi = float(np.quantile(finite, 0.995))
            if hi - lo < 1e-8:
                # Constant image → centre the colormap on the value
                lo, hi = lo - 0.5, hi + 0.5
        else:
            lo, hi = 0.0, 1.0
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=_FS_TITLE)
    if row_label is not None:
        ax.set_ylabel(row_label, fontsize=_FS_ROW)
    if metric_str:
        ax.text(
            0.02,
            0.98,
            metric_str,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_FS_NOTE,
            color="white",
            bbox=dict(
                facecolor="black", edgecolor="none", alpha=0.55, pad=1.5, boxstyle="round,pad=0.2"
            ),
        )


def add_residual_panel(
    ax,
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "RdBu_r",
    abs_max: float | None = None,
    show_l1_annotation: bool = True,
    add_colorbar: bool = True,
    fig=None,
) -> tuple[float, np.ndarray]:
    """Show signed residual with a *symmetric* colormap + colorbar.

    Returns ``(abs_max_used, residual)`` so callers can chain a
    histogram panel with the same scale.
    """
    residual = pred - ref
    if abs_max is None:
        abs_max = float(np.abs(residual).max())
        if abs_max < 1e-6:
            abs_max = 1e-6

    im = ax.imshow(
        residual,
        cmap=cmap,
        vmin=-abs_max,
        vmax=abs_max,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=_FS_TITLE)

    if show_l1_annotation:
        ax.text(
            0.02,
            0.98,
            f"L1={l1(pred, ref):.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_FS_NOTE,
            color="black",
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.75, pad=1.5, boxstyle="round,pad=0.2"
            ),
        )

    if add_colorbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=_FS_TICK)
        cb.outline.set_visible(False)

    return abs_max, residual


def add_residual_histogram(
    ax,
    residual: np.ndarray,
    *,
    abs_max: float | None = None,
    title: str | None = None,
    bins: int = 60,
    color: str = "#56B4E9",
) -> None:
    """Histogram of residual values with mean L1 / std / bias annotations.

    Pairs naturally with :func:`add_residual_panel` (use the same
    ``abs_max`` for shared x-axis range).
    """
    flat = residual.ravel().astype(np.float64)
    if abs_max is None:
        abs_max = float(np.abs(flat).max())
        if abs_max < 1e-6:
            abs_max = 1e-6

    ax.hist(
        flat,
        bins=bins,
        range=(-abs_max, abs_max),
        color=color,
        edgecolor="none",
        alpha=0.85,
    )
    ax.axvline(0.0, color="#444", lw=0.6, ls="--")
    mean = float(flat.mean())  # signed bias
    sd = float(flat.std())
    l1_val = float(np.abs(flat).mean())
    # Annotation in top-RIGHT to avoid overlap with the residual-map's
    # colorbar which sits on the left edge of this column.
    ax.text(
        0.98,
        0.98,
        f"L1 {l1_val:.4f}\nbias {mean:+.4f}\nσ {sd:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=_FS_NOTE,
        bbox=dict(
            facecolor="white", edgecolor="none", alpha=0.85, pad=1.8, boxstyle="round,pad=0.2"
        ),
    )

    ax.set_xlabel("residual", fontsize=_FS_NOTE)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=_FS_TICK)
    ax.set_xlim(-abs_max, abs_max)
    if title is not None:
        ax.set_title(title, fontsize=_FS_TITLE)
    # Keep frame minimal
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


__all__ = [
    "add_image_panel",
    "add_residual_histogram",
    "add_residual_panel",
    "l1",
    "psnr",
    "ssim_simple",
]
