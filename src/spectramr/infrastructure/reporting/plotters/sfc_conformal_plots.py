"""Reporting figures for the SFC/conformal + fMRI/MRF 2026 arms.

All four plotters follow the standard ``make(df, out_path, **kwargs)
-> Path | None`` registry contract. Inputs are intentionally permissive
— each plotter accepts either a long-form ``pd.DataFrame`` or a small
set of plain-tensor keyword arguments — so a smoke caller can render
them with synthetic data and the production reporting pipeline can
drive them off a real run's diagnostics CSV.

References:
    [4]  L. V. Ahlfors, *Lectures on Quasiconformal Mappings*, AMS, 2006.
    [18] X. Pennec et al., "A Riemannian framework for tensor
         computing", *Int. J. Comput. Vis.*, 66(1), 2006.
    [27] A. Srivastava et al., "Shape analysis of elastic curves",
         *IEEE TPAMI*, 33(7), 2011.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectramr.infrastructure.reporting.style import save_plot

logger = logging.getLogger(__name__)


def _save(fig, out_path: str | Path, formats=None, dpi: int = 600) -> Path | None:
    fig.tight_layout()
    return save_plot(fig, out_path, formats=formats, dpi=dpi)


# --------------------------------------------------------------------- #
# fig_c1_beltrami_field — visualises a Beltrami μ field (SFC §1/§4).
# --------------------------------------------------------------------- #


def make_beltrami_field(
    df=None,
    out_path: str | Path = "fig_c1_beltrami_field.pdf",
    *,
    mu: np.ndarray | None = None,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    **kwargs,
) -> Path | None:
    """Render a single Beltrami coefficient field as |μ| heat-map +
    phase quiver."""
    if mu is None:
        logger.info("fig_c1: no mu provided — skipping.")
        return None
    if mu.ndim == 3:
        mu = mu[0]
    if mu.ndim != 2:
        logger.warning("fig_c1: expected 2-D mu, got shape %s", mu.shape)
        return None
    fig, (ax_mag, ax_phase) = plt.subplots(1, 2, figsize=(8.0, 3.5))
    im0 = ax_mag.imshow(np.abs(mu), cmap="viridis", vmin=0.0)
    ax_mag.set_title("|μ| (Beltrami magnitude)")
    plt.colorbar(im0, ax=ax_mag, shrink=0.85)
    im1 = ax_phase.imshow(np.angle(mu), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    ax_phase.set_title("arg μ (Beltrami phase)")
    plt.colorbar(im1, ax=ax_phase, shrink=0.85)
    return _save(fig, out_path, formats, dpi)


# --------------------------------------------------------------------- #
# fig_c2_spd_geodesic — geodesic distance trajectory on Sym⁺(n)
# --------------------------------------------------------------------- #


def make_spd_geodesic_trajectory(
    df=None,
    out_path: str | Path = "fig_c2_spd_geodesic.pdf",
    *,
    distances: np.ndarray | None = None,
    step_column: str = "step",
    distance_column: str = "geodesic_fc_error",
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    **kwargs,
) -> Path | None:
    """Render the affine-invariant geodesic distance trajectory across
    training. Either pass a long-form DataFrame with ``step_column`` and
    ``distance_column``, or a 1-D ``distances`` array."""
    if df is not None and step_column in df.columns and distance_column in df.columns:
        steps = df[step_column].to_numpy()
        d = df[distance_column].to_numpy()
    elif distances is not None:
        steps = np.arange(len(distances))
        d = np.asarray(distances)
    else:
        logger.info("fig_c2: no data — skipping.")
        return None
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(steps, d, linewidth=1.6)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Affine-invariant geodesic distance")
    ax.set_title("FC manifold geodesic error")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path, formats, dpi)


# --------------------------------------------------------------------- #
# fig_c3_teich_schedule — Teichmüller schedule r(t) vs. log K(t).
# --------------------------------------------------------------------- #


def make_teichmuller_schedule(
    df=None,
    out_path: str | Path = "fig_c3_teich_schedule.pdf",
    *,
    r: np.ndarray | None = None,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    **kwargs,
) -> Path | None:
    """Render the Teichmüller schedule and the implied log-dilatation
    along the unit interval ``t ∈ [0, 1]``."""
    if r is None and df is not None and "r" in df.columns:
        r = df["r"].to_numpy()
    if r is None:
        logger.info("fig_c3: no schedule — skipping.")
        return None
    r = np.asarray(r).clip(0.0, 0.999)
    t = np.linspace(0.0, 1.0, len(r))
    log_k = np.log((1.0 + r) / np.clip(1.0 - r, 1e-6, None))
    fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(7.5, 3.2))
    ax_r.plot(t, r, linewidth=1.6)
    ax_r.set_xlabel("t")
    ax_r.set_ylabel("r(t)")
    ax_r.set_title("Teichmüller radial parameter")
    ax_r.grid(True, alpha=0.3)
    ax_k.plot(t, log_k, linewidth=1.6, color="tab:orange")
    ax_k.set_xlabel("t")
    ax_k.set_ylabel("log K(t)")
    ax_k.set_title("log maximal dilatation")
    ax_k.grid(True, alpha=0.3)
    return _save(fig, out_path, formats, dpi)


# --------------------------------------------------------------------- #
# fig_c4_fingerprint_embedding — 2-D scatter of MRF fingerprints in
# the learned conformal embedding (MRF §3).
# --------------------------------------------------------------------- #


def make_fingerprint_embedding(
    df=None,
    out_path: str | Path = "fig_c4_fingerprint_embedding.pdf",
    *,
    embedding: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    **kwargs,
) -> Path | None:
    """2-D PCA projection of the 5-D conformal fingerprint embedding.

    Caller supplies an ``[N, 5]`` array and optional integer ``labels``
    for colour coding (e.g. tissue type). A 2-D PCA is computed inline
    so the figure can be rendered without sklearn.
    """
    if embedding is None and df is not None and "embed_0" in df.columns:
        cols = [c for c in df.columns if c.startswith("embed_")]
        embedding = df[cols].to_numpy()
        labels = df.get("label")
    if embedding is None:
        logger.info("fig_c4: no embedding — skipping.")
        return None
    x = np.asarray(embedding) - np.asarray(embedding).mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    proj = u[:, :2] * s[:2]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    if labels is not None:
        sc = ax.scatter(
            proj[:, 0], proj[:, 1], c=np.asarray(labels), cmap="tab10", s=12, alpha=0.85
        )
        plt.colorbar(sc, ax=ax, shrink=0.8, label="class")
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=12, alpha=0.85)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Conformal fingerprint embedding (PCA(5→2))")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path, formats, dpi)


__all__ = [
    "make_beltrami_field",
    "make_fingerprint_embedding",
    "make_spd_geodesic_trajectory",
    "make_teichmuller_schedule",
]
