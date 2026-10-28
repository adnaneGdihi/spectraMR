"""Figure 2.15 — IB active-acquisition trajectory (Idea 4 of audit_plan_novel.md).

Plots per-step reconstruction quality (PSNR, dB) versus number of acquired
k-space lines for one or more acquisition policies — typically the
IB-active policy from :class:`IBActiveAcquisitionStrategy` against the
``bald`` and uniform-random baselines.

The figure's data source is the CSV emitted by
``python -m spectramr.cli simulate-acquisition <yaml> --output <csv>``; the
expected columns are ``step``, ``psnr_db``, ``acquired_lines`` and an
optional ``policy`` column for multi-policy overlays.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spectramr.infrastructure.reporting.metadata import (
    RunMetadata,
    stamp_figure,
    write_sidecar,
)
from spectramr.infrastructure.reporting.style import colour_for, use_default_style, save_plot

logger = logging.getLogger(__name__)


def _coerce_dataframe(
    df: pd.DataFrame | None,
    *,
    csv_paths: list[Path] | None,
) -> pd.DataFrame:
    """Return a long-form dataframe with policy/step/psnr columns.

    Two paths are supported:

    - The caller passes a dataframe (the standard plotter signature) with
      a ``method`` or ``policy`` column distinguishing the acquisition
      strategies. We rename ``method → policy`` for plotting.
    - The caller passes ``csv_paths=[...]`` of files emitted by
      ``simulate-acquisition``. Each file's basename becomes the policy.
    """
    if df is not None and "psnr_db" in df.columns and "step" in df.columns:
        out = df.copy()
        if "policy" not in out.columns and "method" in out.columns:
            out = out.rename(columns={"method": "policy"})
        if "policy" not in out.columns:
            out["policy"] = "policy"
        return out[["step", "psnr_db", "policy"]]

    frames: list[pd.DataFrame] = []
    if csv_paths:
        for p in csv_paths:
            try:
                local = pd.read_csv(p)
            except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
                logger.warning("fig_2_15: could not read %s: %s", p, exc)
                continue
            if "psnr_db" not in local.columns or "step" not in local.columns:
                logger.warning(
                    "fig_2_15: %s missing required columns (step, psnr_db); skipping",
                    p,
                )
                continue
            local["policy"] = p.stem
            frames.append(local[["step", "psnr_db", "policy"]])
    if not frames:
        return pd.DataFrame(columns=["step", "psnr_db", "policy"])
    return pd.concat(frames, ignore_index=True)


def make(
    df: pd.DataFrame | None,
    out_path: str | Path,
    *,
    csv_paths: list[str | Path] | None = None,
    formats: tuple[str, ...] | None = None,
    dpi: int = 600,
    metadata: RunMetadata | None = None,
) -> Path | None:
    """Render the active-acquisition trajectory figure.

    Args:
        df: Optional long-form dataframe with columns ``step``,
            ``psnr_db``, and ``policy`` (or ``method``). When ``None``
            or missing the required columns, the plotter falls back to
            ``csv_paths``.
        out_path: Destination PDF.
        csv_paths: Optional list of CSVs (one per acquisition policy)
            emitted by ``simulate-acquisition``.
        metadata: Provenance stamp for ``stamp_figure``.

    Returns:
        The output path on success, or ``None`` when there is no
        plottable data.
    """
    out_path = Path(out_path)
    paths = [Path(p) for p in csv_paths] if csv_paths else None
    data = _coerce_dataframe(df, csv_paths=paths)
    if data.empty:
        logger.info("fig_2_15: no data — skipping")
        return None

    use_default_style()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for policy, sub in data.groupby("policy"):
        sub_sorted = sub.sort_values("step")
        ax.plot(
            sub_sorted["step"].to_numpy(),
            sub_sorted["psnr_db"].to_numpy(),
            label=str(policy),
            color=colour_for(str(policy)),
            linewidth=1.4,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Acquisition step")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("IB active acquisition — reconstruction quality vs. step")
    ax.legend(loc="lower right", fontsize=7, frameon=True)
    ax.grid(True, alpha=0.3)

    if metadata is not None:
        stamp_figure(fig, metadata)
    fig.tight_layout()
    return save_plot(fig, out_path, formats=formats, dpi=dpi, metadata=metadata)


__all__ = ["make"]
