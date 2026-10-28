"""Pareto plot for the learnable-acquisition shootout.

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-11.

Thin wrapper that converts the per-arm result list to a long-format
DataFrame and delegates to the canonical Pareto plotter
(:func:`spectramr.infrastructure.reporting.plotters.headline_pareto.make`).

The wrapper exists so callers don't have to know the long-format
schema by hand; it's purely a convenience over the SSOT plotter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from spectramr.infrastructure.reporting.plotters import headline_pareto


@dataclass
class ArmResult:
    name: str
    sampling_budget: float  # 1/R
    psnr: float
    ssim: float | None = None


def _to_long_dataframe(arms: list[ArmResult]) -> pd.DataFrame:
    """Convert ArmResult list to the long-format the SSOT plotter expects."""
    rows: list[dict[str, str | float]] = []
    for arm in arms:
        rows.append({"method": arm.name, "metric": "psnr", "value": arm.psnr})
        rows.append({"method": arm.name, "metric": "sampling_budget", "value": arm.sampling_budget})
        if arm.ssim is not None:
            rows.append({"method": arm.name, "metric": "ssim", "value": arm.ssim})
    return pd.DataFrame(rows)


def render_learnable_acquisition_pareto(
    out_path: str | Path,
    arms: Iterable[ArmResult],
    title: str = "Learnable-acquisition shootout — Pareto",
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
) -> Path | None:
    """Delegates to ``headline_pareto.make`` after schema conversion."""
    df = _to_long_dataframe(list(arms))
    return headline_pareto.make(
        df,
        out_path,
        metric="psnr",
        cost="sampling_budget",
        maximise_metric=True,
        title=title,
        formats=formats,
        dpi=dpi,
    )


__all__ = ["ArmResult", "render_learnable_acquisition_pareto"]
