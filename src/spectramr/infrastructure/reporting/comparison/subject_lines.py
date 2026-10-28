r"""Each subject's effect, arm by arm, as a difference from the baseline.

One panel per metric. A thin grey polyline is one subject (inference: one case): zero
at the baseline, then that subject's value under each arm minus its value under the
baseline. The coloured marker is the mean difference. Parallel lines mean a uniform
effect; fanning or crossing lines mean the effect depends on the subject, which an
averaged difference hides (inter-repetition motion favours a single-acquisition arm on
some subjects only). Differences remove the large between-subject spread, so effects of
a fraction of it stay readable. A reciprocal metric is on the ``1/N_eq`` scale.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..metadata import RunMetadata
from ..style import colour_for
from ._grid import MAX_PANELS, new_grid, write_chunks
from .metric_bars import _label


def subject_differences(paired: dict, phase: str, metric: str, variants: list[str]) -> pd.DataFrame:
    """Subjects by variants: arm minus baseline, ``NaN`` where a subject was not paired."""
    columns = {}
    for variant in variants:
        pair = paired.get((phase, metric, variant))
        if pair is not None:
            columns[variant] = pair.table["a"] - pair.table["b"]
    return pd.DataFrame(columns).reindex(columns=[v for v in variants if v in columns])


def _draw(chunk, paired, arms, baseline, phase):
    variants = [a for a in arms if a != baseline]
    fig, axes = new_grid(len(chunk))
    for ax, metric in zip(axes, chunk, strict=True):
        diffs = subject_differences(paired, phase, metric, variants)
        xs = [0, *[variants.index(v) + 1 for v in diffs.columns]]
        for _, row in diffs.iterrows():
            ax.plot(xs, [0.0, *row.to_numpy(dtype=float)], color="#999999", lw=0.5, alpha=0.5)
        for v in diffs.columns:
            x = variants.index(v) + 1
            colour = colour_for(v, arms.index(v))
            ax.plot(x, diffs[v].mean(), "o", color=colour, mec="black", mew=0.4, ms=5, zorder=4)
        ax.plot(
            xs, [0.0, *diffs.mean().to_numpy(dtype=float)], "-", color="black", lw=1.0, zorder=3
        )
        ax.axhline(0.0, color="#555555", lw=0.6)
        ax.set_xticks([0, *range(1, len(variants) + 1)])
        ax.set_xticklabels([baseline, *variants], rotation=30, ha="right", fontsize=6)
        scale = next((p.scale for k, p in paired.items() if k[1] == metric), "metric")
        ax.set_ylabel(
            _label(metric) + (" (1/N_eq)" if "1/N_eq" in scale else "") + ": arm - baseline"
        )
    fig.suptitle(
        f"{phase}: one line per subject (inference: per case); markers are mean differences",
        fontsize="x-small",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def make(
    df,
    out_path: str | Path,
    *,
    paired: dict | None = None,
    arms: list[str] | None = None,
    baseline: str | None = None,
    phase: str = "val",
    max_panels: int = MAX_PANELS,
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """``paired`` is ``{(phase, metric, variant): PairedSubjects}`` from the contrast statistics."""
    if not paired or not arms or baseline is None:
        return None
    metrics = list(dict.fromkeys(m for (p, m, _), _v in paired.items() if p == phase))
    if not metrics:
        return None
    return write_chunks(
        out_path,
        metrics,
        max_panels,
        lambda chunk: _draw(chunk, paired, arms, baseline, phase),
        formats=formats,
        dpi=dpi,
        metadata=metadata,
    )


__all__ = ["make"]
