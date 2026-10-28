r"""One metric per arm on a number line, against named reference values.

For the NEX-equivalent the references are what an acquisition would reach: the identity
(1), two averages (2), three averages (3). Each arm is a mean with a bootstrap 95 %
interval over subjects (inference: cases), with the subject values behind it; arms are
sorted best first by the metric's declared direction. A reciprocal metric is averaged on
the ``1/N_eq`` scale, which is the unbiased one, and shown back on the metric's own scale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spectramr.core.metrics.metric_directions import resolve_direction
from spectramr.core.metrics.reciprocal import is_reciprocal

from ..metadata import RunMetadata, stamp_figure, write_sidecar
from ..plotters._stats import bootstrap_ci
from ..style import colour_for, column_width, pretty_label, save_figure, use_default_style


def arm_values(paired: dict, phase: str, metric: str, baseline: str) -> dict:
    """``{arm: subject values}`` on the analysis scale, from the paired subject tables."""
    values: dict = {}
    for (p, m, variant), pair in paired.items():
        if (p, m) != (phase, metric):
            continue
        values[variant] = pair.table["a"].to_numpy(dtype=float)
        values.setdefault(baseline, pair.table["b"].to_numpy(dtype=float))
    return values


def on_metric_scale(values: np.ndarray, metric: str) -> tuple[float, float, float]:
    """Mean and 95 % interval, back on the metric's own scale (``1/x`` for a reciprocal)."""
    mean = float(values.mean())
    lo, hi = bootstrap_ci(values) if values.size > 1 else (np.nan, np.nan)
    if not is_reciprocal(metric):
        return mean, float(lo), float(hi)
    with np.errstate(divide="ignore"):
        inv = lambda x: np.inf if x <= 0 else 1.0 / x  # noqa: E731 - one-line scale map
        return inv(mean), inv(hi), inv(lo)


def make(
    df,
    out_path: str | Path,
    *,
    paired: dict | None = None,
    arms: list[str] | None = None,
    baseline: str | None = None,
    ladder=None,
    phase: str = "val",
    formats=("pdf", "png"),
    dpi: int = 600,
    metadata: RunMetadata | None = None,
    **_ignored,
) -> Path | None:
    """The ladder of ``ladder.metric``; ``None`` if no arm has values for it at ``phase``."""
    if not paired or not arms or baseline is None or ladder is None:
        return None
    raw = arm_values(paired, phase, ladder.metric, baseline)
    raw = {a: v[np.isfinite(v)] for a, v in raw.items()}
    raw = {a: v for a, v in raw.items() if v.size}
    if not raw:
        return None
    stats = {a: on_metric_scale(v, ladder.metric) for a, v in raw.items()}
    higher = resolve_direction(ladder.metric)
    order = sorted(stats, key=lambda a: stats[a][0], reverse=bool(higher))
    use_default_style()
    fig, ax = plt.subplots(figsize=(column_width("double") * 0.7, 0.4 * len(order) + 1.5))
    for y, arm in enumerate(order):
        colour = colour_for(arm, arms.index(arm) if arm in arms else y)
        mean, lo, hi = stats[arm]
        if is_reciprocal(ladder.metric):
            with np.errstate(divide="ignore"):
                points = np.where(raw[arm] > 0, 1.0 / raw[arm], np.nan)
        else:
            points = raw[arm]
        jitter = np.random.default_rng(y).uniform(-0.15, 0.15, points.size)
        ax.scatter(points, y + jitter, s=3, color=colour, alpha=0.3, linewidths=0)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(y, lo, hi, color=colour, lw=1.6)
        ax.plot(mean, y, "o", color=colour, mec="black", mew=0.4, ms=5, zorder=4)
    for k, (name, value) in enumerate(ladder.references.items()):
        ax.axvline(value, color="#777777", ls=":", lw=0.9)
        ax.annotate(
            name,
            (value, 1.0),
            xycoords=ax.get_xaxis_transform(),
            xytext=(2, -2 - 7 * (k % 2)),
            textcoords="offset points",
            fontsize=5,
            va="top",
            color="#555555",
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    ax.invert_yaxis()
    arrow = "" if higher is None else (" ↑" if higher else " ↓")
    ax.set_xlabel(pretty_label(ladder.metric) + arrow)
    fig.suptitle(
        f"{phase}: mean and bootstrap 95 % interval over subjects; best first", fontsize="x-small"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if metadata is not None:
        stamp_figure(fig, metadata)
    out_path = Path(out_path)
    written = save_figure(fig, out_path.parent, out_path.stem, formats=formats, dpi=dpi)
    plt.close(fig)
    path = written.get(formats[0])
    if metadata is not None and path is not None:
        write_sidecar(path, metadata)
    return path


__all__ = ["arm_values", "make", "on_metric_scale"]
