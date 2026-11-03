"""A banded board over one stratum (a contrast), with the whole board's circular set.

A per-contrast board measured on its own few contents could flag a different circular set,
and a mean over contrasts would then average different metric sets. So a stratum board
drops exactly the metrics the all-contents board left out, and carries their reasons.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np

from spectramr.core.metrics.meta_evaluation.banded_leaderboard import (
    BandedResult,
    banded_leaderboard,
)
from spectramr.core.metrics.meta_evaluation.damage import (
    MIN_AXES_PER_BAND,
    MIN_LEVELS_PER_BAND,
    Bands,
)

__all__ = ["stratum_leaderboards"]


def stratum_leaderboards(
    values: Mapping[str, np.ndarray],
    axis: np.ndarray,
    content: np.ndarray,
    damage: Mapping[str, np.ndarray],
    higher_is_better: Mapping[str, bool],
    *,
    bands: Mapping[str, Bands],
    whole: Mapping[str, BandedResult],
    min_levels: int = MIN_LEVELS_PER_BAND,
    min_axes: int = MIN_AXES_PER_BAND,
) -> dict[str, BandedResult]:
    """One :func:`banded_leaderboard` per yardstick, circular set taken from ``whole``."""
    out: dict[str, BandedResult] = {}
    for y, d in damage.items():
        src = whole[y]
        kept = {m: v for m, v in values.items() if m not in src.circular}
        res = banded_leaderboard(
            kept, axis, content, d, higher_is_better, bands[y], min_levels, min_axes
        )
        out[y] = replace(
            res,
            circular={m: r for m, r in src.circular.items() if m in values},
            circular_declared=src.circular_declared,
            circular_measured=src.circular_measured,
            circular_threshold=src.circular_threshold,
        )
    return out
