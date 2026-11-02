"""The composite rank over the banded scores: τ, discriminability and sensitivity.

Each component is ranked on its own (competition ranks, ties shared), and the composite is
the weighted mean of those ranks, so no component's units leak into another. Scores are
rounded before ranking so float noise never splits a tie.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "COMPONENTS",
    "competition_ranks",
    "composite_score",
    "direction_ambiguous",
    "parse_composite_weights",
    "pool_strata",
    "sensitivity_table",
]

COMPONENTS: tuple[str, ...] = ("tau", "discriminability", "sensitivity")
_ROUND = 9


def parse_composite_weights(text: str | None) -> dict[str, float]:
    """``"tau=2,discriminability=1,sensitivity=1"`` -> weights summing to 1 (default 1/3 each).

    An omitted component weighs 0. An unknown or repeated key, a negative or non-finite
    value, or all-zero weights raise ``ValueError``.
    """
    if text is None or not text.strip():
        raw = dict.fromkeys(COMPONENTS, 1.0)
    else:
        raw, seen = dict.fromkeys(COMPONENTS, 0.0), set()
        for part in text.split(","):
            key, sep, val = (s.strip() for s in part.partition("="))
            if not sep or key not in COMPONENTS or key in seen:
                raise ValueError(
                    f"--composite-weights: unknown or repeated entry {part!r}; "
                    f"expected key=value with keys from {COMPONENTS}"
                )
            x = float(val)
            if not math.isfinite(x) or x < 0:
                raise ValueError(f"--composite-weights: {key} is negative or non-finite ({val})")
            raw[key] = x
            seen.add(key)
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("--composite-weights: at least one weight must be positive")
    return {k: v / total for k, v in raw.items()}


def competition_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 1 for the highest score over the last axis, ties shared (1, 1, 3).

    NaN is unranked; ``-inf`` ranks after every finite score.
    """
    x = np.round(np.asarray(scores, np.float64), _ROUND)
    fin = ~np.isnan(x)
    better = (x[..., None, :] > x[..., :, None]) & fin[..., None, :]
    return np.where(fin, 1.0 + better.sum(-1), np.nan)


def pool_strata(
    tau: np.ndarray, disc: np.ndarray, sens: np.ndarray, incomplete: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each score's mean over the strata (axis -2), so a populous stratum cannot dominate.

    A metric incomplete in any stratum, or scored in some strata and not others, is
    incomplete. One stratum returns its own scores.
    """

    def mean(x: np.ndarray) -> np.ndarray:
        ok = np.isfinite(x)
        n = ok.sum(-2)
        return np.divide(np.where(ok, x, 0.0).sum(-2), n, out=np.full(n.shape, np.nan), where=n > 0)

    has = np.isfinite(tau)
    partial = np.asarray(incomplete, bool).any(-2) | (has.any(-2) & ~has.all(-2))
    return mean(tau), mean(disc), mean(sens), partial


def composite_score(
    tau: np.ndarray,
    disc: np.ndarray,
    sens: np.ndarray,
    weights: Mapping[str, float],
    ranked: np.ndarray,
) -> np.ndarray:
    """Weighted mean of the component competition ranks (lower is better) over ``ranked``.

    A ranked metric missing one component takes that component's worst rank; metrics
    outside ``ranked`` are NaN.
    """
    total = 0.0
    for name, x in zip(COMPONENTS, (tau, disc, sens), strict=True):
        x = np.where(ranked, x, np.nan)
        r = competition_ranks(x)
        worst = 1.0 + np.isfinite(x).sum(-1, keepdims=True)
        total = total + weights[name] * np.where(np.isfinite(r), r, worst)
    return np.round(np.where(ranked, total, np.nan), _ROUND)


def direction_ambiguous(
    per_cell: Mapping[tuple[str, str, str], float],
    family_of: Mapping[str, str],
    voting_bands: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, bool]:
    """Per metric: banded τ >= +threshold on one axis family and <= -threshold on another."""
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (m, a, b), t in per_cell.items():
        if b in voting_bands:
            by[(m, family_of[a])].append(t)
    fam: dict[str, list[float]] = defaultdict(list)
    for (m, _f), ts in by.items():
        fam[m].append(float(np.mean(ts)))
    return {m: max(v) >= threshold and min(v) <= -threshold for m, v in fam.items()}


def sensitivity_table(
    axis_sensitivity: Mapping[str, np.ndarray],
    metrics: Sequence[str],
    exclude: frozenset[str] = frozenset(),
) -> dict[str, list[tuple[str, float, int]]]:
    """``{axis: [(metric, sensitivity, competition rank)]}``, best first; NaN left out."""
    out = {}
    for a, sens in axis_sensitivity.items():
        x = np.array([np.nan if m in exclude else v for m, v in zip(metrics, sens, strict=True)])
        r = competition_ranks(x)
        rows = [(m, float(x[i]), int(r[i])) for i, m in enumerate(metrics) if np.isfinite(x[i])]
        out[a] = sorted(rows, key=lambda t: (t[2], t[0]))
    return out
