"""Banded equal-damage leaderboard: compare metrics at equal measured damage.

A mean of per-degradation scores lets the harshest-swept degradation dominate. Here each
damage band is scored on its own: every degradation with enough distinct damage levels in
a band casts one vote there (Kendall τ-b between the oriented metric and damage, pooled
over contents), the covered degradations weigh equally within a band, and the bands that
at least ``MIN_AXES_PER_BAND`` degradations cover weigh equally overall.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from spectramr.core.metrics.meta_evaluation.damage import (
    DAMAGE_BANDS,
    MIN_AXES_PER_BAND,
    MIN_LEVELS_PER_BAND,
    Bands,
    band_votes,
    compare_yardsticks,
    disqualified_metrics,
)
from spectramr.core.metrics.meta_evaluation.pooled_tau import (
    CIRCULARITY_THRESHOLD,
    PooledTau,
    measured_circular,
)
from spectramr.core.metrics.meta_evaluation.pooled_tau import (
    tau_b_rows as _tau_b_rows,  # noqa: F401  (kept importable here)
)

__all__ = [
    "BandedResult",
    "banded_json",
    "banded_leaderboard",
    "banded_leaderboards",
    "overall_ranking",
]


@dataclass(frozen=True)
class BandedResult:
    """One yardstick's banded leaderboard. Absent keys are uncovered, never zero."""

    per_cell: dict[tuple[str, str, str], float]  # (metric, axis, band) -> pooled τ-b
    per_band: dict[tuple[str, str], float]  # (metric, band) -> mean over covered axes
    overall: dict[str, float]  # metric -> mean over the voting bands it is scored on
    coverage: dict[tuple[str, str], int]  # (axis, band) -> max distinct levels over contents
    #: metric -> covered (axis, band) cells of a voting band it has no finite score on.
    #: Its ``overall`` is then partial and must not be ranked with the complete metrics.
    incomplete: dict[str, tuple[tuple[str, str], ...]]
    #: Bands covered by at least ``min_axes`` degradations; only these enter ``overall``.
    voting_bands: tuple[str, ...] = ()
    min_axes: int = MIN_AXES_PER_BAND
    #: metric -> why it was left out as circular under this yardstick (never ranked).
    circular: dict[str, str] = field(default_factory=dict)
    circular_declared: tuple[str, ...] = ()
    circular_measured: tuple[str, ...] = ()
    circular_threshold: float | None = None


def _band_masks(damage: np.ndarray, bands: Bands) -> list[tuple[str, np.ndarray]]:
    """Half-open ``[lo, hi)`` membership per band, with :func:`damage.band_of`'s raises."""
    if not np.all(np.isfinite(damage)) or np.any(damage < 0):
        raise ValueError("damage must be finite and non-negative for every sample")
    masks = [(name, (damage >= lo) & (damage < hi)) for name, lo, hi in bands]
    inside = np.logical_or.reduce([m for _, m in masks])
    if not np.all(inside):
        raise ValueError(f"damage {damage[~inside][:5].tolist()} falls outside the bands")
    return masks


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values))


def banded_leaderboard(
    values: Mapping[str, np.ndarray],
    axis: np.ndarray,
    content: np.ndarray,
    damage: np.ndarray,
    higher_is_better: Mapping[str, bool],
    bands: Bands,
    min_levels: int = MIN_LEVELS_PER_BAND,
    min_axes: int = MIN_AXES_PER_BAND,
) -> BandedResult:
    """Score every metric per (axis, band) cell at equal measured damage.

    One row per (axis, content, level) sample. A content votes in a cell when
    :func:`damage.band_votes` accepts the number of *distinct* damage values it has in
    the band, so a saturated degradation whose damage is tied is uncovered there rather
    than scored 0; ``imperceptible`` never votes. The metric is oriented so that a larger
    value means worse quality, then correlated with damage by a τ-b pooled over the
    voting contents (pairs counted within a content, summed across contents).

    A constant metric scores 0.0: a metric that does not move across a band is blind
    there, and that is the signal.

    The covered cells are fixed by the damage alone and are common to every metric. A
    metric whose non-finite samples leave a covered cell without a score is neither
    zero-filled (that would call an unmeasurable metric blind) nor silently averaged over
    fewer cells: the cell is listed in ``incomplete`` and its ``overall`` is partial.

    Args:
        values: ``{metric: [N]}`` metric values.
        axis: ``[N]`` degradation name of each sample.
        content: ``[N]`` content (reference image) id of each sample.
        damage: ``[N]`` damage of each sample on one yardstick.
        higher_is_better: Direction of every metric in ``values``.
        bands: That yardstick's ``(band, lo, hi)`` tuple, e.g. ``DAMAGE_BANDS["rel_l2"]``.
        min_levels: Distinct damage levels a content needs in a band to vote there.
        min_axes: Covered degradations a band needs to count in ``overall``.
    """
    if min_axes < 1:
        raise ValueError(f"min_axes must be >= 1, got {min_axes}")
    axis, content = np.asarray(axis), np.asarray(content)
    damage = np.asarray(damage, dtype=np.float64)
    n = damage.shape[0]
    if axis.shape != (n,) or content.shape != (n,):
        raise ValueError(f"axis {axis.shape}, content {content.shape}, damage {damage.shape}")
    missing = sorted(set(values) - set(higher_is_better))
    if missing:
        raise KeyError(f"no direction (higher_is_better) for metrics {missing}")
    oriented: dict[str, np.ndarray] = {}
    for m, v in values.items():
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (n,):
            raise ValueError(f"values[{m!r}] has shape {arr.shape}, expected ({n},)")
        oriented[m] = -arr if higher_is_better[m] else arr
    masks = _band_masks(damage, bands)
    names = list(oriented)
    stacked = np.stack([oriented[m] for m in names]) if names else np.empty((0, n))

    per_cell: dict[tuple[str, str, str], float] = {}
    coverage: dict[tuple[str, str], int] = {}
    covered: set[tuple[str, str]] = set()
    for a in np.unique(axis):
        on_axis = axis == a
        for band, in_band in masks:
            key = (str(a), band)
            acc = PooledTau(len(names))
            for c in np.unique(content[on_axis]):
                sel = np.flatnonzero(on_axis & (content == c) & in_band)
                d = damage[sel]
                distinct = int(np.unique(d).size)
                coverage[key] = max(coverage.get(key, 0), distinct)
                if not band_votes(band, distinct, min_levels):
                    continue
                covered.add(key)
                acc.add(stacked[:, sel], d, min_levels)
            if key in covered:
                tau = acc.tau()
                for r in np.flatnonzero(acc.voted):
                    per_cell[(names[r], *key)] = float(tau[r])

    axes_per_band: dict[str, int] = defaultdict(int)
    for _a, band in covered:
        axes_per_band[band] += 1
    voting = tuple(b for b, _lo, _hi in bands if axes_per_band.get(b, 0) >= min_axes)
    by_band: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (m, _a, band), s in per_cell.items():
        by_band[(m, band)].append(s)
    per_band = {k: _mean(s) for k, s in by_band.items()}
    by_metric: dict[str, list[float]] = defaultdict(list)
    for (m, band), s in per_band.items():
        if band in voting:
            by_metric[m].append(s)
    overall = {m: _mean(s) for m, s in by_metric.items()}
    incomplete: dict[str, tuple[tuple[str, str], ...]] = {}
    for m in oriented:
        gaps = tuple(sorted(c for c in covered if c[1] in voting and (m, *c) not in per_cell))
        if gaps:
            incomplete[m] = gaps
    return BandedResult(
        per_cell=per_cell,
        per_band=per_band,
        overall=overall,
        coverage=coverage,
        incomplete=incomplete,
        voting_bands=voting,
        min_axes=min_axes,
    )


def banded_leaderboards(
    values: Mapping[str, np.ndarray],
    axis: np.ndarray,
    content: np.ndarray,
    damage: Mapping[str, np.ndarray],
    higher_is_better: Mapping[str, bool],
    min_levels: int = MIN_LEVELS_PER_BAND,
    bands: Mapping[str, Bands] = DAMAGE_BANDS,
    min_axes: int = MIN_AXES_PER_BAND,
    circular_threshold: float = CIRCULARITY_THRESHOLD,
) -> dict[str, BandedResult]:
    """One :func:`banded_leaderboard` per yardstick in ``damage``, without its circular metrics.

    A metric is circular under a yardstick when the yardstick declares it
    (:func:`damage.disqualified_metrics`) or when its pooled within-content |τ| with the
    damage reaches ``circular_threshold`` (:func:`pooled_tau.measured_circular`). Both
    lists are kept on the result, and each dropped metric carries its reason.
    ``bands`` maps each yardstick to its bands (``damage.bands_from_edges``). A board over
    one stratum takes the whole board's circular set instead: ``stratum_board``.
    """
    out: dict[str, BandedResult] = {}
    for y, d in damage.items():
        if y not in bands:
            raise KeyError(f"No damage bands for yardstick {y!r}. Have: {sorted(bands)}.")
        declared = disqualified_metrics(y)
        taus = measured_circular(values, d, axis, content, circular_threshold)
        measured = frozenset(taus)
        reasons: dict[str, str] = {}
        for m in sorted((declared | measured) & set(values)):
            why = [f"declared monotone in {y}"] if m in declared else []
            if m in measured:
                why.append(
                    f"measured: pooled within-content |tau| = {abs(taus[m]):.3f} "
                    f">= {circular_threshold} against {y}"
                )
            reasons[m] = "; ".join(why)
        kept = {m: v for m, v in values.items() if m not in reasons}
        res = banded_leaderboard(
            kept, axis, content, d, higher_is_better, bands[y], min_levels, min_axes
        )
        out[y] = replace(
            res,
            circular=reasons,
            circular_declared=tuple(sorted(declared)),
            circular_measured=tuple(sorted(measured)),
            circular_threshold=float(circular_threshold),
        )
    return out


def overall_ranking(result: BandedResult) -> list[tuple[str, int]]:
    """``(metric, rank)``: complete metrics first, then incomplete ones, each by ``overall``.

    Ranks are competition ranks (1, 1, 3): metrics with equal scores share one, and the
    name only fixes the listing order inside a tie.
    """

    def key(m: str) -> tuple[bool, float]:
        return (m in result.incomplete, -result.overall[m])

    order = sorted(result.overall, key=lambda m: (*key(m), m))
    out: list[tuple[str, int]] = []
    for pos, m in enumerate(order, start=1):
        tied = out and key(out[-1][0]) == key(m)
        out.append((m, out[-1][1] if tied else pos))
    return out


def banded_json(results: Mapping[str, BandedResult]) -> dict[str, object]:
    """``banded_leaderboard`` / ``coverage`` / ``yardstick_agreement`` with string keys only."""
    leaderboard: dict[str, object] = {}
    coverage: dict[str, dict[str, dict[str, int]]] = {}
    for y, r in results.items():
        per_band: dict[str, dict[str, float]] = defaultdict(dict)
        for (m, band), s in sorted(r.per_band.items()):
            per_band[band][m] = float(s)
        ranking = overall_ranking(r)
        leaderboard[y] = {
            "overall": {m: float(r.overall[m]) for m, _ in ranking},
            "ranks": dict(ranking),
            "per_band": dict(per_band),
            "voting_bands": list(r.voting_bands),
            "min_axes_per_band": r.min_axes,
            "incomplete": {
                m: [list(c) for c in cells] for m, cells in sorted(r.incomplete.items())
            },
            "circular": dict(r.circular),
            "circular_declared": list(r.circular_declared),
            "circular_measured": list(r.circular_measured),
            "circular_threshold": r.circular_threshold,
        }
        cov: dict[str, dict[str, int]] = defaultdict(dict)
        for (a, band), n in sorted(r.coverage.items()):
            cov[a][band] = int(n)
        coverage[y] = dict(cov)
    # Agreement on -rank: ties stay tied and partial scores stay below complete ones.
    agreement = compare_yardsticks(
        {y: {m: -float(k) for m, k in overall_ranking(r)} for y, r in results.items()}
    )
    return {
        "banded_leaderboard": leaderboard,
        "coverage": coverage,
        "yardstick_agreement": agreement,
    }
