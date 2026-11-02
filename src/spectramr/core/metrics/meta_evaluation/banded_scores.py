"""Discriminability, per-degradation sensitivity and a composite rank on the banded board.

τ-b saturates: every strictly monotone metric scores 1 in a band, whether it moves 1e-7 or
10 dB per level. Discriminability is the fraction of adjacent damage steps a metric moves
by more than ``z·√2`` floors (a step differences two noisy levels); sensitivity is its
Theil-Sen slope against damage, per floor. Each is kept per content as a weighted sum (or
a weighted median), so the nominal scores are :func:`reduce_scores` with unit weights and a
subject-bootstrap replicate is the same call with resampled weights, never a re-sweep.

One scale per axis. On an axis where any (metric, content) has a seed spread, the scale is
``max(floor, rtol·|metric range|)``, and an entry without a measured floor scores NaN, never
on another scale. On a deterministic axis (every floor exactly 0) sensitivity is a
fraction of the metric's range per unit damage and discriminability does not apply. An
axis without floors scores neither. Sensitivity is compared within a cell only (per-cell
percentile of the metrics), so its units never mix across axes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from spectramr.core.metrics.meta_evaluation.axis_limits import FLAT_RTOL
from spectramr.core.metrics.meta_evaluation.banded_composite import competition_ranks
from spectramr.core.metrics.meta_evaluation.damage import (
    MIN_AXES_PER_BAND,
    MIN_LEVELS_PER_BAND,
    NON_VOTING_BANDS,
    Bands,
)
from spectramr.core.metrics.meta_evaluation.masked_order import (
    block_end,
    distinct_levels,
    masked_median,
    weighted_median,
)

__all__ = ["DISCRIMINABILITY_Z", "CellStats", "Scores", "cell_stats", "reduce_scores"]

#: Floors an adjacent step must exceed (times √2) to count as resolved.
DISCRIMINABILITY_Z: float = 2.0
#: Scale of an entry: its floor, ``rtol·range`` (a floor at or below it, e.g. a metric
#: that does not move across seeds), the metric's range (deterministic axis), or none.
SCALE_SOURCES = ("measured", "rtol_clamped", "fraction_of_range", "none")


@dataclass(frozen=True)
class CellStats:
    """Per (metric, cell, content) sufficient statistics; a cell is one (axis, voting band)."""

    metrics: tuple[str, ...]
    cells: tuple[tuple[str, str], ...]
    band_names: tuple[str, ...]
    s: np.ndarray  # (M, K, N) Σ sign(Δx)·sign(Δd) over in-band pairs
    nx: np.ndarray  # (M, K, N) pairs untied in x
    nd: np.ndarray  # (M, K, N) pairs untied in damage, over the metric's finite levels
    hits: np.ndarray  # (M, K, N) adjacent steps resolved above the threshold
    steps: np.ndarray  # (M, K, N) adjacent rising steps with a measured scale
    slope: np.ndarray  # (M, K, N) in-band Theil-Sen slope / scale; NaN where none
    row_vote: np.ndarray  # (M, K, N) the metric has min_levels distinct finite levels there
    vote: np.ndarray  # (K, N) the content has min_levels distinct levels there
    disc_ok: np.ndarray  # (K,) discriminability applies (the cell's axis has a seed spread)
    real: np.ndarray  # (N,) False for a pooled distributional stratum
    axis_slope: dict[str, np.ndarray]  # axis -> (M, N) full-range slope / scale per content
    axis_mode: dict[str, str]  # axis -> "measured" | "deterministic" | "none"
    scale_source: dict[str, dict[str, str]]  # axis -> metric -> worst entry, SCALE_SOURCES
    rtol_fraction: dict[str, float]  # axis -> share of scored entries that are rtol_clamped
    z: float
    rtol: float


@dataclass(frozen=True)
class Scores:
    """Batched scores over ``B`` weightings: overall ``(B, M)``, per band ``(B, M, nb)``."""

    tau: np.ndarray
    disc: np.ndarray
    sens: np.ndarray
    incomplete: np.ndarray
    tau_scored: np.ndarray  # (B, M, K)
    missing: np.ndarray  # (B, M, K) covered voting cells lacking a component
    band_tau: np.ndarray
    band_disc: np.ndarray
    band_sens: np.ndarray
    voting: np.ndarray  # (B, nb)


def _per_scale(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = np.where(np.isfinite(x) & (scale == 0), 0.0, np.nan)
    return np.divide(x, scale, out=out, where=scale > 0)


def _span(arrs: Sequence[np.ndarray]) -> np.ndarray:
    hi = np.max([np.where(np.isfinite(v), v, -np.inf).max(axis=(1, 2)) for v in arrs], 0)
    lo = np.min([np.where(np.isfinite(v), v, np.inf).min(axis=(1, 2)) for v in arrs], 0)
    return np.where(np.isfinite(hi - lo), hi - lo, 0.0)


def _axis_scale(
    f: np.ndarray | None, shape: tuple[int, ...], span: np.ndarray, rtol: float
) -> tuple[str, np.ndarray, np.ndarray]:
    """``(mode, scale, clamped)`` of one axis under the one-scale-per-axis rule.

    ``clamped`` marks measured floors at or below ``rtol·span`` (zero seed spread is
    real), whose scale is ``rtol·span``.
    """
    fv = np.full(shape, np.nan) if f is None else np.asarray(f, np.float64)
    has, floor = np.isfinite(fv), rtol * span[:, None]
    if (has & (fv > 0)).any():
        clamped = has & (np.nan_to_num(fv) < floor)
        return "measured", np.where(has, np.maximum(fv, floor), np.nan), clamped
    if has.any():
        return "deterministic", np.broadcast_to(span[:, None], shape).copy(), np.zeros(shape, bool)
    return "none", np.full(shape, np.nan), np.zeros(shape, bool)


def _source(mode: str, scored: np.ndarray, clamped: np.ndarray, scale: np.ndarray) -> str:
    """One metric's stamp on an axis: its worst scored entry, in SCALE_SOURCES order."""
    ok = scored & np.isfinite(scale)
    if not ok.any():
        return "none"
    if mode == "deterministic":
        return "fraction_of_range"
    return "rtol_clamped" if (ok & clamped).any() else "measured"


def cell_stats(
    values: Mapping[str, np.ndarray],
    damage: Mapping[str, np.ndarray],
    floors: Mapping[str, np.ndarray],
    higher_is_better: Sequence[bool],
    bands: Bands,
    *,
    metrics: Sequence[str],
    real: np.ndarray,
    min_levels: int = MIN_LEVELS_PER_BAND,
    z: float = DISCRIMINABILITY_Z,
    rtol: float = FLAT_RTOL,
) -> CellStats:
    """:class:`CellStats` from ``{axis: (M, N, T_a)}`` values and ``{axis: (N, T_a)}`` damage.

    ``floors[axis]`` is ``(M, N)`` seed sigma, NaN where unmeasured (an absent axis has
    none). Vectorised over metrics, contents and level pairs; the Python loop runs over
    axes and bands only.
    """
    hib = np.asarray(higher_is_better, bool)
    voting_bands = [(b, lo, hi) for b, lo, hi in bands if b not in NON_VOTING_BANDS]
    arrs = {a: np.asarray(values[a], np.float64) for a in values}
    span = _span(list(arrs.values()))
    out: dict[str, list[np.ndarray]] = defaultdict(list)
    cells, axis_slope, mode, source, rtol_share = [], {}, {}, {}, {}
    for a, v in arrs.items():
        o = np.where(hib[:, None, None], -v, v)
        d = np.asarray(damage[a], np.float64)
        order = np.argsort(d, axis=1, kind="stable")
        ds = np.take_along_axis(d, order, 1)
        os_ = np.take_along_axis(o, np.broadcast_to(order, o.shape), 2)
        fin, end = np.isfinite(os_), block_end(ds)
        mode[a], scale, clamped = _axis_scale(floors.get(a), o.shape[:2], span, rtol)
        scored = fin.any(-1)
        rtol_share[a] = float((scored & clamped).sum() / max(1, scored.sum()))
        source[a] = {
            m: _source(mode[a], scored[k], clamped[k], scale[k]) for k, m in enumerate(metrics)
        }
        thr = z * np.sqrt(2.0) * scale
        i, j = np.triu_indices(ds.shape[1], 1)
        dd, dv = ds[:, j] - ds[:, i], os_[:, :, j] - os_[:, :, i]
        fp = fin[:, :, i] & fin[:, :, j]
        sx, sd = np.where(fp, np.sign(dv), 0.0), np.sign(dd)
        ok_slope = fp & (dd > 0)[None]
        slope = np.divide(dv, dd[None], out=np.full(dv.shape, np.nan), where=ok_slope)
        axis_slope[a] = _per_scale(masked_median(slope, ok_slope), scale)
        rise = np.diff(ds, axis=1) > 0
        dstep = np.diff(os_, axis=2)
        fstep = fin[:, :, :-1] & fin[:, :, 1:] & np.isfinite(thr)[..., None]
        for band, lo, hi in voting_bands:
            inb = (ds >= lo) & (ds < hi)
            pm = inb[:, i] & inb[:, j]
            out["s"].append(np.einsum("mnp,np->mn", sx, sd * pm))
            out["nx"].append(np.einsum("mnp,np->mn", (sx != 0).astype(float), pm.astype(float)))
            out["nd"].append(
                np.einsum("mnp,np->mn", fp.astype(float), ((sd != 0) & pm).astype(float))
            )
            sv = fstep & (inb[:, :-1] & inb[:, 1:] & rise)[None]
            out["steps"].append(sv.sum(-1).astype(float))
            out["hits"].append((sv & (dstep > thr[..., None])).sum(-1).astype(float))
            out["slope"].append(_per_scale(masked_median(slope, ok_slope & pm[None]), scale))
            out["row_vote"].append(distinct_levels(inb[None] & fin, end) >= min_levels)
            out["vote"].append(distinct_levels(inb, end) >= min_levels)
            out["disc_ok"].append(np.array(mode[a] == "measured"))
            cells.append((str(a), band))
    m, n = len(metrics), np.asarray(real).size
    empty = np.zeros((m, 0, n))
    st = {k: np.stack(out[k], axis=1) if out[k] else empty for k in ("s", "nx", "nd", "hits")}
    st |= {k: np.stack(out[k], axis=1) if out[k] else empty for k in ("steps", "slope")}
    return CellStats(
        metrics=tuple(metrics),
        cells=tuple(cells),
        band_names=tuple(b for b, _lo, _hi in voting_bands),
        **st,
        row_vote=(np.stack(out["row_vote"], axis=1) if out["row_vote"] else empty).astype(bool),
        vote=np.stack(out["vote"]) if out["vote"] else np.zeros((0, n), bool),
        disc_ok=np.array(out["disc_ok"], bool).reshape(len(cells)),
        real=np.asarray(real, bool),
        axis_slope=axis_slope,
        axis_mode=mode,
        scale_source=source,
        rtol_fraction=rtol_share,
        z=float(z),
        rtol=float(rtol),
    )


def _cell_percentile(sens: np.ndarray) -> np.ndarray:
    """``(B, M, K)`` -> each metric's percentile among the metrics scored in that cell."""
    x = np.moveaxis(sens, 1, -1)  # (B, K, M)
    n = np.isfinite(x).sum(-1, keepdims=True)
    p = np.where(n > 1, 1.0 - (competition_ranks(x) - 1.0) / np.maximum(n - 1, 1), 1.0)
    return np.moveaxis(np.where(np.isfinite(x), p, np.nan), -1, 1)


def reduce_scores(
    st: CellStats,
    weights: np.ndarray,
    min_axes: int = MIN_AXES_PER_BAND,
    exclude: np.ndarray | None = None,
) -> Scores:
    """Banded τ, discriminability and sensitivity under each row of ``weights`` ``(B, N)``.

    A cell is covered when a real content with positive weight votes there; a pooled
    stratum adds scores only inside covered cells, with the weight given. Per band, the
    mean over the covered cells a metric is scored on; overall, the mean over the bands
    at least ``min_axes`` covered axes reach; unit weights reproduce ``banded_leaderboard``.
    Sensitivity enters as its per-cell percentile among the metrics not in ``exclude``
    (the circular ones). A metric missing τ, an applicable discriminability or a
    sensitivity on a covered voting cell is incomplete.
    """
    w = np.asarray(weights, np.float64)
    keep = np.ones(len(st.metrics), bool) if exclude is None else ~np.asarray(exclude, bool)
    eff = st.row_vote & st.vote[None] & keep[:, None, None]

    def wsum(x: np.ndarray) -> np.ndarray:
        return np.tensordot(w, np.where(eff, x, 0.0), axes=([1], [2]))  # (B, M, K)

    covered = (w > 0) @ (st.vote & st.real).T.astype(float) > 0  # (B, K)
    scored = covered[:, None, :] & (wsum(np.ones(eff.shape)) > 0)
    den = np.sqrt(wsum(st.nx) * wsum(st.nd))
    tau = np.where(
        scored, np.divide(wsum(st.s), den, out=np.zeros(den.shape), where=den > 0), np.nan
    )
    n_steps = wsum(st.steps)
    disc = np.divide(wsum(st.hits), n_steps, out=np.full(den.shape, np.nan), where=n_steps > 0)
    disc = np.where(scored & st.disc_ok, disc, np.nan)
    sens = weighted_median(st.slope, eff & np.isfinite(st.slope), w)
    sens = _cell_percentile(np.where(scored, sens, np.nan))
    g = np.array([[c[1] == b for b in st.band_names] for c in st.cells], float).reshape(
        len(st.cells), len(st.band_names)
    )
    voting = covered.astype(float) @ g >= min_axes  # (B, nb)

    def per_band(x: np.ndarray) -> np.ndarray:
        ok = np.isfinite(x)
        cnt = ok.astype(float) @ g
        return np.divide(
            np.where(ok, x, 0.0) @ g, cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0
        )

    def overall(bx: np.ndarray) -> np.ndarray:
        ok = np.isfinite(bx) & voting[:, None, :]
        cnt = ok.sum(-1)
        return np.divide(
            np.where(ok, bx, 0.0).sum(-1), cnt, out=np.full(cnt.shape, np.nan), where=cnt > 0
        )

    bt, bd, bs = per_band(tau), per_band(disc), per_band(sens)
    miss = ~scored | (scored & st.disc_ok & np.isnan(disc)) | (scored & np.isnan(sens))
    in_vote = (voting.astype(float) @ g.T > 0)[:, None, :]
    missing = covered[:, None, :] & miss & in_vote & keep[:, None]
    return Scores(
        tau=overall(bt),
        disc=overall(bd),
        sens=overall(bs),
        incomplete=missing.any(-1),
        tau_scored=scored,
        missing=missing,
        band_tau=bt,
        band_disc=bd,
        band_sens=bs,
        voting=voting,
    )
