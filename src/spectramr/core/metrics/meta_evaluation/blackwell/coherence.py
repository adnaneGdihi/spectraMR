"""Revealed-design coherence test (Theorem C1.7) on Fourier-diagonal profiles.

A set of strict preferences ``i > j`` is reproduced by some concave, Loewner-isotone
information criterion iff the Afriat system

    phi_j <= phi_i + <G_i, J_j - J_i>   (all i != j),    phi_i >= phi_j + 1   ((i, j) in P)

has a solution with every ``G_i >= 0``. On diagonal profiles ``G_i`` is diagonal; averaging
it over a group on which every profile is constant leaves each inner product unchanged.
So groups of identical profiles lose nothing. Coarser bins restrict ``G`` instead: a feasible
binned system proves coherence, while an infeasible one is inconclusive.

HiGHS drops matrix entries below ``small_matrix_value`` (1e-9), and the information-poor
frequencies are exactly where a steep criterion puts its mass. Each group's column is
therefore scaled to O(1) before solving, and the returned solution is re-checked in
float64 against the full system, including any entries the solver dropped. A solution
that fails the re-check is reported as unverified rather than coherent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .catalogue import DiagonalExperiment
from .order import RelationGraph

#: Feasibility re-check tolerance on the unscaled system.
_VERIFY_TOL = 1e-6


@dataclass(frozen=True)
class CoherenceResult:
    """``coherent`` is True (verified feasible), False (infeasible on an exact
    representation) or None (unverified, or infeasible on a binned one)."""

    coherent: bool | None
    mass: float | None  # minimal total supergradient mass sum_i ||G_i||_1, when coherent
    status: str
    n_preferences: int
    exact: bool


def group_profiles(
    experiments: Sequence[DiagonalExperiment], groups: torch.Tensor, max_groups: int
) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(profiles (E, G), sizes (G,), exact)``: summed information per group.

    ``groups`` come from :func:`~.catalogue.info_groups`, on which every profile is
    constant. When there are more than ``max_groups`` of them, neighbouring groups are
    merged into ``max_groups`` bins and ``exact`` is False.
    """
    flat = groups.reshape(-1).cpu()
    n_found = int(flat.max()) + 1
    exact = n_found <= max_groups
    if not exact:
        flat = (flat * max_groups) // n_found
    n_groups = int(flat.max()) + 1
    sizes = np.bincount(flat.numpy(), minlength=n_groups).astype(np.float64)
    profiles = np.zeros((len(experiments), n_groups))
    for e, exp in enumerate(experiments):
        info = torch.exp(exp.log_info.reshape(-1).cpu()).numpy()
        profiles[e] = np.bincount(flat.numpy(), weights=info, minlength=n_groups)
    return profiles, sizes, exact


def revealed_design_lp(
    profiles: np.ndarray,
    sizes: np.ndarray,
    preferences: Sequence[tuple[int, int]],
    *,
    exact: bool = True,
) -> CoherenceResult:
    """Solve the Afriat system for ``preferences`` (``(i, j)`` means ``i`` preferred).

    Minimises the total supergradient mass ``sum_i sum_g g_ig * size_g``, which feeds the
    conditioning index. Variables are ``phi`` (free) and the column-scaled ``g'_ig =
    g_ig * scale_g``.
    """
    n_e, _n_g = profiles.shape
    prefs = [(int(i), int(j)) for i, j in preferences]
    if not prefs:
        return CoherenceResult(True, 0.0, "no preferences", 0, exact)
    scale = profiles.max(axis=0)
    keep = scale > 0.0
    p = profiles[:, keep] / scale[keep]
    size_scaled = sizes[keep] / scale[keep]
    k = p.shape[1]
    rows, cols, vals = [], [], []
    r = 0
    for i in range(n_e):
        for j in range(n_e):
            if i == j:
                continue
            rows += [r, r]
            cols += [j, i]
            vals += [1.0, -1.0]
            rows += [r] * k
            cols += list(range(n_e + i * k, n_e + (i + 1) * k))
            vals += list(-(p[j] - p[i]))
            r += 1
    for i, j in prefs:
        rows += [r, r]
        cols += [j, i]
        vals += [1.0, -1.0]
        r += 1
    a_ub = coo_matrix((vals, (rows, cols)), shape=(r, n_e + n_e * k)).tocsr()
    b_ub = np.zeros(r)
    b_ub[n_e * (n_e - 1) :] = -1.0
    cost = np.concatenate([np.zeros(n_e), np.tile(size_scaled, n_e)])
    cost = cost / cost.max()
    bounds = [(None, None)] * n_e + [(0.0, None)] * (n_e * k)
    res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status == 2:
        coherent = False if exact else None
        return CoherenceResult(coherent, None, "infeasible", len(prefs), exact)
    if res.status != 0:
        return CoherenceResult(None, None, f"solver status {res.status}", len(prefs), exact)
    x = res.x
    slack = a_ub @ x - b_ub
    if float(slack.max()) > _VERIFY_TOL:
        return CoherenceResult(None, None, "unverified", len(prefs), exact)
    mass = float(np.tile(size_scaled, n_e) @ x[n_e:])
    return CoherenceResult(True, mass, "feasible", len(prefs), exact)


def preferences_by_t(values: torch.Tensor, t_min: float) -> list[tuple[int, int]]:
    """Pairs ``(i, j)`` whose paired t-statistic over contents exceeds ``t_min``.

    ``values`` is ``(C, E)``. This is a screening rule, not simultaneous certification,
    so an infeasible system built from it is a diagnostic rather than a falsification
    (Theorem C1.7 needs every preference to be true).
    """
    v = values.double()
    if v.shape[0] < 2:
        raise ValueError("the preference rule needs at least two contents")
    d = v[:, :, None] - v[:, None, :]  # (C, E, E): d[c, i, j] = v_i - v_j
    # Each pair uses the contents finite on both sides, so one NaN content (a metric
    # declining that image) cannot erase every preference.
    valid = torch.isfinite(d)
    n = valid.sum(dim=0)
    d = torch.where(valid, d, torch.zeros_like(d))
    mean = d.sum(dim=0) / n.clamp_min(1)
    var = ((d - mean) ** 2 * valid).sum(dim=0) / (n - 1).clamp_min(1)
    sd = var.sqrt()
    t = mean / (sd / n.clamp_min(1).double().sqrt())
    t = torch.where(sd > 0, t, torch.sign(mean) * torch.inf)
    t = torch.where(n >= 2, t, torch.zeros_like(t))
    t = torch.nan_to_num(t, nan=0.0)
    idx = torch.nonzero(t > t_min, as_tuple=False)
    return [(int(i), int(j)) for i, j in idx if i != j]


def split_preferences(
    preferences: Sequence[tuple[int, int]], graph: RelationGraph
) -> tuple[list[tuple[int, int]], int, int]:
    """Preferences for the coherence LP, and how many were set aside.

    A strict preference inside an equivalent pair, or against a certified strict
    relation, makes the system infeasible by itself. Those failures are already counted
    by the equivalence and violation rates. Leaving them out makes ``coherent`` say
    something beyond them: whether the remaining preferences, mostly on incomparable
    pairs, fit one information criterion.

    Returns ``(kept, n_split_equivalent, n_reversed_strict)``.
    """
    equivalent = {frozenset(p) for p in graph.equivalent}
    reversed_strict = {(j, i) for i, j in graph.strict}
    kept, n_eq, n_rev = [], 0, 0
    for p in preferences:
        if frozenset(p) in equivalent:
            n_eq += 1
        elif p in reversed_strict:
            n_rev += 1
        else:
            kept.append(p)
    return kept, n_eq, n_rev


__all__ = [
    "CoherenceResult",
    "group_profiles",
    "preferences_by_t",
    "revealed_design_lp",
    "split_preferences",
]
