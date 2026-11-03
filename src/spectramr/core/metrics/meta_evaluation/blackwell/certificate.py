"""Explicit incoherence certificates: what a metric's preferences cannot reconcile.

An infeasible coherence LP says no concave isotone criterion reproduces the preferences,
but not why. Two certificates built from the preferences alone name the cause:

* a **cycle** ``a > b > ... > a``. No criterion of any kind reproduces it;
* **convex dominance** (Theorem C1.8(a)). The metric prefers every ``i`` in a set over
  ``j`` (directly or through a chain), yet ``J_j >= sum_i lambda_i J_i`` for some mixture
  of that set. No quasi-concave isotone criterion reproduces it.

Both are sound. Neither is complete for partial preference data (Theorem C1.8(b')), so an
infeasible system with no certificate is reported as such. Dominance is checked on the
exact coefficient groups only: comparing summed bins would not imply the comparison
per coefficient.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linprog

#: Mixture weights below this are dropped from a reported certificate.
_WEIGHT_FLOOR = 1e-6


def preference_closure(preferences: Sequence[tuple[int, int]], n: int) -> np.ndarray:
    """Boolean ``(n, n)`` transitive closure: ``[i, j]`` when ``i`` is preferred over ``j``."""
    closure = np.zeros((n, n), dtype=bool)
    for i, j in preferences:
        closure[i, j] = True
    for k in range(n):  # Warshall
        closure |= closure[:, [k]] & closure[[k], :]
    return closure


def find_cycle(closure: np.ndarray) -> list[int] | None:
    """A witnessed cycle ``[a, b, a]`` (``a`` and ``b`` each preferred, through chains, over
    the other), if one exists."""
    on_cycle = np.nonzero(np.diag(closure))[0]
    if on_cycle.size == 0:
        return None
    a = int(on_cycle[0])
    partners = np.nonzero(closure[a] & closure[:, a])[0]
    partners = partners[partners != a]
    return [a, int(partners[0]), a] if partners.size else [a, a]


def dominance_certificates(
    profiles: np.ndarray, closure: np.ndarray, *, limit: int = 5
) -> list[tuple[int, dict[int, float]]]:
    """Up to ``limit`` pairs ``(j, {i: lambda_i})`` with ``J_j >= sum_i lambda_i J_i``.

    ``profiles`` is ``(E, G)`` on exact groups. Each certificate solves one small LP
    over the experiments preferred to ``j``. Columns are scaled to O(1) as in the
    coherence LP, and the inequality is re-checked in float64.
    """
    scale = profiles.max(axis=0)
    p = profiles[:, scale > 0] / scale[scale > 0]
    found: list[tuple[int, dict[int, float]]] = []
    for j in range(p.shape[0]):
        preferred = np.nonzero(closure[:, j])[0]
        preferred = preferred[preferred != j]
        if preferred.size == 0:
            continue
        res = linprog(
            np.zeros(preferred.size),
            A_ub=p[preferred].T,
            b_ub=p[j],
            A_eq=np.ones((1, preferred.size)),
            b_eq=[1.0],
            bounds=[(0.0, None)] * preferred.size,
            method="highs",
        )
        if res.status != 0 or float((p[preferred].T @ res.x - p[j]).max()) > 1e-7:
            continue
        weights = {
            int(i): float(w) for i, w in zip(preferred, res.x, strict=True) if w > _WEIGHT_FLOOR
        }
        found.append((j, weights))
        if len(found) == limit:
            break
    return found


def incoherence_certificate(
    profiles: np.ndarray,
    preferences: Sequence[tuple[int, int]],
    names: Sequence[str],
    *,
    exact: bool,
    limit: int = 5,
) -> dict[str, object]:
    """The cycle and dominance certificates for ``preferences``, named.

    ``status`` is ``"cycle"``, ``"dominance"``, ``"none found"`` or, on a binned
    representation, ``"not checked (binned)"``.
    """
    closure = preference_closure(preferences, len(names))
    cycle = find_cycle(closure)
    if cycle is not None:
        return {"status": "cycle", "cycle": [names[k] for k in cycle], "dominance": []}
    if not exact:
        return {"status": "not checked (binned)", "cycle": None, "dominance": []}
    certs = dominance_certificates(profiles, closure, limit=limit)
    named = [
        {"dominated": names[j], "preferred_mixture": {names[i]: w for i, w in mix.items()}}
        for j, mix in certs
    ]
    return {"status": "dominance" if named else "none found", "cycle": None, "dominance": named}


__all__ = [
    "dominance_certificates",
    "find_cycle",
    "incoherence_certificate",
    "preference_closure",
]
