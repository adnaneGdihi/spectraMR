"""Powered ranking certification — the finite-sample gate for pairwise verdicts.

This is the executable counterpart of the Lean theorems
``Sim2Rank.powered_sample_size`` / ``Sim2Rank.sample_size_for_recovery`` /
``Sim2Rank.pool_deviation_union_bound`` (``docs/lean/Sim2Rank/Sim2Rank/Powered.lean``)
and the paper's Corollary 5.2. It closes the gap raised in the meta-evaluation
critique §3 ("asymptotic-only theorems with no closed-form sample-size formula").

Theory
------
For a bounded rank-correlation statistic (Kendall / Spearman — the ADR ``rho``,
LGDR, and CDSCR rankers) computed on ``N`` i.i.d. evaluation samples,
Theorem 5.1(3) gives the misordering bound

    P(s_hat_A <= s_hat_B) <= 2 * exp(-m * Delta**2 / 8),   m = floor(N / 2),

for two metrics with population score gap ``Delta = s_A - s_B > 0``. Inverting the
bound (this is exactly what the Lean ``powered_sample_size`` proves):

    * to *certify* a pair at confidence ``1 - alpha`` the gap must clear
        Delta >= sqrt(8 * ln(2/alpha) / m)                  (gap threshold)
    * equivalently the sample size must satisfy
        N >= 2 * ceil(8 * ln(2/alpha) / Delta**2)           (critique eq. 2)

:func:`certify_partial_order` refuses to assert any pairwise ordering whose
observed gap falls below the powered threshold, and reports only the certified
partial order plus the ambiguous (statistically indistinguishable) pairs.

Exactness regime
----------------
The bound's sub-Gaussian proxy ``c = 1/m`` is exact for statistics bounded in
``[-1, 1]`` (rank correlations). For other bounded scores it is conservative; for
unbounded scores (KSG MI) it does not apply — see limitation L7/L14 in the paper.
The ``gap_threshold`` returned here is therefore calibrated for the
rank-correlation scale, and is most meaningful for rankers whose raw score is a
bounded concordance statistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def min_samples_for_gap(delta: float, alpha: float = 0.05) -> int:
    """Minimum evaluation sample size ``N`` to recover a pair at gap ``delta``.

    Implements the critique's eq. (2) / Lean ``powered_sample_size``::

        N >= 2 * ceil(8 * ln(2/alpha) / delta**2)

    Args:
        delta: population score gap ``s_A - s_B`` (on the rank-correlation scale,
            so ``delta in (0, 2]`` for Kendall/Spearman). Must be positive.
        alpha: target misordering probability (confidence ``1 - alpha``).

    Returns:
        The minimum number of i.i.d. evaluation samples ``N`` (an even integer,
        ``= 2 * m_min``) that guarantees ``P(misorder) <= alpha``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    m_min = math.ceil(8.0 * math.log(2.0 / alpha) / (delta * delta))
    return 2 * m_min


def min_samples_for_pool(delta: float, n_metrics: int, alpha: float = 0.05) -> int:
    """Minimum ``N`` to recover the *entire* ``K``-metric sort at min gap ``delta``.

    This is the whole-pool version (paper Corollary 5.2 / Lean
    ``pool_deviation_union_bound``). Recovering the *whole sort* requires every one
    of the ``C(K, 2) = K(K-1)/2`` pairwise orderings to be correct, so the union
    bound is over **pairs**, not the ``K`` metrics: with per-pair misorder
    ``2*exp(-m*delta^2/8)`` the total error is ``<= K(K-1)*exp(-m*delta^2/8)``, giving::

        N >= 2 * ceil(8 * ln(2 * C(K,2) / alpha) / delta**2)   ( = ln(K(K-1)/alpha) )

    (The earlier ``ln(2K/alpha)`` under-counted the comparisons and so under-powered
    large pools -- critique L6.) For ``K = 20``, ``delta = 0.1``, ``alpha = 0.01`` this
    is ~1.7e4 samples, orders of magnitude above a handful of severity steps.

    Args:
        delta: minimum pairwise population gap across the pool.
        n_metrics: pool size ``K``.
        alpha: target probability that the *whole* sort is wrong.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0, got {delta}")
    if n_metrics < 1:
        raise ValueError(f"n_metrics must be >= 1, got {n_metrics}")
    # Number of pairwise comparisons that must all be correct (clamp to >= 1 so a
    # singleton pool, which has no pairs, still yields a finite sensible bound).
    n_pairs = max(1, n_metrics * (n_metrics - 1) // 2)
    m_min = math.ceil(8.0 * math.log(2.0 * n_pairs / alpha) / (delta * delta))
    return 2 * m_min


def powered_gap_threshold(n_samples: int, alpha: float = 0.05) -> float:
    """Smallest score gap certifiable at confidence ``1 - alpha`` given ``N`` samples.

    The inverse of :func:`min_samples_for_gap` (inverting Theorem 5.1(3))::

        delta_min = sqrt(8 * ln(2/alpha) / m),   m = floor(N / 2)

    Returns ``math.inf`` when ``N < 2`` (no pair can be powered with fewer than one
    disjoint block).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    m = n_samples // 2
    if m < 1:
        return math.inf
    return math.sqrt(8.0 * math.log(2.0 / alpha) / m)


@dataclass
class PoweredPartialOrder:
    """A certified partial order over a metric pool at a fixed power level.

    Attributes:
        n_samples: the evaluation sample size ``N`` the certification used.
        alpha: the target misordering probability per pair, or over the family of
            pairs under ``correction="holm"``.
        gap_threshold: ``delta_min`` — the smallest gap one pair certifies at
            ``(N, alpha)`` before any multiplicity correction.
        certified: ``(better, worse)`` ordered pairs whose gap clears the threshold.
        ambiguous: unordered ``(a, b)`` pairs whose gap is below the threshold and
            are therefore statistically indistinguishable at this power.
    """

    n_samples: int
    alpha: float
    gap_threshold: float
    certified: list[tuple[str, str]]
    ambiguous: list[tuple[str, str]]
    correction: str = "none"

    @property
    def n_certified(self) -> int:
        return len(self.certified)

    @property
    def n_ambiguous(self) -> int:
        return len(self.ambiguous)

    def is_certified(self, better: str, worse: str) -> bool:
        """True iff ``better > worse`` is a powered (certified) verdict."""
        return (better, worse) in self.certified


CORRECTIONS = ("none", "holm")


def _pair_pvalue(gap: float, n_samples: int) -> float:
    """Hoeffding p-value of one pair: ``min(1, 2 exp(-m gap^2 / 8))``, ``m = N // 2``.

    ``p <= alpha`` iff ``|gap| >= powered_gap_threshold(N, alpha)``, so the two gates
    agree on a single pair. A non-finite gap (a crashed metric) has ``p = 1``.
    """
    m = n_samples // 2
    if m < 1 or not math.isfinite(gap):
        return 1.0
    return min(1.0, 2.0 * math.exp(-m * gap * gap / 8.0))


def certify_partial_order(
    scores: dict[str, float],
    n_samples: int,
    alpha: float = 0.05,
    *,
    correction: str = "none",
) -> PoweredPartialOrder:
    """Certify only the pairwise orderings whose gap clears the powered threshold.

    Every ordered pair ``(a, b)`` whose ``|scores[a] - scores[b]|`` exceeds
    ``delta_min`` is certified in the dominant direction; pairs below threshold are
    reported as ambiguous rather than ordered. This is the powered-ranking gate: it
    reports the partial order the data can actually support at ``(N, alpha)`` instead
    of forcing a (possibly underpowered) total order.

    Args:
        scores: per-metric scores (higher = better; rank-correlation scale).
        n_samples: evaluation sample size ``N`` (independent units, not correlated
            replicates).
        alpha: target misordering probability per pair (``"none"``), or the
            family-wise error rate over all pairs (``"holm"``).
        correction: ``"none"`` or ``"holm"`` (Holm step-down on the per-pair
            Hoeffding p-values). Anything else raises.

    Returns:
        A :class:`PoweredPartialOrder`.
    """
    if correction not in CORRECTIONS:
        raise ValueError(f"correction={correction!r}; expected one of {CORRECTIONS}")
    threshold = powered_gap_threshold(n_samples, alpha)
    if correction == "holm":
        return _certify_holm(scores, n_samples, alpha, threshold)
    names = list(scores.keys())
    certified: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            gap = scores[a] - scores[b]
            # crash→NaN contract: a non-finite gap (either score NaN/±inf — the
            # metric crashed on its cohort) is not a defensible ordering. Require a
            # finite gap explicitly: ``abs(inf) >= threshold`` is True, so without
            # this guard an ``inf`` score would spuriously certify the crashed
            # metric as the better one. Non-finite pairs are ambiguous.
            if math.isfinite(gap) and abs(gap) >= threshold:
                certified.append((a, b) if gap > 0 else (b, a))
            else:
                ambiguous.append((a, b))
    return PoweredPartialOrder(
        n_samples=n_samples,
        alpha=alpha,
        gap_threshold=threshold,
        certified=certified,
        ambiguous=ambiguous,
    )


def _certify_holm(
    scores: dict[str, float], n_samples: int, alpha: float, threshold: float
) -> PoweredPartialOrder:
    """Holm step-down over the pairs: the ``k``-th smallest p-value must clear ``alpha/(K-k)``."""
    names = list(scores)
    pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    pvals = [_pair_pvalue(scores[a] - scores[b], n_samples) for a, b in pairs]
    order = sorted(range(len(pairs)), key=lambda k: (pvals[k], k))
    rejected: set[int] = set()
    for step, k in enumerate(order):
        if pvals[k] > alpha / (len(pairs) - step):
            break
        rejected.add(k)
    certified: list[tuple[str, str]] = []
    ambiguous: list[tuple[str, str]] = []
    for k, (a, b) in enumerate(pairs):
        if k in rejected:
            certified.append((a, b) if scores[a] > scores[b] else (b, a))
        else:
            ambiguous.append((a, b))
    return PoweredPartialOrder(
        n_samples=n_samples,
        alpha=alpha,
        gap_threshold=threshold,
        certified=certified,
        ambiguous=ambiguous,
        correction="holm",
    )
