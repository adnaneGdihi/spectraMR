"""Per-metric gate statistics: violation rates, probes, concordance, certifiability.

Every statistic drops non-finite values, and one with nothing finite left returns
``None`` rather than a rate. A metric that returns NaN everywhere (a crash under the
safe-call adapter) would otherwise score 0 violations and pass as perfect.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from ..ebh import e_bh, sign_evalue

#: Below this share of finite metric values the metric is reported as carrying no signal.
MIN_FINITE_FRACTION = 0.5


def _finite(d: np.ndarray) -> np.ndarray:
    return d[np.isfinite(d)]


def violations(diffs: Sequence[np.ndarray], q: float) -> tuple[float | None, int | None]:
    """Share of relations whose mean difference is negative, and the e-BH count.

    ``diffs`` holds one per-content difference vector per relation (better minus worse).
    A relation with no finite content is left out of both.
    """
    kept = [_finite(d) for d in diffs]
    kept = [d for d in kept if d.size]
    if not kept:
        return None, None
    point = float(np.mean([d.mean() < 0 for d in kept]))
    certified = int(e_bh([sign_evalue(-d) for d in kept], q).sum())
    return point, certified


def equivalence_violations(
    diffs: Sequence[np.ndarray], q: float, t_min: float = 3.0
) -> tuple[float | None, int | None]:
    """Display sensitivity on Blackwell-equivalent pairs (Theorem C1.3(e)).

    An equivalent pair carries the same information, so a valid metric should not prefer
    either side. The point rate is the share of pairs whose paired t over contents
    exceeds ``t_min`` in either direction. The certified count tests both signs: the mean
    of the two one-sided e-values is again an e-value under the null that neither side
    is preferred.
    """
    kept = [_finite(d) for d in diffs]
    kept = [d for d in kept if d.size >= 2]
    if not kept:
        return None, None

    def _t(d: np.ndarray) -> float:
        sd = d.std(ddof=1)
        return (
            float(abs(d.mean()) / (sd / np.sqrt(d.size)))
            if sd > 0
            else (0.0 if d.mean() == 0 else np.inf)
        )

    point = float(np.mean([_t(d) > t_min for d in kept]))
    e = [(sign_evalue(d) + sign_evalue(-d)) / 2 for d in kept]
    return point, int(e_bh(e, q).sum())


def probe_violations(
    member_a: Sequence[np.ndarray], member_b: Sequence[np.ndarray], q: float
) -> tuple[float | None, int | None]:
    """Share of mixture probes rated below both members (Theorem C1.8(a)), and the share
    whose two preferences e-BH rejects together. Each input is member minus mixture."""
    pairs = [(_finite(a), _finite(b)) for a, b in zip(member_a, member_b, strict=True)]
    pairs = [(a, b) for a, b in pairs if a.size and b.size]
    if not pairs:
        return None, None
    point = float(np.mean([a.mean() > 0 and b.mean() > 0 for a, b in pairs]))
    rej = e_bh([sign_evalue(a) for a, _ in pairs] + [sign_evalue(b) for _, b in pairs], q)
    return point, int((rej[: len(pairs)] & rej[len(pairs) :]).sum())


def concordance(values: torch.Tensor, strict: Sequence[tuple[int, int]]) -> float:
    """Share of matched draws ordered as ``strict`` says, ties one half, NaNs dropped.

    ``values`` is ``(C, E, R)``. Draw ``r`` shares its noise across experiments, so
    ``values[c, i, r] - values[c, j, r]`` compares a matched pair.
    """
    if not strict:
        return float("nan")
    better = torch.tensor([i for i, _ in strict])
    worse = torch.tensor([j for _, j in strict])
    d = values[:, better] - values[:, worse]
    d = d[torch.isfinite(d)]
    if d.numel() == 0:
        return float("nan")
    return float((d > 0).double().mean() + 0.5 * (d == 0).double().mean())


def relation_gate(
    raw: torch.Tensor, sharp: torch.Tensor, graph: Any, q: float
) -> tuple[float, dict[str, object]]:
    """``(score, fields)`` for one metric on a relation graph.

    ``raw`` is ``(C, E, R)`` under the identity display and ``sharp`` the envelope's held-out
    draws. The score is the raw concordance. The fields are the canonicalised
    concordance, the violation rates and the equivalence display rate, each with its
    e-BH count.
    """
    mean_raw, mean_sharp = raw.mean(-1), sharp.mean(-1)
    bvr_raw, cert_raw = violations(
        [(mean_raw[:, i] - mean_raw[:, j]).numpy() for i, j in graph.strict], q
    )
    bvr_sharp, cert_sharp = violations(
        [(mean_sharp[:, i] - mean_sharp[:, j]).numpy() for i, j in graph.strict], q
    )
    eq_point, eq_cert = equivalence_violations(
        [(mean_raw[:, i] - mean_raw[:, j]).numpy() for i, j in graph.equivalent], q
    )
    return concordance(raw, graph.strict), {
        "concordance_sharp": concordance(sharp, graph.strict),
        "bvr_raw": bvr_raw,
        "bvr_raw_certified": cert_raw,
        "bvr_sharp": bvr_sharp,
        "bvr_sharp_certified": cert_sharp,
        "equivalence_display_rate": eq_point,
        "equivalence_display_certified": eq_cert,
    }


def certifiable(n_contents: int, n_hypotheses: int, q: float) -> bool:
    """Whether e-BH could reject even one of ``n_hypotheses`` with ``n_contents``.

    The largest sign e-value comes from all-positive signs. A single rejection needs
    ``e >= n_hypotheses / q``. When it cannot reach that, a certified count of zero
    means "could not certify", not "certified none".
    """
    if n_hypotheses == 0:
        return False
    return sign_evalue(np.ones(n_contents)) >= n_hypotheses / q


__all__ = [
    "MIN_FINITE_FRACTION",
    "certifiable",
    "concordance",
    "equivalence_violations",
    "probe_violations",
    "relation_gate",
    "violations",
]
