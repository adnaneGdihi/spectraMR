"""``lesion_vs_control`` -- the only comparison that answers the actual question.

Everything else in the region axis is descriptive. This is the inference.

The claim under test is *"metric M tracks degradation differently inside pathology
than in normal tissue"*. The naive test -- compare M's severity-correlation in
``path:lesion_any`` against its correlation on the whole slice -- **cannot support
that claim**, because the lesion ROI differs from the whole slice in two ways at
once: it is *pathological* and it is *small*. Any difference is therefore
attributable to either. That is pitfall #17 (confounded ablation), and it is fatal:
the experiment would produce a number that looks like an answer and is not one.

The matched control holds size constant. ``control:lesion_any`` is the **same
size**, in normal-appearing contralateral tissue, on the **same slice**, under the
**same artifact and severity** (:mod:`spectramr.data.annotations.matched_control`).
So ``delta = rho_lesion - rho_control`` varies only in the tissue, and only this
delta is interpretable.

Paired, not unpaired
--------------------
The two correlations come from the *same slices*. A slice that is intrinsically
hard (low SNR, motion-prone anatomy) depresses both, and an unpaired test would
count that shared, irrelevant variance as noise -- badly underpowering the design.
So the bootstrap resamples **slices**, not observations, and recomputes both
correlations on each draw. The CI is over the *paired* difference. When the
caller knows which subject each slice came from, the unit is the **subject**
instead: slices of one volume share anatomy (and a degradation realisation), so
resampling them as independent units would narrow the CI by pseudo-replication.

Resampling observations instead of slices would also break the pairing: a draw
could contain a lesion score from slice 7 and a control score from slice 12, which
is not a difference between tissues at all.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

__all__ = [
    "PairedDelta",
    "lesion_vs_control",
    "spearman_rho",
]


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation. NaN when it is not defined, never a fabricated 0.0.

    Undefined means undefined: fewer than 2 points, or a constant input (a metric
    that returned the same value at every severity has zero rank variance, so its
    correlation with severity has no value -- and reporting 0.0 would say "this
    metric is uncorrelated with severity", which is a *finding*, not a missing one).
    """
    if len(x) != len(y):
        raise ValueError(f"paired inputs must be the same length; got {len(x)}, {len(y)}")
    if len(x) < 2:
        return float("nan")

    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    if not (torch.isfinite(xt).all() and torch.isfinite(yt).all()):
        return float("nan")

    xr = _tied_ranks(xt)
    yr = _tied_ranks(yt)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = float(torch.linalg.vector_norm(xr) * torch.linalg.vector_norm(yr))
    if denom == 0.0:
        return float("nan")  # a constant input has no rank variance
    return float((xr @ yr) / denom)


def _tied_ranks(v: torch.Tensor) -> torch.Tensor:
    """Midranks: tied values share the AVERAGE of the ranks they span.

    ``argsort().argsort()`` is not a rank function -- it breaks ties arbitrarily.
    On a constant input it returns ``[0, 1, 2, ...]``, inventing an ordering that is
    not in the data, and the resulting "correlation" is a pure artifact of tie-break
    order. A metric that returns the same value at every severity would then score
    +-1 instead of the honest 'undefined'. Ties are common here: quantised metrics
    saturate, and a dead metric is constant by definition.
    """
    order = v.argsort()
    ranks = torch.empty_like(v, dtype=torch.float64)
    ranks[order] = torch.arange(v.numel(), dtype=torch.float64)

    sorted_v = v[order]
    i = 0
    while i < v.numel():
        j = i + 1
        while j < v.numel() and sorted_v[j] == sorted_v[i]:
            j += 1
        if j - i > 1:  # a tie run -- give every member the midrank
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


@dataclass(frozen=True, slots=True)
class PairedDelta:
    """One (metric, artifact, contrast) row of the money table."""

    metric: str
    artifact: str
    contrast: str
    rho_lesion: float
    rho_control: float
    delta: float
    ci_low: float
    ci_high: float
    n_slices: int
    n_boot: int
    resampling_unit: str = "slice"
    n_units: int = 0

    @property
    def significant(self) -> bool:
        """The CI excludes zero. NaN bounds are never 'significant'."""
        if not (math.isfinite(self.ci_low) and math.isfinite(self.ci_high)):
            return False
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def as_row(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "artifact": self.artifact,
            "contrast": self.contrast,
            "rho_lesion": round(self.rho_lesion, 4),
            "rho_control": round(self.rho_control, 4),
            "delta": round(self.delta, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "significant": self.significant,
            "n_slices": self.n_slices,
            "n_boot": self.n_boot,
            "resampling_unit": self.resampling_unit,
            "n_units": self.n_units,
        }


def lesion_vs_control(
    metric: str,
    artifact: str,
    contrast: str,
    *,
    severity: Sequence[float],
    lesion_scores: Sequence[float],
    control_scores: Sequence[float],
    slice_ids: Sequence[str],
    subject_ids: Sequence[str] | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDelta:
    """Bootstrap the paired difference in severity-correlation, lesion vs control.

    Args:
        severity: theta at each observation.
        lesion_scores: the metric inside ``path:lesion_any``.
        control_scores: the metric inside its size-matched control -- **same slice,
            same artifact, same severity**.
        slice_ids: which slice each observation came from. The bootstrap resamples
            **these**, not the observations, so the pairing survives every draw.
        subject_ids: optional subject of each observation. When given, the bootstrap
            resamples subjects (every slice of a drawn subject comes along) instead.

    Returns:
        A :class:`PairedDelta`. ``significant`` is True only when the CI on
        ``rho_lesion - rho_control`` excludes zero.
    """
    n = len(severity)
    if not (len(lesion_scores) == len(control_scores) == len(slice_ids) == n):
        raise ValueError(
            "severity, lesion_scores, control_scores and slice_ids must align: got "
            f"{n}, {len(lesion_scores)}, {len(control_scores)}, {len(slice_ids)}. "
            "A misalignment here would compare a lesion on one slice against a "
            "control on another -- which is not a tissue difference at all."
        )

    rho_l = spearman_rho(severity, lesion_scores)
    rho_c = spearman_rho(severity, control_scores)
    delta = rho_l - rho_c

    n_slices = len(set(slice_ids))
    unit_ids, unit = _resampling_units(slice_ids, subject_ids)
    by_unit: dict[str, list[int]] = {}
    for i, uid in enumerate(unit_ids):
        by_unit.setdefault(uid, []).append(i)
    units = sorted(by_unit)
    n_units = len(units)

    if n_units < 2 or not math.isfinite(delta):
        return PairedDelta(
            metric=metric,
            artifact=artifact,
            contrast=contrast,
            rho_lesion=rho_l,
            rho_control=rho_c,
            delta=delta,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_slices=n_slices,
            n_boot=0,
            resampling_unit=unit,
            n_units=n_units,
        )

    gen = torch.Generator().manual_seed(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        pick = torch.randint(0, n_units, (n_units,), generator=gen).tolist()
        idx = [i for p in pick for i in by_unit[units[p]]]

        d = spearman_rho(
            [severity[i] for i in idx], [lesion_scores[i] for i in idx]
        ) - spearman_rho([severity[i] for i in idx], [control_scores[i] for i in idx])
        if math.isfinite(d):
            deltas.append(d)

    if len(deltas) < 2:
        return PairedDelta(
            metric=metric,
            artifact=artifact,
            contrast=contrast,
            rho_lesion=rho_l,
            rho_control=rho_c,
            delta=delta,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_slices=n_slices,
            n_boot=len(deltas),
            resampling_unit=unit,
            n_units=n_units,
        )

    t = torch.tensor(deltas, dtype=torch.float64)
    lo = float(torch.quantile(t, alpha / 2))
    hi = float(torch.quantile(t, 1 - alpha / 2))

    return PairedDelta(
        metric=metric,
        artifact=artifact,
        contrast=contrast,
        rho_lesion=rho_l,
        rho_control=rho_c,
        delta=delta,
        ci_low=lo,
        ci_high=hi,
        n_slices=n_slices,
        n_boot=len(deltas),
        resampling_unit=unit,
        n_units=n_units,
    )


def _resampling_units(
    slice_ids: Sequence[str], subject_ids: Sequence[str] | None
) -> tuple[Sequence[str], str]:
    """The bootstrap unit of each observation: its subject when known, else its slice."""
    if subject_ids is None:
        return slice_ids, "slice"
    if len(subject_ids) != len(slice_ids):
        raise ValueError(
            f"subject_ids has {len(subject_ids)} entries for {len(slice_ids)} observations."
        )
    owner: dict[str, str] = {}
    for sid, subj in zip(slice_ids, subject_ids, strict=True):
        if not subj:
            raise ValueError(f"slice {sid!r} has an empty subject id.")
        if owner.setdefault(sid, subj) != subj:
            raise ValueError(
                f"slice {sid!r} is assigned to subject {owner[sid]!r} and {subj!r}; a slice "
                "belongs to exactly one subject."
            )
    return subject_ids, "subject"


def money_table(rows: Sequence[Mapping[str, object]], **kw: object) -> list[PairedDelta]:
    """Convenience: run :func:`lesion_vs_control` over pre-grouped rows."""
    return [
        lesion_vs_control(
            str(r["metric"]),
            str(r["artifact"]),
            str(r["contrast"]),
            severity=r["severity"],  # type: ignore[arg-type]
            lesion_scores=r["lesion_scores"],  # type: ignore[arg-type]
            control_scores=r["control_scores"],  # type: ignore[arg-type]
            slice_ids=r["slice_ids"],  # type: ignore[arg-type]
            subject_ids=r.get("subject_ids"),  # type: ignore[arg-type]
            **kw,  # type: ignore[arg-type]
        )
        for r in rows
    ]
