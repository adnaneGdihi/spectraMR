"""Shared dataclasses for the meta-evaluation framework.

Five rankers (CDSCR, LGDR, SFA, ESD, FSPD) each consume a ``MetricSet`` and
a ``MetricEvaluationDataset`` and emit a ``RankingResult``. Keeping all three
contracts here avoids circular imports between ranker modules.

The container types here are intentionally light. They are *not* Pydantic
models — meta-evaluation runs as an offline analysis after training, not
inside the frozen ``TrainingSettings`` graph, and Pydantic's validation cost
on per-sample tensors would dominate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from spectramr.core.metrics.context import MetricContext

# A metric callable: (pred, target, **kwargs) -> scalar. Matches the existing
# ``IMetric`` protocol in src/core/metrics/registry.py so any metric already
# registered there can be ranked without an adapter.
MetricCallable = Callable[..., float]


@dataclass
class MetricSet:
    """Bundle of metrics under evaluation.

    ``higher_is_better`` is required because rankers that look at gradient
    direction (SFA) and group-action sensitivity (ESD) need to know which sign
    convention each metric uses. Defaults to ``False`` (lower is better — the
    common case for distortion metrics like RMSE/HFEN).
    """

    metrics: dict[str, MetricCallable]
    higher_is_better: dict[str, bool] = field(default_factory=dict)
    differentiable: dict[str, bool] = field(default_factory=dict)

    def names(self) -> list[str]:
        return list(self.metrics.keys())

    def is_higher_better(self, name: str) -> bool:
        return self.higher_is_better.get(name, False)

    def is_differentiable(self, name: str) -> bool:
        # SFA falls back to SPSA when this is False.
        return self.differentiable.get(name, True)


@dataclass
class DegradationSample:
    """One (clean, degraded) pair plus its degradation parameters.

    Tensors are kept on CPU until a ranker explicitly moves them — avoids
    unnecessary host-device traffic when sweeping thousands of severities.
    """

    clean: torch.Tensor  # [C, H, W] or [C, D, H, W]
    degraded: torch.Tensor  # same shape as ``clean``
    degradation_family: str  # e.g. "gaussian_noise", "motion", "undersampling"
    severity: dict[str, float]  # parameter name -> value (Sobol coordinate)
    content_id: str  # subject / slice identifier — used as covariate C
    seed: int
    # The person scanned; slices and contrasts of one subject are not independent
    # replicates. ``None`` means unknown, and consumers fall back to ``content_id``.
    subject_id: str | None = field(default=None, kw_only=True)
    # Real acquisition assets for this content (coil maps, acquired k-space,
    # reconstructor, noise cov, acq params). Populated by the asset-preparation
    # stage when the input carries them (multi-coil k-space); ``None`` for a
    # magnitude-only cohort. When present, the NR/physics context is built from
    # THESE instead of the synthetic birdcage stand-in.
    assets: MetricContext | None = None

    @property
    def residual(self) -> torch.Tensor:
        return self.clean - self.degraded

    @property
    def residual_energy(self) -> float:
        return float(torch.linalg.vector_norm(self.residual).item())

    @property
    def rel_l2(self) -> float:
        """Relative L2 of the residual -- the scale-free MEASURED distortion this
        step actually produced (how much noise was added), not the declared
        severity. Companion to :attr:`residual_energy` (absolute).

        PROVENANCE / diagnostic ONLY: it is a monotone function of the L2-family
        metrics (mse / psnr / nrmse / nmse), so ranking those against it would be
        circular. See ``meta_evaluation.damage.YARDSTICKS``, which
        declares exactly that disqualification. Mirrors its ``rel_l2``.
        """
        den = torch.linalg.vector_norm(self.clean).clamp_min(1e-12)
        return float((torch.linalg.vector_norm(self.residual) / den).item())


@dataclass
class MetricEvaluationDataset:
    """Frozen collection of ``DegradationSample`` plus per-metric scalar values.

    The metric values are pre-computed once and reused across rankers — most
    rankers (CDSCR, LGDR, FSPD) only need the scalars, not the tensors.
    """

    samples: list[DegradationSample]
    metric_values: dict[str, list[float]]  # metric_name -> [N] aligned with samples

    def __post_init__(self) -> None:
        n = len(self.samples)
        for name, values in self.metric_values.items():
            if len(values) != n:
                raise ValueError(
                    f"metric_values['{name}'] has {len(values)} entries but there are {n} samples"
                )

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def metric_array(self, name: str) -> torch.Tensor:
        return torch.tensor(self.metric_values[name], dtype=torch.float64)

    def residual_energies(self) -> torch.Tensor:
        return torch.tensor([s.residual_energy for s in self.samples], dtype=torch.float64)

    def severity_matrix(self, families: Sequence[str], params: Sequence[str]) -> torch.Tensor:
        """Stack severity coordinates into an [N, len(params)] matrix.

        Missing values (sample's degradation family does not include the param)
        are filled with 0. This is the matrix consumed by FSPD.
        """
        rows = []
        for s in self.samples:
            row = [float(s.severity.get(p, 0.0)) for p in params]
            rows.append(row)
        return torch.tensor(rows, dtype=torch.float64)


@dataclass
class RankingResult:
    """Output of ``BaseRanker.rank``.

    ``scores`` is the raw per-metric statistic the ranker computes — its sign
    and scale depend on the method. ``ranks`` is the ordered list of metric
    names from best to worst (best = highest score after the ranker's own
    sign convention is applied). ``diagnostics`` holds method-specific extras
    (eigenstructure, profiles, confidence bands) used by the report figures.
    """

    method: str
    scores: dict[str, float]
    ranks: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def standardize(self, universe: Iterable[str] = ()) -> dict[str, float]:
        """Return z-scored per-metric scores, covering ``scores`` and ``universe``.

        Used by the aggregator. The sign of the score is preserved — ranker
        implementations are responsible for ensuring "higher score = better
        metric" before this point. The mean and std come from the finite scores
        only, so one crash cannot poison every z. Under the crash→NaN contract a
        metric that is NaN, ``-inf`` or absent from ``scores`` (but named in
        ``universe``) did not score on this axis and is placed one unit **below**
        the worst finite z; ``+inf`` (a legitimate best case, e.g. SCVR) one unit
        above the best. Omitting a crash instead let a metric average over only the
        axes where it worked, and so rise (the same contract as
        :func:`ranking_guards.order_from_scores`).
        """
        finite = {k: v for k, v in self.scores.items() if math.isfinite(v)}
        z: dict[str, float] = {}
        if finite:
            values = torch.tensor(list(finite.values()), dtype=torch.float64)
            mean = float(values.mean().item())
            std = float(values.std(unbiased=False).item())
            if std < 1e-12:
                z = dict.fromkeys(finite, 0.0)
            else:
                z = {k: (v - mean) / std for k, v in finite.items()}
        worst = min(z.values(), default=0.0) - 1.0
        best = max(z.values(), default=0.0) + 1.0
        for k in (set(self.scores) | set(universe)) - set(finite):
            z[k] = best if self.scores.get(k) == math.inf else worst
        return z


@dataclass
class DefensibilityCalibration:
    """Permutation-calibrated false-positive control for the defensibility flag.

    The raw defensibility flag (positive on ``>= min_positive_axes`` axes, no axis
    below ``floor_z``) is a deterministic rule with no calibrated false-positive
    rate: because the rankers share trajectories and the metric pool, "positive on
    5 of 6 axes" is *not* a product of independent marginals. This dataclass carries
    the exchangeability-null calibration (critique §2):

    * ``null_fpr`` — empirical probability that a label-exchanged null metric fires
      the flag (the flag's actual false-positive rate at the current thresholds).
    * ``pvalues`` — per-metric permutation p-value of the breadth-of-agreement
      statistic (number of positive axes) against the pooled exchangeability null.
    * ``calibrated_flags`` — the raw flag AND ``p <= alpha``: defensible *and*
      beyond chance.
    """

    null_fpr: float
    pvalues: dict[str, float]
    calibrated_flags: dict[str, bool]
    n_perm: int
    alpha: float


@dataclass
class AggregateRankingResult:
    """Aggregated five-method ranking with defensibility flag per metric."""

    final_scores: dict[str, float]
    final_ranks: list[str]
    per_method_z: dict[str, dict[str, float]]  # method -> {metric -> z}
    defensibility_flags: dict[str, bool]
    weights: dict[str, float]
    # Populated only when AggregatorConfig.calibrate is set (permutation cost).
    calibration: DefensibilityCalibration | None = None
    # What ``defensibility_flags`` *means* for this aggregator — the two
    # cross-axis combiners write different events into the same field:
    #   * "breadth"   — z-weighted ``aggregate``: positive on >= min_positive_axes
    #                   axes, none below floor_z (many metrics can be defensible).
    #   * "condorcet" — ``kemeny_consensus``: a strict Condorcet winner (margin > 0
    #                   vs every opponent; at most one metric, except ties).
    # Stamped so a bundle is self-describing and downstream consumers
    # (to_summary / comparison) never compare flags across incompatible semantics.
    defensibility_semantics: str = "breadth"
    # 1-indexed mid-ranks aligned with ``final_ranks``: equal scores share a rank.
    final_mid_ranks: dict[str, float] = field(default_factory=dict)


def asdict_safe(obj: Any) -> Any:
    """Best-effort conversion to plain dict / list for JSON dumping.

    The diagnostics fields can hold tensors and numpy arrays — this normalizes
    them. Missing torch / numpy dependencies are tolerated for type stubs.
    """
    if isinstance(obj, Mapping):
        return {k: asdict_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [asdict_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj
