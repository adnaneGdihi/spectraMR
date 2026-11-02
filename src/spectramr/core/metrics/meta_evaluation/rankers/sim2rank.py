"""Sim2Rank Composite Ranker.

Folds the three most distinctive scoring axes of the legacy
``scripts/sim2rank`` framework into a single ``BaseRanker``-compliant
implementation:

1. **ADR** — Asymptotic Discriminability Ranking
   ``S_ADR = |rho| × (non-saturated fraction)`` per metric trajectory,
   averaged over degradation families. Captures monotonic + non-saturated
   tracking of severity (Sheikh et al. 2006, IEEE TIP).

2. **SCVR** — Signal-to-Content Variance Ratio
   Two-way ANOVA F-statistic on the ``(metric, content, family, severity)``
   tensor. Measures how much of a metric's variance is driven by *severity*
   (signal) versus *content* (anatomy nuisance) — Ponomarenko et al. 2015.

3. **CDRS** — Clean-to-Degraded Robustness Score
   ``CDRS = C_m × M_m`` where ``C_m`` is clean-baseline stability
   ``exp(-CV)`` and ``M_m`` is mean Kendall tau across families penalised
   by its inter-family variance.

The three sub-scores are min-max-normalised to ``[0, 1]`` and combined
with weights ``(0.5, 0.3, 0.2)`` matching the legacy composite. The
ranker emits the composite as ``scores`` and the per-axis breakdowns in
``diagnostics``.

This ranker is *complementary* to FSPD: FSPD does FPCA on the response
surface across families × severities, whereas Sim2Rank scores each
trajectory independently and explicitly partials out content variance —
the two rank disagreements are the strongest meta-evaluation signal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..ranking_guards import assert_not_degenerate
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker


@dataclass
class Sim2RankConfig:
    epsilon: float = 0.01  # ADR saturation threshold; read only when nonsat="epsilon"
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)  # ADR, SCVR, CDRS
    # ADR non-saturation estimator -- see _increment_perplexity for why this is the
    # only one of the three that discriminates. "epsilon" (legacy, coarse) and
    # "strict_rank" (the L1/§9 repair, which turns out to be degenerate on real-valued
    # data) are retained for reproduction and as an invariance diagnostic.
    nonsat: str = "perplexity"
    # §18/L10: sub-score normaliser. "rank" (midrank fraction — invariant to
    # outliers and to monotone reparam) or "minmax" (legacy, outlier-fragile: one
    # metric with a 1e8 SCVR F-ratio flattens every other metric to ~0, #240).
    normalizer: str = "rank"


# ---------------------------------------------------------------------------
# Statistical primitives — reimplemented in torch to keep zero-dep parity
# ---------------------------------------------------------------------------


def _spearman_rho(x: torch.Tensor, y: torch.Tensor) -> float:
    """Ordinal (tie-BLIND) Spearman rank correlation; NaN-safe.

    Ranks via ``argsort(argsort)`` — ties get distinct ordinal ranks, so this is NOT
    mid-rank tie-corrected and diverges from ``scipy.stats.spearmanr`` on tied inputs.

    .. warning::

        Retained only for :mod:`spectramr.application.use_cases.nr_validation.cards`,
        which computes a Spearman on tie-free continuous data. **Do not use it in a
        ranker**: on a saturated (partially tied) trajectory it returns ``|rho| = 1``
        and over-credits the metric. ADR / CDRS / Sim2Rank use
        :func:`_spearman_midrank` instead (critique H1).
    """
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return 0.0
    xm = x[mask].double()
    ym = y[mask].double()
    rx = torch.argsort(torch.argsort(xm)).double()
    ry = torch.argsort(torch.argsort(ym)).double()
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float((rx.norm() * ry.norm()).item())
    if denom < 1e-12:
        return 0.0
    return float((rx * ry).sum().item() / denom)


def _kendall_tau(x: torch.Tensor, y: torch.Tensor) -> float:
    """O(T^2) Kendall tau; vectorised. Used inside CDRS."""
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return 0.0
    xm = x[mask].double()
    ym = y[mask].double()
    sign_x = torch.sign(xm.unsqueeze(0) - xm.unsqueeze(1))
    sign_y = torch.sign(ym.unsqueeze(0) - ym.unsqueeze(1))
    tri = torch.triu(torch.ones_like(sign_x), diagonal=1).bool()
    n_pairs = float(tri.sum().item())
    if n_pairs < 1:
        return 0.0
    concordance = (sign_x * sign_y)[tri].sum().item()
    return float(concordance) / n_pairs


def _midranks(x: torch.Tensor) -> torch.Tensor:
    """0-based average (mid) ranks of a 1-D tensor; tied values share a rank.

    Midranks depend only on the order/equality relations among entries, so they
    are invariant under every strictly increasing pointwise reparameterisation —
    the property both the ADR strict-rank fraction and the rank normaliser rely on.
    """
    xm = x.double()
    n = xm.numel()
    order = torch.argsort(xm)
    sorted_x = xm[order]
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg = (i + j) / 2.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman_midrank(x: torch.Tensor, y: torch.Tensor) -> float:
    """Tie-corrected (mid-rank) Spearman correlation; NaN-safe.

    Unlike :func:`_spearman_rho` (tie-BLIND ``argsort(argsort)``, which returns
    ``|rho| = 1`` on a saturated/partially-tied trajectory), ties share their
    average rank — matching ``scipy.stats.spearmanr``. ADR uses this so a metric
    that saturates over a sub-range is not over-credited on the tracking factor
    (critique H1).
    """
    mask = torch.isfinite(x) & torch.isfinite(y)
    if int(mask.sum().item()) < 3:
        return 0.0
    rx = _midranks(x[mask])
    ry = _midranks(y[mask])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float((rx.norm() * ry.norm()).item())
    if denom < 1e-12:
        return 0.0
    return float((rx * ry).sum().item() / denom)


# ---------------------------------------------------------------------------
# Per-axis scoring components
# ---------------------------------------------------------------------------


def compute_adr(
    trajectory: torch.Tensor,
    epsilon: float = 0.01,
    higher_is_better: bool = False,
    nonsat: str = "perplexity",
) -> float:
    r"""Asymptotic Discriminability Ranking score for a single trajectory.

    .. math::
        S_{ADR}(m) = \underbrace{\max(0,\, -s\rho)}_{\text{directed tracking}}
                     \cdot \underbrace{\frac{1}{T-1}\sum_{t=1}^{T-1}
                     \mathbb{1}\!\left[R_{t+1} \neq R_t\right]}_{\text{non-saturation}}

    where :math:`\rho` is the tie-corrected Spearman correlation of the trajectory
    against the severity index, :math:`s = +1` when ``higher_is_better`` and
    :math:`-1` otherwise, and :math:`R_t` are the midranks of the trajectory.

    **Direction is load-bearing** (#237). The score uses the *signed* correlation:
    a higher-is-better metric must *decrease* with severity (:math:`\rho < 0`) to
    earn credit. The previous ``|\rho|`` form was blind to sign, so a metric that
    responded in the wrong direction — a bugged implementation, an inverted sign
    convention, a similarity metric mislabelled as a distance — scored a perfect
    1.0. It now scores 0. This also makes ``higher_is_better`` a real parameter:
    under ``|\rho|`` (and ``|\nabla|``) it had *no effect whatsoever* on the output.

    **Non-saturation** (``nonsat``, default ``"perplexity"``):

    * ``"perplexity"`` (default) — the increment-perplexity. Saturation lives in the
      *concentration* of the normalised step-increments, not their mean, so this is
      the only one of the three that separates a linear metric (perplexity ≈ T) from
      a hard-saturating one (perplexity ≈ 1) (#241). ``epsilon`` is **not read** here.
    * ``"strict_rank"`` — the fraction of adjacent steps whose *midrank* strictly
      changes. Exactly invariant under every strictly increasing reparameterisation of
      the unitless severity index, but **degenerate on real-valued data**: real
      trajectories never tie, so every strictly-monotone metric scores 1.0 (one
      distinct score across the whole pool) and a linear metric is indistinguishable
      from a hard-saturating one. Retained only as an invariance diagnostic; despite
      the L1/§9 write-up it is *not* the production default.
    * ``"epsilon"`` — the legacy eps-thresholded increment count, and *not*
      reparameterisation-invariant: the same metric under the same physics scored
      0.947 / 0.895 / 1.000 under :math:`\theta`, :math:`\sqrt{\theta}`,
      :math:`\theta^2` (#241, L1/§9). Retained only to reproduce a pre-fix run;
      must be requested explicitly.

    .. warning::

        Compute this on **raw** trajectories. Feeding it the output of
        ``isotonic_calibrate`` is not a discriminability test: the calibrator is the
        monotone projection of *m* onto the severity grid, so for any monotone
        metric the calibrated trajectory *is* the severity grid exactly, giving
        :math:`\rho = 1` and a non-saturation factor of 1 — i.e. **ADR = 1.0 for
        every monotone metric**, tied (#236).

    Returns 0 for constant, NaN-dominated, or wrong-direction trajectories.

    Raises:
        ValueError: if ``nonsat`` is not one of ``"perplexity"``, ``"epsilon"``,
            ``"strict_rank"``.
    """
    if nonsat not in ("perplexity", "epsilon", "strict_rank"):
        raise ValueError(
            f"unknown nonsat {nonsat!r}; expected 'perplexity', 'epsilon' or 'strict_rank'"
        )

    traj = trajectory.detach().double()
    finite = torch.isfinite(traj)
    if int(finite.sum().item()) < 3:
        return 0.0
    if float(finite.float().mean().item()) < 0.5:
        return 0.0
    # Replace remaining NaNs with column mean.
    if (~finite).any():
        mean = float(traj[finite].mean().item())
        traj = torch.where(finite, traj, torch.full_like(traj, mean))

    tmin, tmax = float(traj.min().item()), float(traj.max().item())
    if tmax - tmin < 1e-12:
        return 0.0
    norm = (traj - tmin) / (tmax - tmin)

    timestep = torch.arange(traj.numel(), dtype=torch.float64)
    # Tie-corrected Spearman (critique H1): a saturated/tied trajectory must not
    # score |rho| = 1 the way the tie-blind argsort(argsort) estimator does.
    rho = _spearman_midrank(norm, timestep)
    # Signed: a higher-is-better metric must fall with severity to earn credit.
    sign = 1.0 if higher_is_better else -1.0
    tracking = max(0.0, -sign * rho)

    grads = (norm[1:] - norm[:-1]).abs()
    if nonsat == "perplexity":
        saturation_ratio = _increment_perplexity(grads)
    elif nonsat == "strict_rank":
        mid = _midranks(norm)
        changes = (mid[1:] != mid[:-1]).double()
        saturation_ratio = float(changes.mean().item()) if changes.numel() else 0.0
    else:
        saturation_ratio = float((grads > epsilon).float().mean().item())

    return tracking * saturation_ratio


def _increment_perplexity(grads: torch.Tensor) -> float:
    r"""Effective fraction of severity steps over which the metric is still responding.

        .. math::

            p_t =
    rac{|\Delta m_t|}{\sum_u |\Delta m_u|}, \qquad
                    ext{non\_sat} =
    rac{\exp H(p)}{T-1}

        The **perplexity** of the increment distribution, normalised by the number of
        steps. This is the continuous form of the quantity the legacy binary indicator was
        trying to count — "how many steps is this metric actually still moving on" — and it
        is the only one of the three that works:

        ========================  ==========  =========  ============  ==========
        trajectory                perplexity  epsilon    strict_rank   want
        ========================  ==========  =========  ============  ==========
        perfectly linear             1.000       1.00        1.00       high
        mild saturation              0.871       1.00        1.00       ~high
        hard saturation              0.359       0.63        1.00       low
        one step then flat           0.053       0.05        0.05       ~0
        ========================  ==========  =========  ============  ==========

        ``strict_rank`` (the L1/§9 repair) is **degenerate**: it only detects *exact* ties,
        which real-valued data essentially never has, so every strictly monotone metric
        scores 1.0 (measured: 1 distinct score across 40 metrics). ``epsilon`` is a
        fraction-of-steps count, so it takes only ``T`` values (5 distinct across 40).
        Perplexity gave 40 distinct across 40.

        Why any ``mean(f(|Δm|))`` form fails: the increments of a min-max normalised
        monotone trajectory **sum to 1** by telescoping, so their mean is *always*
        ``1/(T-1)`` regardless of shape. Saturation lives in the increments'
        **concentration**, not their average — which is also exactly why the Jacobian's
        per-cell normalisation produced a constant matrix (#242).

        .. note::

            This is deliberately **not** invariant to reparameterising the severity grid.
            Saturation is only meaningful relative to a severity axis that means something;
            damage bands (:mod:`spectramr.core.metrics.meta_evaluation.damage`) give it one (#245).
            On a unitless index, any saturation measure can be reparameterised away —
            which is the real content of critique L1.
    """
    total = float(grads.sum().item())
    n = grads.numel()
    if n == 0 or total < 1e-12:
        return 0.0
    p = grads.double() / total
    p = p[p > 0]
    entropy = float(-(p * p.log()).sum().item())
    return float(math.exp(entropy) / n)


def compute_scvr(metric_tensor: torch.Tensor) -> float:
    """Severity-signal-to-content-nuisance variance ratio on a ``(S, C, T)`` tensor.

    The two leading dims are **content** ``S`` (subject/anatomy) and degradation
    **family** ``C``; the last is severity ``T``. The statistic is the ratio of the
    severity main-effect mean-square (signal) to the *content* main-effect
    mean-square (anatomy nuisance):

    .. math::
        \\mathrm{SCVR} = \\frac{\\mathrm{MS}_{\\text{severity}}}{\\mathrm{MS}_{\\text{content}}}

    The content nuisance averages over **both** family and severity, so it isolates
    pure anatomy variance: a metric whose discriminability is *family-specific* must
    NOT have that family variance counted as nuisance (critique B3). This is a
    variance ratio, **not** a calibrated two-way ANOVA F-statistic — the residual /
    interaction mean-square is unrecoverable from the per-cell-mean tensor (the
    within-cell replicates were averaged away upstream), so no F null applies
    (critique H5).

    Returns ``+inf`` when there is severity signal but no estimable content
    nuisance (a single content, or homogeneous anatomy) — the *best* case, which
    the sub-score normaliser maps to the top. Returns 0 for degenerate inputs.
    """
    if metric_tensor.ndim != 3:
        raise ValueError("metric_tensor must be (S, C, T)")
    S, C, T = metric_tensor.shape
    finite = torch.isfinite(metric_tensor)
    if int(finite.sum().item()) < 4 or T < 2 or (S * C) < 2:
        return 0.0
    safe = torch.where(finite, metric_tensor, torch.zeros_like(metric_tensor))
    counts = finite.float()

    grand_count = counts.sum().clamp(min=1.0)
    grand_mean = safe.sum() / grand_count

    # Severity main effect (signal): per-timestep mean across content AND family.
    tcounts = counts.sum(dim=(0, 1)).clamp(min=1.0)
    tmean = safe.sum(dim=(0, 1)) / tcounts
    ss_deg = (((tmean - grand_mean) ** 2) * tcounts).sum()
    ms_deg = ss_deg / max(T - 1, 1)

    # Content main effect (nuisance): per-content mean across family AND severity.
    # Averaging over family keeps degradation-family variance OUT of the nuisance.
    content_counts = counts.sum(dim=(1, 2)).clamp(min=1.0)
    content_mean = safe.sum(dim=(1, 2)) / content_counts
    ss_content = (((content_mean - grand_mean) ** 2) * content_counts).sum()
    ms_content = ss_content / max(S - 1, 1)

    if float(ms_content.item()) < 1e-12:
        return float("inf") if float(ms_deg.item()) > 0 else 0.0
    return float(ms_deg.item() / ms_content.item())


def compute_cdrs(
    clean_scores: list[float],
    family_trajectories: list[torch.Tensor],
    higher_is_better: bool = False,
) -> tuple[float, float, float]:
    r"""Clean-to-Degraded Robustness Score.

    ``CDRS = C_m × M_m`` where:

    * ``C_m = exp(-std(clean) / scale)`` is the dispersion-based stability of the
      metric on clean content. The ``scale`` is ``max(|mean|, std, eps)`` — a robust
      magnitude. Using a plain coefficient of variation ``std / |mean|`` blows up for
      a metric *centred at zero* (a signed/zero-mean metric, or any error metric,
      which evaluates to ~0 on a clean self-comparison), spuriously scoring it
      maximally unstable regardless of its actual dispersion (critique M1, #238); the
      ``max(|mean|, std)`` floor caps the relative dispersion at 1 in that regime.
    * ``M_m = mean_d tau_d^+ × (1 - std_d tau_d^+)`` averages the **directed** Kendall
      tau of every family trajectory — :math:`\tau^+ = \max(0, -s\tau)`, with
      :math:`s = +1` when ``higher_is_better`` — and penalises it by the **standard
      deviation** (not the variance — they have different units; ``1 - std`` is the
      dimensionally-consistent penalty) across families.

      Direction matters here for the same reason it does in :func:`compute_adr`
      (#237): under the previous ``|tau|`` a metric that tracked severity *backwards*
      earned full monotonicity credit.

    .. note::

        ``clean_scores`` must be genuine **across-anatomy** baselines (distinct clean
        subjects / contrasts). Feeding noise-perturbed copies of a single clean image
        drives the variance to 0, so ``C_m -> 1`` for every metric and the test is
        uninformative (#238).

    Returns ``(cdrs, c_m, m_m)``.
    """
    eps = 1e-9
    clean = torch.tensor([c for c in clean_scores if math.isfinite(c)], dtype=torch.float64)
    if clean.numel() < 2:
        c_m = 1.0
    else:
        mean = float(clean.mean().abs().item())
        std = float(clean.std(unbiased=False).item())
        scale = max(mean, std, eps)
        c_m = float(math.exp(-std / scale))

    sign = 1.0 if higher_is_better else -1.0
    taus = []
    for traj in family_trajectories:
        if traj.numel() < 3:
            continue
        idx = torch.arange(traj.numel(), dtype=torch.float64)
        taus.append(max(0.0, -sign * _kendall_tau(traj, idx)))
    if not taus:
        return 0.0, c_m, 0.0
    tau_t = torch.tensor(taus, dtype=torch.float64)
    mean_tau = float(tau_t.mean().item())
    std_tau = float(tau_t.std(unbiased=False).item())
    penalty = max(0.0, 1.0 - std_tau)
    m_m = mean_tau * penalty
    return c_m * m_m, c_m, m_m


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _dataset_axes(
    dataset: MetricEvaluationDataset,
) -> tuple[list[float], list[str], list[str], list[int]]:
    """Return the metric-independent axes of ``dataset`` plus a scatter index.

    ``severities``/``contents``/``families`` depend only on ``dataset.samples``,
    never on which metric is being tabulated, yet :func:`_build_trajectory_table`
    is called once per metric — at 139 metrics that recomputed three sorted sets
    over every sample 139 times over. They are resolved once here and memoised on
    the dataset instance.

    The fourth element is ``flat_index``: for sample ``i``, the offset of its slot
    in a flattened ``(content, family, severity)`` grid. It lets the per-metric
    fill become a flat list assignment instead of a tuple build plus two dict
    lookups per sample.

    Cache validity: the stamp is ``(id(samples), len(samples))``. The dataset
    holds a reference to its own ``samples`` list, so that id cannot be recycled
    while the cache is reachable — the stamp therefore detects both a reassigned
    and a resized ``samples``. It does NOT detect a ``content_id`` /
    ``degradation_family`` / ``severity`` mutated in place on a sample already in
    the list; ``MetricEvaluationDataset`` documents itself as a frozen collection
    built once and reused across rankers, and nothing in-tree mutates one.
    """
    stamp = (id(dataset.samples), len(dataset.samples))
    cached = getattr(dataset, "_axes_cache", None)
    if cached is not None and cached[0] == stamp:
        return tuple(cached[1])

    samples = dataset.samples
    thetas = [float(s.severity.get("theta", 0.0)) for s in samples]
    severities = sorted(set(thetas))
    contents = sorted({s.content_id for s in samples})
    families = sorted({s.degradation_family for s in samples})

    sev_index = {v: i for i, v in enumerate(severities)}
    content_index = {c: i for i, c in enumerate(contents)}
    family_index = {f: i for i, f in enumerate(families)}
    n_fam, n_sev = len(families), len(severities)
    flat_index = []
    for s, theta in zip(samples, thetas, strict=True):
        pair = content_index[s.content_id] * n_fam
        pair += family_index[s.degradation_family]
        flat_index.append(pair * n_sev + sev_index[theta])

    payload = (severities, contents, families, flat_index)
    dataset._axes_cache = (stamp, payload)  # type: ignore[attr-defined]
    return payload


def _build_trajectory_table(
    dataset: MetricEvaluationDataset, metric_name: str
) -> tuple[dict[tuple[str, str], list[float]], list[float], list[str], list[str]]:
    """Return per-(content, family) trajectories sorted by severity.

    Output:
        traj_by_pair: dict {(content_id, family) -> [v1, ..., vT]} where the
            list is ordered along the sorted severity grid.
        severities: ordered list of unique theta values.
        contents: ordered list of unique content_ids.
        families: ordered list of unique family names.

    The axes come from :func:`_dataset_axes` (memoised per dataset); only the
    scatter of this metric's values is per-metric work. Two samples that land on
    the same ``(content, family, theta)`` slot resolve last-wins, exactly as the
    per-sample assignment loop this replaced did.
    """
    severities, contents, families, flat_index = _dataset_axes(dataset)
    n_sev = len(severities)

    grid = [float("nan")] * (len(contents) * len(families) * n_sev)
    for pos, v in zip(flat_index, dataset.metric_values[metric_name], strict=True):
        grid[pos] = float(v)

    traj: dict[tuple[str, str], list[float]] = {}
    off = 0
    for c in contents:
        for f in families:
            traj[(c, f)] = grid[off : off + n_sev]
            off += n_sev
    return traj, severities, contents, families


def _per_family_average(
    traj: dict[tuple[str, str], list[float]],
    families: list[str],
    contents: list[str],
) -> dict[str, torch.Tensor]:
    """Average each family's trajectory over content, ignoring NaNs.

    Batched over families: one ``(F, C, T)`` tensor and one pass of the masked
    mean, rather than a fresh ``(C, T)`` tensor plus two ``isfinite`` per family.
    Families are grouped by trajectory width first, so a ragged ``traj`` keeps
    working exactly as it did when each family got its own tensor.

    Two behaviours of the per-family loop are load-bearing and preserved here:
    a family with no ``(content, family)`` row at all is **absent** from the
    result (not a row of zeros), and an all-NaN column averages to ``0.0``
    (not NaN) because the finite count is clamped to 1. A missing content row is
    padded with NaN, which the mask sends to a ``0.0`` numerator term and a ``0``
    count term — arithmetically identical to the row being skipped.
    """
    if not families or not contents:
        return {}

    # Group families by trajectory width; the common case is a single group.
    by_width: dict[int, list[str]] = {}
    for fam in families:
        width = -1
        for c in contents:
            arr = traj.get((c, fam))
            if arr is not None:
                width = len(arr)
                break
        if width < 0:
            continue  # no row for this family in any content -> dropped
        by_width.setdefault(width, []).append(fam)

    out: dict[str, torch.Tensor] = {}
    for width, fams in by_width.items():
        blank = [float("nan")] * width
        block = []
        for fam in fams:
            rows = []
            for c in contents:
                arr = traj.get((c, fam))
                # ``is None`` rather than ``.get(key, blank)``: the loop this
                # replaced treated a present-but-None value as absent too.
                rows.append(blank if arr is None else arr)
            block.append(rows)
        t = torch.tensor(block, dtype=torch.float64)  # (F, C, T)
        finite = torch.isfinite(t)
        denom = finite.float().sum(dim=1).clamp(min=1.0)
        avg = torch.where(finite, t, torch.zeros_like(t)).sum(dim=1) / denom
        for j, fam in enumerate(fams):
            # Disjoint row views of a tensor created here; no consumer can reach
            # the parent, and the rows do not alias one another.
            out[fam] = avg[j]
    return out


def _content_family_tensor(
    traj: dict[tuple[str, str], list[float]],
    families: list[str],
    contents: list[str],
) -> torch.Tensor:
    """``(len(contents), len(families), T)`` trajectories, NaN where absent.

    One ``torch.tensor`` call over a NaN-padded nested list, instead of a tensor
    construction plus an indexed assignment per ``(content, family)`` cell
    (#1532) -- the same batching #1531 applied to the sibling helpers.

    Two behaviours of the per-cell version are load-bearing and reproduced here
    deliberately:

    * A missing cell stays **NaN**, never ``0.0``. Every downstream consumer
      masks on ``isfinite``, so a zero pad would silently join the average.
    * A **length-1** row **broadcasts** across all ``T`` timesteps, because the
      indexed assignment this replaces went through torch broadcasting. The
      obvious pad-to-max rewrite writes ``[9.0, nan, nan]`` where this writes
      ``[9.0, 9.0, 9.0]`` -- a changed result that no exception reports, which
      is why it is pinned by a test rather than left to review.

    Any other width still raises, as the indexed assignment did. Batching is not
    licence to start accepting a ragged table (the raise is the existing
    contract, not a bug being fixed in passing).
    """
    if not families or not contents:
        return torch.zeros((0, 0, 0))
    T = len(next(iter(traj.values()))) if traj else 0
    absent = [float("nan")] * T
    grid: list[list[list[float]]] = []
    for c in contents:
        row: list[list[float]] = []
        for fam in families:
            arr = traj.get((c, fam))
            if arr is None:
                row.append(absent)
            elif len(arr) == T:
                row.append(arr)
            elif len(arr) == 1:
                row.append([arr[0]] * T)
            else:
                raise RuntimeError(
                    f"The expanded size of the tensor ({T}) must match the "
                    f"existing size ({len(arr)}) at non-singleton dimension 0. "
                    f"Ragged trajectory at content {c!r}, family {fam!r}."
                )
        grid.append(row)
    return torch.tensor(grid, dtype=torch.float64)


def _norm_nonfinite(v: float) -> float:
    """Map a non-finite sub-score to the correct *end* of ``[0, 1]``.

    ``+inf`` is the BEST case (e.g. ``compute_scvr`` returns ``+inf`` for a metric
    with zero content-variance — all signal, no anatomy nuisance), so it maps to
    the top (``1.0``); ``-inf`` and ``NaN`` (a crashed/degenerate sub-score) map to
    the bottom (``0.0``). The previous code sent EVERY non-finite value to ``0.0``,
    inverting the ideal-discriminability case to the worst rank (critique H4/B2).
    """
    return 1.0 if v == math.inf else 0.0


def _minmax_normalise(values: list[float]) -> list[float]:
    if not values:
        return values
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [_norm_nonfinite(v) for v in values]
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-12:
        return [0.5 if math.isfinite(v) else _norm_nonfinite(v) for v in values]
    return [((v - lo) / (hi - lo)) if math.isfinite(v) else _norm_nonfinite(v) for v in values]


def _rank_normalise(values: list[float]) -> list[float]:
    """Outlier-robust midrank-fraction normaliser in ``[0, 1]`` (critique §18/L10).

    Min-max is fragile because the range is a sufficient statistic for any single
    outlier — one extreme metric collapses every other gap. The midrank fraction
    ``(R_i - 1)/(K - 1)`` is invariant under arbitrary monotone reparameterisation
    *and* arbitrary outliers (the outlier takes the extreme rank but does not move
    anyone else's). Non-finite values sort to the bottom rank.
    """
    n = len(values)
    if n == 0:
        return values
    if n == 1:
        return [0.5]
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [_norm_nonfinite(v) for v in values]
    # +inf ranks ABOVE every finite value (best case), -inf/NaN BELOW (critique H4).
    floor = min(finite) - 1.0
    ceil = max(finite) + 1.0
    x = torch.tensor(
        [v if math.isfinite(v) else (ceil if v == math.inf else floor) for v in values],
        dtype=torch.float64,
    )
    mid = _midranks(x)  # 0-based midranks in [0, n-1]
    return [float(r / (n - 1)) for r in mid.tolist()]


def _normalise(values: list[float], mode: str) -> list[float]:
    """Dispatch to the configured sub-score normaliser. Raises on unknown mode."""
    if mode == "minmax":
        return _minmax_normalise(values)
    if mode == "rank":
        return _rank_normalise(values)
    raise ValueError(f"unknown normalizer {mode!r}; expected 'minmax' or 'rank'")


def _masked_normalise(values: list[float], mode: str, measuring: list[bool]) -> list[float]:
    """Normalise only the measuring entries; non-measuring entries get 0.

    This is the ranker-side of the leaderboard fix (#412). Both ``minmax`` and
    ``rank`` hand a tie block of non-measuring metrics (ADR ≡ 0 — all-NaN / dead /
    inverted) a *positive* normalised value (0.5 for a constant block; a positive
    midrank for a tie block), so a metric that measures nothing would score above
    genuinely live ones in the composite. Excluding them from the pool and forcing
    them to 0 sinks them below every live metric and stops them compressing the
    live scores. Keeps this ranking surface consistent with the main board.
    """
    kept = [v for v, m in zip(values, measuring, strict=True) if m]
    normed_kept = _normalise(kept, mode)
    out: list[float] = []
    it = iter(normed_kept)
    for m in measuring:
        out.append(next(it) if m else 0.0)
    return out


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


class Sim2RankRanker(BaseRanker):
    method_name = "Sim2Rank"

    def __init__(self, config: Sim2RankConfig | None = None) -> None:
        self.config = config or Sim2RankConfig()

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return RankingResult(method=self.method_name, scores={}, ranks=[], diagnostics={})

        # "Clean baseline" is approximated by the lowest-severity bucket per
        # family, since the simulator emits samples on theta ∈ (0, 1). We
        # average the smallest-theta values across content + family for each
        # metric.
        adr_per_metric: dict[str, float] = {}
        scvr_per_metric: dict[str, float] = {}
        cdrs_per_metric: dict[str, float] = {}
        per_family_adr: dict[str, dict[str, float]] = {}
        cdrs_components: dict[str, tuple[float, float]] = {}

        for name in metric_set.names():
            higher = metric_set.is_higher_better(name)

            traj, severities, contents, families = _build_trajectory_table(dataset, name)
            family_avgs = _per_family_average(traj, families, contents)

            # ADR: per-family, then average.
            adr_breakdown: dict[str, float] = {}
            for fam, t in family_avgs.items():
                adr_breakdown[fam] = compute_adr(
                    t,
                    epsilon=self.config.epsilon,
                    higher_is_better=higher,
                    nonsat=self.config.nonsat,
                )
            per_family_adr[name] = adr_breakdown
            adr_per_metric[name] = (
                float(sum(adr_breakdown.values()) / max(len(adr_breakdown), 1))
                if adr_breakdown
                else 0.0
            )

            # SCVR: build (content, family, severity) tensor.
            cf_tensor = _content_family_tensor(traj, families, contents)
            scvr_per_metric[name] = compute_scvr(cf_tensor) if cf_tensor.numel() else 0.0

            # CDRS: clean-baseline = lowest-severity bucket; family trajectories
            # are the per-family averages.
            clean_scores: list[float] = []
            if severities:
                for c in contents:
                    for fam in families:
                        arr = traj.get((c, fam))
                        if arr is None:
                            continue
                        v = arr[0]
                        if math.isfinite(v):
                            clean_scores.append(v)
            cdrs, c_m, m_m = compute_cdrs(
                clean_scores, list(family_avgs.values()), higher_is_better=higher
            )
            cdrs_per_metric[name] = cdrs
            cdrs_components[name] = (c_m, m_m)

        # Composite: weighted sum of normalised sub-scores. Non-measuring metrics
        # (ADR ≡ 0 — all-NaN / dead / inverted, since compute_adr clamps every
        # non-live cause to 0) are held out of the normalisation pool so they
        # cannot occupy a positive midrank above a live metric (#412, applied here
        # to keep this ranking surface consistent with the main leaderboard).
        names = metric_set.names()
        norm = self.config.normalizer
        measuring = [adr_per_metric[n] > 1e-9 for n in names]
        adr_n = _masked_normalise([adr_per_metric[n] for n in names], norm, measuring)
        scvr_n = _masked_normalise([scvr_per_metric[n] for n in names], norm, measuring)
        cdrs_n = _masked_normalise([cdrs_per_metric[n] for n in names], norm, measuring)

        w_adr, w_scvr, w_cdrs = self.config.weights
        wsum = w_adr + w_scvr + w_cdrs
        scores: dict[str, float] = {}
        for i, name in enumerate(names):
            scores[name] = (w_adr * adr_n[i] + w_scvr * scvr_n[i] + w_cdrs * cdrs_n[i]) / max(
                wsum, 1e-12
            )

        # A ranker whose scores collapse into ties has failed, and the leaderboard
        # would otherwise print a top-5 resolved by pool order (#236, #243).
        assert_not_degenerate(list(scores.values()), name=self.method_name)

        ranks = self._ranks_from_scores(scores)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=ranks,
            diagnostics={
                "adr": adr_per_metric,
                "scvr": scvr_per_metric,
                "cdrs": cdrs_per_metric,
                "per_family_adr": per_family_adr,
                "cdrs_components": {
                    name: {"c_m": c_m, "m_m": m_m} for name, (c_m, m_m) in cdrs_components.items()
                },
                "weights": {"adr": w_adr, "scvr": w_scvr, "cdrs": w_cdrs},
            },
        )
