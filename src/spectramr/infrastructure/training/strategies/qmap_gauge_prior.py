"""Gauge prior for tissue maps fitted with a per-image gain (``model_kwargs.gain_ratio_prior``).

From weighted contrasts with an unknown gain per image, (rho, T1, T2) leave exactly two
directions free: the T2/T1 and FLAIR/T1 gain ratios (``gi_scbd_rank_gate``). The series-gain
gate (#2614) measured those ratios per subject on M4Raw and found stable medians but a
per-subject spread, so fixing them was rejected. This module turns the measurement into a
soft prior instead: each image's implied log ratio pays its squared z-score against the
measured median and spread. The gauge is closed by data, with the uncertainty the data has.

Expected block, values from the gate's train split::

    gain_ratio_prior:
      weight: 1.0
      T2: {median: 4.2, std_log: 0.16}
      FLAIR: {median: 0.51, std_log: 0.29}
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import torch

_RATIO_SERIES = ("T2", "FLAIR")  # each relative to T1, series 0


class GainRatioPrior(NamedTuple):
    weight: float
    log_median: torch.Tensor  # (2,) log of T2/T1 and FLAIR/T1
    std_log: torch.Tensor  # (2,)


def parse_gain_ratio_prior(block: Mapping[str, Any] | None) -> GainRatioPrior | None:
    """Validate ``model_kwargs.gain_ratio_prior``; ``None`` when the arm declares none."""
    if block is None:
        return None
    unknown = set(block) - {"weight", *_RATIO_SERIES}
    if unknown:
        raise ValueError(f"gain_ratio_prior: unknown key(s) {sorted(unknown)}")
    missing = [s for s in _RATIO_SERIES if s not in block]
    if missing:
        raise ValueError(f"gain_ratio_prior must give {list(_RATIO_SERIES)}; missing {missing}")
    weight = float(block.get("weight", 1.0))
    if weight <= 0:
        raise ValueError(f"gain_ratio_prior.weight={weight}; must be > 0 (omit the block for none)")
    med, std = [], []
    for s in _RATIO_SERIES:
        entry = dict(block[s])
        m, sd = float(entry["median"]), float(entry["std_log"])
        if m <= 0:
            raise ValueError(f"gain_ratio_prior.{s}.median={m}; a gain ratio must be > 0")
        if sd <= 0:
            raise ValueError(f"gain_ratio_prior.{s}.std_log={sd}; must be > 0")
        med.append(torch.tensor(m).log())
        std.append(sd)
    return GainRatioPrior(weight, torch.stack(med), torch.tensor(std))


def projected_gains(
    renders: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """``(B, 3)`` least-squares gain of each series per image, differentiable in the renders."""
    out = []
    for r, y, m in zip(renders, targets, masks, strict=True):
        w = m.to(r.dtype)
        num = (r * y * w).flatten(1).sum(1)
        out.append(num / (r * r * w).flatten(1).sum(1).clamp_min(1e-12))
    return torch.stack(out, dim=1)


def prior_predicted_gains(
    fitted: torch.Tensor, available: torch.Tensor, prior: GainRatioPrior
) -> torch.Tensor:
    """``(B, 3)`` gains predicted from the AVAILABLE series' fits and the median ratios.

    Each available series gives an estimate of the T1 gain (its fit divided by its median
    ratio); their geometric mean, times a series' median ratio, predicts that series. A
    held-out series' own fit is never read, so its prediction cannot see its target.
    """
    log_mu = torch.cat([torch.zeros(1), prior.log_median]).to(fitted)  # T1 ratio is 1
    w = available.to(fitted)
    if (w.sum(1) < 1).any():
        raise ValueError("every sample needs at least one available series to predict from")
    log_t1 = ((fitted.clamp_min(1e-12).log() - log_mu) * w).sum(1, keepdim=True) / w.sum(
        1, keepdim=True
    )
    return (log_t1 + log_mu).exp()


def gain_ratio_penalty(
    renders: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    prior: GainRatioPrior,
) -> torch.Tensor:
    """``weight * mean_images sum_ratios ((log ratio - log median) / std_log)^2``."""
    g = projected_gains(renders, targets, masks).clamp_min(1e-12)
    log_ratio = torch.log(g[:, 1:]) - torch.log(g[:, :1])
    z = (log_ratio - prior.log_median.to(g)) / prior.std_log.to(g)
    return prior.weight * z.pow(2).sum(1).mean()
