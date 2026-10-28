r"""Chainwise trajectory monitoring for the cold-diffusion reverse loop.

The papers reduce "did the trajectory fabricate?" to per-step functionals of
the evolving :math:`\hat x_0` estimate:

- :math:`\kappa_s` — the relative data-consistency residual at step ``s``
  (the same math as the terminal ``ndcr`` / ``kappa_T`` metric, evaluated
  mid-trajectory on the observed support);
- the **excursion** :math:`V_\varrho` — how far the trajectory strays beyond
  the admissible radius :math:`\varrho` around the (validation-only) target;
- the **first violation index** :math:`s^*` — the earliest step at which a
  monitored bound is broken, localising *where* a reconstruction went wrong.

:class:`TrajectoryMonitor` takes ``(step_idx, pred_x0, current_mask)`` once per
reverse step. The predict-side loop that fired it was deleted when predict moved
onto ``generator.sample()`` (#2067), and that path has no per-step hook yet, so
nothing in ``train``/``predict`` drives the monitor today. It observes the
NORMALIZED k-space domain; every recorded quantity is relative, so the
normalization scale cancels.

The admissible radius is never hardcoded: :func:`calibrate_admissible_radius`
derives it from clean validation trajectories as a conformal quantile with a
DKW finite-sample band (the papers' provenance requirement).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from spectramr.core.metrics.dkw import dkw_slack

__all__ = [
    "TrajectoryMonitor",
    "calibrate_admissible_radius",
    "hallucination_rate",
    "severity",
]


def _relative_residual(pred: Tensor, ref: Tensor, mask: Tensor | None, eps: float) -> float:
    """``‖(pred − ref)·M‖ / (‖ref·M‖ + eps)`` over all dims (scalar)."""
    if mask is not None:
        m = mask.real if torch.is_complex(mask) else mask
        pred = pred * m
        ref = ref * m
    diff = pred - ref
    if torch.is_complex(diff):
        num = torch.sqrt((diff.real**2 + diff.imag**2).sum())
        den = torch.sqrt((ref.real**2 + ref.imag**2).sum())
    else:
        num = torch.sqrt((diff**2).sum())
        den = torch.sqrt((ref**2).sum())
    return float((num / (den + eps)).detach())


class TrajectoryMonitor:
    """Per-trajectory accumulator, usable directly as ``step_callback``.

    Args:
        y_kspace: the acquired (normalized) k-space measurement — the same
            tensor the reverse loop starts from.
        measurement_mask: observed-support mask; ``None`` means fully sampled
            (κ_s is then a full-support residual).
        kappa_threshold: optional bound on κ_s; steps above it count as
            violations. ``None`` disables κ-based violation detection.
        admissible_radius: optional ϱ from :func:`calibrate_admissible_radius`;
            requires ``target``. Steps with relative distance-to-target > ϱ
            count as violations.
        target: fully-sampled reference k-space (validation only). Enables the
            excursion / max-violation-ratio diagnostics; leave ``None`` at
            deployment, where κ_s is the only computable trace.
    """

    def __init__(
        self,
        y_kspace: Tensor,
        measurement_mask: Tensor | None = None,
        *,
        kappa_threshold: float | None = None,
        admissible_radius: float | None = None,
        target: Tensor | None = None,
        eps: float = 1e-8,
    ) -> None:
        if admissible_radius is not None and target is None:
            raise ValueError(
                "admissible_radius needs a target to measure distance against; "
                "pass target (validation) or drop the radius (deployment)."
            )
        if admissible_radius is not None and admissible_radius <= 0:
            raise ValueError(f"admissible_radius must be > 0, got {admissible_radius!r}.")
        self.y_kspace = y_kspace
        self.measurement_mask = measurement_mask
        self.kappa_threshold = kappa_threshold
        self.admissible_radius = admissible_radius
        self.target = target
        self.eps = eps
        self.step_indices: list[int] = []
        self.kappa_per_step: list[float] = []
        self.distance_per_step: list[float] = []

    def __call__(self, step_idx: int, pred_x0: Tensor, current_mask: Tensor | None = None) -> None:
        """The ``step_callback`` protocol; ``current_mask`` is accepted but the
        residual always uses the fixed measurement support."""
        self.step_indices.append(int(step_idx))
        self.kappa_per_step.append(
            _relative_residual(pred_x0, self.y_kspace, self.measurement_mask, self.eps)
        )
        if self.target is not None:
            self.distance_per_step.append(_relative_residual(pred_x0, self.target, None, self.eps))

    def _violations(self) -> list[bool]:
        flags = [False] * len(self.kappa_per_step)
        if self.kappa_threshold is not None:
            for i, kappa in enumerate(self.kappa_per_step):
                flags[i] = flags[i] or kappa > self.kappa_threshold
        if self.admissible_radius is not None:
            for i, dist in enumerate(self.distance_per_step):
                flags[i] = flags[i] or dist > self.admissible_radius
        return flags

    def first_violation_index(self) -> int | None:
        """Position (0-based, in firing order) of the first violating step —
        the papers' :math:`s^*`. ``None`` when nothing is monitored or bound."""
        for i, violated in enumerate(self._violations()):
            if violated:
                return i
        return None

    def excursion(self) -> float | None:
        """:math:`V_\\varrho`: the worst overshoot beyond ϱ (0.0 for a clean
        trajectory). ``None`` when no radius/target was configured."""
        if self.admissible_radius is None or not self.distance_per_step:
            return None
        return max(0.0, max(self.distance_per_step) - self.admissible_radius)

    def summary(self) -> dict:
        """The per-trajectory record the cohort statistics aggregate."""
        out: dict = {
            "num_steps": len(self.kappa_per_step),
            "step_indices": list(self.step_indices),
            "kappa_per_step": list(self.kappa_per_step),
            "trajectory_kappa_final": self.kappa_per_step[-1] if self.kappa_per_step else None,
            "trajectory_kappa_max": max(self.kappa_per_step) if self.kappa_per_step else None,
            "first_violation_index": self.first_violation_index(),
        }
        if self.target is not None:
            out["distance_per_step"] = list(self.distance_per_step)
            if self.admissible_radius is not None:
                out["trajectory_excursion"] = self.excursion()
                out["trajectory_max_violation_ratio"] = (
                    max(self.distance_per_step) / self.admissible_radius
                    if self.distance_per_step
                    else None
                )
        return out


def hallucination_rate(excursions: Sequence[float]) -> float:
    """Fraction of trajectories that left the admissible tube (excursion > 0)."""
    if not excursions:
        raise ValueError("hallucination_rate needs at least one excursion value.")
    return sum(1 for e in excursions if e > 0) / len(excursions)


def severity(excursions: Sequence[float], alpha: float = 0.05) -> float:
    """Tail severity: mean of the worst ``ceil(alpha·n)`` excursions (CVaR_α).

    The rate says *how often* trajectories escape; this says *how badly* the
    worst α-tail escapes — the pair the papers report together.
    """
    if not excursions:
        raise ValueError("severity needs at least one excursion value.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha!r}.")
    worst = sorted(excursions, reverse=True)[: math.ceil(alpha * len(excursions))]
    return sum(worst) / len(worst)


def calibrate_admissible_radius(
    clean_distances: Sequence[float], alpha: float = 0.05
) -> tuple[float, float]:
    """Admissible radius ϱ from clean validation trajectories, with provenance.

    ϱ is the conformal ``(1−α)`` quantile of the clean per-trajectory maximum
    distances (index ``⌈(n+1)(1−α)⌉``, clipped to the sample maximum), so a
    clean trajectory violates it with probability ≤ α. The returned DKW slack
    ``sqrt(ln(2/α)/(2n))`` bounds how far the empirical quantile law can sit
    from the truth at confidence 1−α — report ϱ WITH its band, never alone.

    Returns:
        ``(rho, dkw_band)``.
    """
    n = len(clean_distances)
    if n == 0:
        raise ValueError("calibrate_admissible_radius needs at least one clean distance.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}.")
    ordered = sorted(float(d) for d in clean_distances)
    rank = min(math.ceil((n + 1) * (1.0 - alpha)), n)
    return ordered[rank - 1], dkw_slack(n, alpha)
