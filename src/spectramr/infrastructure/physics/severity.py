"""Declared severity ranges for the digital-twin degradations (the SSOT).

Every degradation is driven by a dimensionless severity ``theta in [0, 1]``. What
that ``theta`` *means physically* — 40 dB SNR down to 2 dB, acceleration R = 1 to
8, a 30-pixel peak displacement — was until now trapped inside each degrader as a
literal (``sigma = 4.0 * theta + 1e-3``), sometimes with the reasoning in a code
comment and nowhere else.

That makes a simple question unanswerable: *"what noise range did this sweep
actually cover?"* You cannot report a severity axis you have not written down, and
a sweep whose physical range lives only in a function body cannot be put in a
table, a paper, or a caption.

This module declares it once. :class:`SeveritySpec` hangs off every
:class:`~spectramr.infrastructure.physics.degradations.DegradationSpec`,
and is the single source for both the sweep's theta grid and the reported
physical range.

Two things this deliberately does **not** paper over:

* **Direction is not always "up".** ``complex_gaussian`` ramps SNR from 40 dB
  (clean) down to 2 dB (worst). Its *physical minimum* is the **most degraded**
  end. Reporting a bare "min / max" column would invert the reader's
  interpretation, so the fields are named ``at_theta_min`` / ``at_theta_max`` —
  by severity, never by magnitude. :attr:`SeveritySpec.direction` states it.

* **Some degradations move more than one knob.** ``rigid_motion`` ramps
  translation *and* rotation together; ``iq_ghost`` ramps a DC offset *and* an I/Q
  imbalance. The severity axis is still one-dimensional (that is fine — it is a
  severity ramp, not an ablation), but the co-varying parameters are declared
  rather than hidden behind whichever one happened to get reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "PhysicalParam",
    "SeverityDirection",
    "SeveritySpec",
    "severity_grid",
    "shape_severity_ratio",
]


def shape_severity_ratio(
    ratio: float,
    schedule: str = "linear",
    power: float = 2.0,
) -> float:
    """Shape a normalised diffusion progress ``ratio`` into a severity ratio.

    The counterpart of ``KSpaceAccelerator._normalized_progress`` for degradation
    severity: undersampling already ramps R over timesteps under a declared
    schedule, and this is the same ramp for the ``DEGRADATION_REGISTRY`` axes.
    It shares the schedule vocabulary (``AccelerationSchedule``) rather than
    minting a parallel one -- two enums with the same members is how two
    spellings of one concept begin.

    ``linear`` returns ``ratio`` UNCHANGED, by early return rather than by
    computing ``ratio ** 1.0``. It is the default, every existing digital-twin
    arm routes through it, and a last-bit difference here would silently move a
    corrupted volume with nothing to attribute it to.

    ``step`` is binary at the midpoint. Undersampling's ``step`` indexes an
    explicit ``acceleration_range`` LIST and falls back to base/max when none was
    declared; severity has no list, only the ``(vmin, vmax)`` pair, so only that
    fallback has a meaning here -- 0 below the midpoint, 1 above.

    Args:
        ratio: normalised progress in [0, 1] (``t / (T - 1)``, or the twin's
            ``corruption_factor``). Values outside are clamped: a caller passing
            t > T should not silently get a severity above the declared maximum.
        schedule: an ``AccelerationSchedule`` value.
        power: exponent for ``power_law`` / ``polynomial``.

    Returns:
        The shaped ratio in [0, 1], to be remapped onto the axis's declared
        ``(vmin, vmax)`` by the caller.
    """
    r = min(1.0, max(0.0, float(ratio)))
    name = str(getattr(schedule, "value", schedule))
    if name == "linear":
        return r
    if name in ("power_law", "polynomial"):
        # Both are ratio**power. Undersampling's own ramp used to handle
        # 'power_law' and let 'polynomial' fall through to linear, so the two
        # ramps disagreed about what 'polynomial' meant; #789 settled it in
        # favour of this reading and KSpaceAccelerator._normalized_progress now
        # computes the same curve.
        if power <= 0.0:
            raise ValueError(f"schedule power must be > 0, got {power}")
        return r ** float(power)
    if name == "exponential":
        # Same 5.0 rate constant as KSpaceAccelerator, so the two ramps are
        # comparable when an arm schedules both.
        return (math.exp(r * 5.0) - 1.0) / (math.exp(5.0) - 1.0)
    if name == "step":
        return 0.0 if r < 0.5 else 1.0
    raise ValueError(
        f"unknown severity schedule {name!r}. Valid: "
        "linear, polynomial, power_law, exponential, step."
    )


class SeverityDirection(StrEnum):
    """Which way the primary physical parameter moves as severity rises."""

    INCREASING = "increasing"  # e.g. noise sigma 0 -> 0.3
    DECREASING = "decreasing"  # e.g. SNR 40 dB -> 2 dB


@dataclass(frozen=True, slots=True)
class PhysicalParam:
    """One physical quantity a degradation ramps, with its value at each end.

    Named by *severity* endpoint, not by magnitude: ``at_theta_max`` is the value
    at the most-degraded end even when that is numerically the smaller number.
    """

    name: str
    units: str
    at_theta_min: float
    at_theta_max: float

    def value_at(self, theta: float) -> float:
        """Affine interpolation. Every current degrader is affine in ``theta``."""
        return self.at_theta_min + (self.at_theta_max - self.at_theta_min) * float(theta)

    @property
    def direction(self) -> SeverityDirection:
        return (
            SeverityDirection.INCREASING
            if self.at_theta_max >= self.at_theta_min
            else SeverityDirection.DECREASING
        )


@dataclass(frozen=True, slots=True)
class SeveritySpec:
    """The declared severity axis of one degradation.

    Args:
        params: the physical quantities this degradation ramps. **The first is the
            primary** — the one reported as *the* severity axis. Any others
            co-vary with it and are declared so the coupling is visible.
        n_steps: default number of severity steps swept.
        theta_min / theta_max: the dimensionless severity range.
        notes: anything a reader needs that the numbers do not say.
    """

    params: tuple[PhysicalParam, ...]
    n_steps: int = 8
    theta_min: float = 0.0
    theta_max: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.params:
            raise ValueError(
                "a SeveritySpec must declare at least one physical parameter -- "
                "an undeclared severity axis is exactly what this class exists to "
                "prevent."
            )
        if self.n_steps < 2:
            raise ValueError(f"n_steps must be >= 2 to form a grid, got {self.n_steps}")
        if not self.theta_max > self.theta_min:
            raise ValueError(
                f"theta_max ({self.theta_max}) must exceed theta_min ({self.theta_min})"
            )

    @property
    def primary(self) -> PhysicalParam:
        return self.params[0]

    @property
    def co_varying(self) -> tuple[PhysicalParam, ...]:
        """Parameters that move together with the primary (empty for most)."""
        return self.params[1:]

    @property
    def direction(self) -> SeverityDirection:
        """Which way the primary parameter moves as severity rises."""
        return self.primary.direction

    @property
    def step(self) -> float:
        """Spacing of the default theta grid."""
        return (self.theta_max - self.theta_min) / (self.n_steps - 1)

    def theta_grid(self, n_steps: int | None = None) -> tuple[float, ...]:
        """The swept severity values, inclusive of both endpoints."""
        n = self.n_steps if n_steps is None else int(n_steps)
        if n < 2:
            raise ValueError(f"n_steps must be >= 2 to form a grid, got {n}")
        span = self.theta_max - self.theta_min
        return tuple(self.theta_min + span * i / (n - 1) for i in range(n))

    def physical_value(self, theta: float, param: str | None = None) -> float:
        """Physical value of ``param`` (default: the primary) at severity ``theta``."""
        if param is None:
            return self.primary.value_at(theta)
        for p in self.params:
            if p.name == param:
                return p.value_at(theta)
        declared = ", ".join(p.name for p in self.params)
        raise KeyError(
            f"{param!r} is not a declared parameter of this severity spec (declared: {declared})"
        )


def severity_grid(
    n_steps: int,
    theta_min: float | None = None,
    theta_max: float = 1.0,
) -> list[float]:
    """The canonical severity (theta) grid for a degradation sweep -- the SSOT.

    Returns ``n_steps`` ascending theta values in ``[theta_min, theta_max]`` as a
    Python list, built with ``torch.linspace(...).tolist()`` so the floats are
    BIT-IDENTICAL to what the sweeps produced before this was centralised (fp
    parity is what keeps the leaderboard invariant under the refactor).

    ``theta_min=None`` -> ``1 / n_steps``: the "uniform grid on ``(0, 1]`` with the
    identity step at 0 dropped" convention. Making it the ONE anchor kills a latent
    bug -- two builders, ``linspace(1/T, 1, T)`` (``DegradationSweep``) and
    ``linspace(theta_min, 1, T)`` (``MetaDegradationSweep``), agreed only by the
    coincidence ``theta_min = 0.05 = 1/20`` and diverged silently at any ``T != 20``.

    theta is the canonical severity symbol; ``corruption_factor`` (the DigitalTwin
    kwarg) and ``alpha(t)`` (figure labels) are aliases for this same quantity.

    This is the ORDERED ramp the monotone estimators use (ADR, isotonic, Jacobian,
    worst-axis). The core simulator's quasirandom ``halton_grid`` is a DIFFERENT
    construction for the space-filling FSPD ranker and is intentionally not routed
    here.
    """
    import torch

    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2 to form a grid, got {n_steps}")
    lo = (1.0 / n_steps) if theta_min is None else float(theta_min)
    return torch.linspace(lo, float(theta_max), n_steps).tolist()
