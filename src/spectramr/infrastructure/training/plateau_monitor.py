"""Reusable plateau-detection primitive.

A :class:`PlateauMonitor` tracks a single monitored scalar (a validation
metric or a training loss) and reports when it stops improving — a *plateau*.
It is the shared core behind two consumers:

* :class:`~spectramr.infrastructure.services.early_stopping.EarlyStoppingService`,
  which stops training on a plateau, and
* :class:`~spectramr.infrastructure.training.loss_schedule_controller.LossScheduleController`,
  which *acts* on a plateau (reweighting / enabling / disabling a loss term).

Before this module the plateau logic was reimplemented three times
(``EarlyStoppingService``, ``TrainingState.check_early_stopping``, and inline in
``PipelineMultiStageStrategy.on_epoch_end``) with divergent comparators. This is
the single source of truth.

Design notes
------------
* **Strict comparator.** Improvement is ``x < best - min_delta`` (MIN) or
  ``x > best + min_delta`` (MAX). A non-strict ``<=`` / ``>=`` treats a
  bit-identical flat metric as a fresh improvement every check, so the wait
  counter never grows and a plateau never fires — exactly the signature of
  identity-collapse / DC-blob runs. Strictness is the guard (pitfall #10).
* **Two usage levels.** :meth:`observe` is the low-level "record + count wait"
  call (``EarlyStoppingService`` layers its own stop decision on top via
  :meth:`is_plateaued`). :meth:`fire_if_plateaued` is the high-level
  "fire once per plateau, then suppress for ``cooldown`` checks" call the
  loss-schedule controller uses for repeatable triggering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from spectramr.config.schemas.enums import MetricMode


@dataclass
class PlateauMonitor:
    """Detect when a monitored scalar stops improving.

    Attributes:
        mode: ``MIN`` (lower is better, e.g. loss) or ``MAX`` (higher is
            better, e.g. PSNR/SSIM).
        patience: Number of consecutive non-improving observations after which
            :meth:`is_plateaued` returns ``True``.
        min_delta: Minimum strict margin an observation must beat the best by to
            count as an improvement.
        cooldown: Checks to suppress after :meth:`fire_if_plateaued` fires, so a
            single sustained plateau does not re-fire on every subsequent check.
        best_value: Best (lowest in MIN / highest in MAX) value seen; ±inf until
            the first observation.
        best_iteration: Iteration at which :attr:`best_value` was set.
        wait_count: Consecutive non-improving observations since the last
            improvement (or last fire).
        cooldown_remaining: Checks left in the current post-fire suppression
            window.
    """

    mode: MetricMode
    patience: int
    min_delta: float = 0.0
    cooldown: int = 0

    best_value: float = field(init=False, default=0.0)
    best_iteration: int = 0
    wait_count: int = 0
    cooldown_remaining: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Return to the initial state (best = ±inf, counters cleared)."""
        self.best_value = float("inf") if self.mode == MetricMode.MIN else float("-inf")
        self.best_iteration = 0
        self.wait_count = 0
        self.cooldown_remaining = 0

    def is_improvement(self, value: float) -> bool:
        """Return ``True`` iff ``value`` strictly beats the best by ``min_delta``."""
        if self.mode == MetricMode.MIN:
            return value < (self.best_value - self.min_delta)
        return value > (self.best_value + self.min_delta)

    def observe(self, value: float, iteration: int) -> bool:
        """Record ``value``; return ``True`` iff it improved on the best so far.

        On improvement the best value/iteration are updated and ``wait_count``
        resets to 0; otherwise ``wait_count`` increments. This is the low-level
        primitive — it does not consult ``patience`` or ``cooldown`` (callers
        layer their own decision via :meth:`is_plateaued`).

        A non-finite ``value`` (NaN/Inf — a real early-training / DC-blob
        signature here) is *ignored entirely*: it neither counts as an
        improvement nor accrues toward a plateau, so a transient NaN cannot
        spuriously fire a plateau action or an early stop.
        """
        if not math.isfinite(value):
            return False
        if self.is_improvement(value):
            self.best_value = value
            self.best_iteration = iteration
            self.wait_count = 0
            return True
        self.wait_count += 1
        return False

    def is_plateaued(self) -> bool:
        """Return ``True`` when ``wait_count`` has reached ``patience``."""
        return self.wait_count >= self.patience

    def fire_if_plateaued(self, value: float, iteration: int) -> bool:
        """Observe ``value`` and fire once when a plateau crosses ``patience``.

        Returns ``True`` exactly on the observation that pushes ``wait_count``
        to ``patience`` while the monitor is not in its post-fire cooldown.
        On firing, ``wait_count`` resets and a ``cooldown``-length suppression
        window opens, so a sustained plateau fires at most once per
        ``cooldown + patience`` window rather than every check.

        During cooldown the value is still observed, so an improvement updates
        the best, but a non-improving check does not count toward the next
        plateau (as in torch's ``ReduceLROnPlateau``). Counting them made a
        sustained plateau re-fire every ``cooldown + 1`` checks, the first of
        them possibly on a NaN observation.
        """
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.observe(value, iteration)
            self.wait_count = 0
            return False

        self.observe(value, iteration)
        if math.isfinite(value) and self.is_plateaued():
            self.wait_count = 0
            self.cooldown_remaining = self.cooldown
            return True
        return False


__all__ = ["PlateauMonitor"]
