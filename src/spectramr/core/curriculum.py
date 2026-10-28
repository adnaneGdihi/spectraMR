"""Declared vs effective timestep curriculum (#1296).

``training.curriculum_start_timestep`` and ``training.curriculum_ramp_rate``
ramp the sampled-timestep ceiling over training, so early steps see mild
degradation and the model reaches the hard high-acceleration regime gradually.

Two conditions silently disable that ramp, and neither says anything:

* declaring only one of the pair (both are ``| None``, and the strategy treats
  a half-declared curriculum as no curriculum);
* ``training.max_iterations <= SHORT_RUN_BYPASS_ITERATIONS`` — a deliberate
  short-run bypass, because a 5k-iteration ramp never reaches meaningful
  acceleration and would train the arm on near-identity inputs.

The bypass is correct and stays. What was missing is that an arm which
*declares* a curriculum and then does not get one looks, from every artifact
the run produces, exactly like an arm that got one. This module resolves the
question once from config alone so the strategy can say so, and so the
startup knob line can record what actually applies.

Config-only and pure: it reads nothing from the model, allocates nothing, and
is called once per run — never from the training loop (non-negotiable 9).

Lives in ``core/`` because both ``infrastructure/training/strategies`` and
``infrastructure/logging`` consume it, and a strategy importing from a logging
module (or the reverse) would be the sideways import that made these two
disagree in the first place.
"""

from __future__ import annotations

import math
from typing import Any, Final, NamedTuple

#: Runs at or below this many iterations skip every curriculum/ramp cap.
#:
#: The literal 5000 was written out at three separate sites in
#: ``strategies/diffusion.py`` -- the timestep-curriculum cap, the validation
#: ``r_max`` ramp and its training-side twin -- with one of them commenting
#: that it "mirrors" another. Three copies of a threshold that must move
#: together is the divergence-by-coincidence shape (#697).
SHORT_RUN_BYPASS_ITERATIONS: Final[int] = 5000

#: Used when ``training.max_iterations`` is unset, matching the strategy's
#: long-standing ``or 500_000`` default at the call sites.
DEFAULT_MAX_ITERATIONS: Final[int] = 500_000

#: A top rung that first becomes drawable after this fraction of the budget is
#: reported as late. Measured on the 69 kspace_filling arms at start 1, rate
#: 0.0005, 70000 iterations: t=28 opens at 0.8 of the budget and receives 1.1 %
#: of all draws against 8.3 % for t=1, although ``balanced_high_t`` weights high
#: t up (#2066).
LATE_TOP_RUNG_FRACTION: Final[float] = 0.5


class CurriculumState(NamedTuple):
    """What the arm asked for, and what it will actually get.

    Attributes:
        declared: both curriculum knobs are set.
        effective: the ramp will actually cap the sampled timestep range.
        suppressed_by: why a declared curriculum is not effective; ``None``
            when nothing was suppressed (including when nothing was declared).
        start_timestep: ``training.curriculum_start_timestep`` as read.
        ramp_rate: ``training.curriculum_ramp_rate`` as read.
        max_iterations: the horizon the bypass was judged against, after the
            ``DEFAULT_MAX_ITERATIONS`` fallback.
        top_opens_at: first iteration at which the top timestep ``T - 1`` is
            drawable, or ``None`` when the curriculum is not effective or the
            horizon is unknown.
    """

    declared: bool
    effective: bool
    suppressed_by: str | None
    start_timestep: int | None
    ramp_rate: float | None
    max_iterations: int
    top_opens_at: int | None = None

    @property
    def truncated(self) -> bool:
        """The top rung never opens inside the budget."""
        return self.top_opens_at is not None and self.top_opens_at > self.max_iterations

    @property
    def opens_late(self) -> bool:
        """The top rung opens after :data:`LATE_TOP_RUNG_FRACTION` of the budget."""
        return (
            self.top_opens_at is not None
            and self.top_opens_at > LATE_TOP_RUNG_FRACTION * self.max_iterations
        )

    def ladder_warning(self) -> str | None:
        """Why the top rung is under-trained, or ``None`` when it is not."""
        if self.top_opens_at is None or not (self.truncated or self.opens_late):
            return None
        where = (
            "never inside the budget"
            if self.truncated
            else f"at {self.top_opens_at / self.max_iterations:.0%} of the budget"
        )
        return (
            f"the curriculum opens the top timestep at iteration {self.top_opens_at} "
            f"of {self.max_iterations} ({where}), so the hardest rung is trained "
            "least; validation at that rung scores timesteps with few or no draws."
        )

    def describe(self) -> str:
        """One field for the startup knob line.

        ``off`` and ``declared-but-off`` are deliberately different strings:
        the first is a choice, the second is a knob that was set and did not
        take, which is the case a reader needs to be able to spot.
        """
        if self.effective:
            opens = "" if self.top_opens_at is None else f",top_opens_at={self.top_opens_at}"
            return f"curriculum=on(t0={self.start_timestep},rate={self.ramp_rate}{opens})"
        if self.suppressed_by is not None:
            # Keyed on `suppressed_by`, not on `declared`: a half-declared
            # curriculum is not "declared" (it cannot run) but reporting it as
            # a plain `off` would hide the very thing worth seeing -- someone
            # set a knob and got nothing.
            return f"curriculum=declared-but-off({self.suppressed_by})"
        return "curriculum=off"


def resolve_curriculum_state(config: Any) -> CurriculumState:
    """Answer "will the declared curriculum actually run?" from config alone.

    Args:
        config: the resolved ``TrainingSettings`` SSOT.

    Returns:
        A :class:`CurriculumState`. This function does not log or raise; the
        caller decides whether a suppressed curriculum deserves a warning.
    """
    training = getattr(config, "training", None)
    start_t = getattr(training, "curriculum_start_timestep", None)
    rate = getattr(training, "curriculum_ramp_rate", None)
    max_iters = getattr(training, "max_iterations", None) or DEFAULT_MAX_ITERATIONS

    # Both halves or neither: `start_t + iteration * rate` needs both, and the
    # schema documents None as "the strategy defaults to no curriculum".
    declared = start_t is not None and rate is not None

    suppressed_by: str | None = None
    if declared and max_iters <= SHORT_RUN_BYPASS_ITERATIONS:
        suppressed_by = (
            f"max_iterations={max_iters} <= {SHORT_RUN_BYPASS_ITERATIONS}, the short-run bypass"
        )
    elif not declared and (start_t is not None or rate is not None):
        # Half-declared. Not "declared", but emphatically not silence either:
        # someone set one knob and got nothing.
        suppressed_by = (
            "only one of curriculum_start_timestep / curriculum_ramp_rate is set; both are required"
        )

    effective = declared and suppressed_by is None
    return CurriculumState(
        declared=declared,
        effective=effective,
        suppressed_by=suppressed_by,
        start_timestep=start_t,
        ramp_rate=rate,
        max_iterations=max_iters,
        top_opens_at=(
            _top_opens_at(config, int(start_t), float(rate))
            if effective and start_t is not None and rate is not None
            else None
        ),
    )


def _top_opens_at(config: Any, start_t: int, rate: float) -> int | None:
    """First iteration with ``int(start_t + i * rate) >= T``, the cap
    ``DiffusionTrainingStrategy.sample_timesteps`` applies to ``arange(t_low, cap)``."""
    diffusion = getattr(getattr(config, "training", None), "diffusion", None)
    horizon = getattr(diffusion, "timesteps", None)
    if not isinstance(horizon, int) or rate <= 0:
        return None
    if start_t >= horizon:
        return 0
    return math.ceil((horizon - start_t) / rate)
