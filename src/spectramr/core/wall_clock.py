"""Wall-clock budget SSOT — yielding before the allocation ends.

A production arm is allowed 120 h per job and needs far more, so the run has to
survive being cut in half. The half that was missing is not *resume* — the
checkpoint carries model, optimizer, scheduler, scaler, EMA, RNG and counter
state already — it is **stopping in time to write one**. A job killed at the
wall loses everything since the last periodic save, which at the corpus-typical
``save_interval: 5000`` is hours of GPU.

So the sbatch reports when the allocation ends, and the training loop stops
itself a margin ahead of that, saves, and exits cleanly enough to be requeued.

**Why a deadline and not a signal.** ``--signal=B:USR1@N`` delivers to the batch
step, which on the array path has ``exec``'d into the dispatcher, which launches
``train`` — or ``torchrun``, which does not forward SIGUSR1 and whose default
disposition kills the launcher and orphans the workers. A deadline is inherited
by every descendant through the environment and needs no forwarding at all.

**The margin is not decoration.** The deadline is inspected every
:data:`MAX_CHECK_INTERVAL` steps at most, so the margin must cover that many
steps *plus* the final checkpoint write; too small and the job dies mid-save
holding a truncated file.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from spectramr.core.env_names import (
    SPECTRAMR_WALL_CLOCK_DEADLINE,
    SPECTRAMR_WALL_CLOCK_MARGIN_S,
    SPECTRAMR_WALL_CLOCK_MARKER,
)

#: Seconds reserved for the final save and teardown when none is declared.
DEFAULT_MARGIN_S = 900.0

#: Filename used when the launcher does not name one.
DEFAULT_MARKER_NAME = "WALL_CLOCK_YIELD"

#: Upper bound on how many steps pass between deadline checks.
#:
#: It cannot be the logging cadence, which is what this used to ride: 65 arms in
#: `experiments/inprogress/` declare `logging.intervals.log: 5000`, so at ~1 s a
#: step the deadline would be inspected every ~83 min against a 900 s margin --
#: the job is killed before the check ever fires, on exactly the long-running
#: arms this exists for. Reading the clock is a vDSO call, not a device
#: synchronisation, so non-negotiable 9 never forbade its own cadence; the
#: per-check cost is the DDP broadcast, which is microseconds at this rate.
MAX_CHECK_INTERVAL = 100

#: Multiplier applied to the previous validation's measured duration when
#: deciding whether the next one fits in the remaining budget.
#:
#: The check above cannot interrupt a validation pass — it runs between
#: iterations and validation is *inside* one — so a deadline landing mid-pass
#: kills the job with no checkpoint, no marker and no requeue. That is not a
#: corner: `experiment_11_attention_none` records ONE validation event at 6.4 h
#: and validation at 41 % of the run's wall clock (its own config note). The
#: factor covers run-to-run variance in a duration measured only once.
VALIDATION_SAFETY_FACTOR = 1.25


class WallClockBudgetError(RuntimeError):
    """A declared wall-clock budget is unusable.

    Raised rather than ignored: an unparseable deadline that degraded to "no
    budget" would produce a run that looks normal and then dies at the wall with
    nothing saved — the exact failure the budget exists to prevent (pitfall 9).
    """


@dataclass(frozen=True)
class WallClockBudget:
    """When this process must stop training and save.

    Attributes:
        deadline: Unix epoch second at which the allocation ends.
        margin_s: Seconds reserved ahead of it for the final checkpoint.
    """

    deadline: float
    margin_s: float

    @property
    def yield_at(self) -> float:
        """Epoch second at which the loop should stop and save."""
        return self.deadline - self.margin_s

    def seconds_remaining(self, now: float | None = None) -> float:
        """Seconds left before :attr:`yield_at`. Negative once it has passed."""
        return self.yield_at - (time.time() if now is None else now)

    def expired(self, now: float | None = None) -> bool:
        """Whether the loop should yield now."""
        return self.seconds_remaining(now) <= 0.0


def resolve_check_interval(log_interval: int) -> int:
    """Steps between deadline checks: the logging cadence, capped.

    Capped rather than followed, so an arm with a coarse `log` interval is
    still checked often enough to act on the margin. A non-positive declaration
    falls back to the cap rather than to 1: it cannot disable the check, and it
    cannot turn a nonsense value into a per-step syscall either.
    """
    if not log_interval or log_interval <= 0:
        return MAX_CHECK_INTERVAL
    return max(1, min(int(log_interval), MAX_CHECK_INTERVAL))


def resolve_yield_marker_path(run_dir: str | Path) -> Path:
    """Where this run announces that it stopped at the wall.

    The launcher names it, because the launcher is what reads it back; a second
    derivation on the writing side is how the two silently disagree and the
    chain stops without a word (non-negotiable 17).
    """
    declared = os.environ.get(SPECTRAMR_WALL_CLOCK_MARKER)
    if declared and declared.strip():
        return Path(declared.strip())
    return Path(run_dir) / DEFAULT_MARKER_NAME


def _read_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise WallClockBudgetError(
            f"{name}={raw!r} is not a number. It is an absolute unix epoch "
            "second (deadline) or a count of seconds (margin); export it from "
            "the sbatch via scripts/common/wall_clock.sh, never by hand."
        ) from exc


def resolve_wall_clock_budget(now: float | None = None) -> WallClockBudget | None:
    """Read the declared budget from the environment, or ``None`` if unset.

    Unset is the normal case — an interactive run, a smoke test, a job with no
    wall-clock limit — and means "train to ``max_iterations``".

    Raises:
        WallClockBudgetError: If a budget is declared but unusable: a
            non-numeric value, a non-positive margin, or an allocation whose
            remaining time is already inside the margin. That last one is the
            one worth failing on — yielding at iteration 1 and requeueing would
            spin a chain forever without ever training a step.
    """
    deadline = _read_float(SPECTRAMR_WALL_CLOCK_DEADLINE)
    if deadline is None:
        return None

    margin = _read_float(SPECTRAMR_WALL_CLOCK_MARGIN_S)
    margin = DEFAULT_MARGIN_S if margin is None else margin
    if margin <= 0:
        raise WallClockBudgetError(
            f"{SPECTRAMR_WALL_CLOCK_MARGIN_S}={margin} must be positive: it is "
            "the time reserved to write the final checkpoint, and zero means "
            "the job is killed mid-save."
        )

    budget = WallClockBudget(deadline=deadline, margin_s=margin)
    if budget.expired(now):
        remaining = deadline - (time.time() if now is None else now)
        raise WallClockBudgetError(
            f"Allocation ends in {remaining:.0f}s, inside the {margin:.0f}s save "
            "margin, so this job would yield before training a single step and "
            "requeue forever. Request more wall time, or lower "
            f"{SPECTRAMR_WALL_CLOCK_MARGIN_S}."
        )
    return budget


__all__ = [
    "DEFAULT_MARGIN_S",
    "DEFAULT_MARKER_NAME",
    "MAX_CHECK_INTERVAL",
    "VALIDATION_SAFETY_FACTOR",
    "WallClockBudget",
    "WallClockBudgetError",
    "resolve_check_interval",
    "resolve_wall_clock_budget",
    "resolve_yield_marker_path",
]
