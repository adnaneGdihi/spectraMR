"""Share of training wall time spent waiting on the data loader.

A loader that cannot keep up leaves the accelerator idle without ever failing, and
a cluster reaper then kills the job for it. Timing ``next(data_iter)`` with
``perf_counter`` costs no device sync (non-negotiable 9) and names the cause in the log.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

WINDOW_STEPS = 50
WAIT_FRACTION_LIMIT = 0.5
_MAX_WARNINGS = 5


class LoaderWaitMonitor:
    """Accumulate per-step loader wait and warn when a window is mostly waiting."""

    def __init__(self, window: int = WINDOW_STEPS, limit: float = WAIT_FRACTION_LIMIT) -> None:
        if window <= 0 or not 0.0 < limit < 1.0:
            raise ValueError(f"window must be > 0 and limit in (0, 1), got {window}, {limit}")
        self._window = window
        self._limit = limit
        self._steps = 0
        self._wait = 0.0
        self._wall = 0.0
        self._warnings = 0
        self._total_steps = 0
        self._total_wait = 0.0
        self._total_wall = 0.0

    def record(self, wait_seconds: float, step_seconds: float) -> float | None:
        """Add one step; return the wait fraction when this step closes a window."""
        self._total_steps += 1
        self._total_wait += wait_seconds
        self._total_wall += step_seconds
        self._steps += 1
        self._wait += wait_seconds
        self._wall += step_seconds
        if self._steps < self._window:
            return None
        fraction = self._wait / self._wall if self._wall > 0.0 else 0.0
        wait, wall = self._wait, self._wall
        self._steps, self._wait, self._wall = 0, 0.0, 0.0
        if fraction > self._limit and self._warnings < _MAX_WARNINGS:
            self._warnings += 1
            logger.warning(
                "[LOADER] data loader starved training: %.0f%% of the last %d steps "
                "(%.0f s of %.0f s) was spent waiting for a batch, so the GPU is idle "
                "that long. Raise data.loader.num_workers, enable persistent_workers, "
                "or find the slow stage in the dataset.",
                100.0 * fraction,
                self._window,
                wait,
                wall,
            )
        return fraction

    def as_result_fields(self) -> dict[str, dict[str, float | int]]:
        """Whole-run totals for ``run_summary.json``; empty when no step was timed.

        The windowed warning needs 50 steps to speak, so a short or finished run
        otherwise leaves no record of whether the accelerator was fed.
        """
        if self._total_steps == 0 or self._total_wall <= 0.0:
            return {}
        return {
            "loader_wait": {
                "fraction": round(self._total_wait / self._total_wall, 4),
                "wait_s": round(self._total_wait, 2),
                "wall_s": round(self._total_wall, 2),
                "steps": self._total_steps,
            }
        }
