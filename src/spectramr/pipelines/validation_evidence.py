"""Whether a run's validation produced evidence, and the verdict when it did not.

A run is reported successful only when training completed *and* every enabled
validation produced at least one finite number (#181). Two shapes defeat a
per-event check and so need a run-level one:

* every event returns metrics that are all non-finite, or carry only the
  failure sentinel and the batch-count stamps, which are finite on every row;
* every event returns an empty dict because no batch had a sample to average.

This module is the one owner of that judgement. The training loop records each
event into a :class:`ValidationTally`; the loop's final return and the
``--validate-only`` path both ask :func:`validation_failure_reason` for the
verdict.

Zero events is deliberately *not* judged here. ``validation_can_fire`` already
raises when a declared validation can never fire within the budget, and the
remaining zero-event cases (sanity mode's compressed budget, a wall-clock yield,
a resume past the last event) are legitimate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: What a strategy returns when validation could not produce a prediction at all.
#: A finite float, so it must never count as evidence: ``validation_metrics.csv``
#: holding a single ``validation_error, 1.0`` row was exactly the #181 evidence.
VALIDATION_FAILURE_SENTINEL = "validation_error"

#: Keys present on every validation row that describe the pass, not the model.
NON_EVIDENCE_KEYS: frozenset[str] = frozenset(
    {
        VALIDATION_FAILURE_SENTINEL,
        "val_batches_failed",
        "val_batches_attempted",
        "val_samples_skipped_max",
        "val_monitor_samples_skipped",
        "val_monitor_coverage",
    }
)


def is_validation_evidence(metrics: Mapping[str, Any] | None) -> bool:
    """True when ``metrics`` holds at least one finite number about the model."""
    if not metrics:
        return False
    for key, value in metrics.items():
        if key in NON_EVIDENCE_KEYS or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return True
    return False


@dataclass
class ValidationTally:
    """Counts of validation events and of those that produced evidence."""

    events: int = 0
    events_with_evidence: int = 0

    def record(self, metrics: Mapping[str, Any] | None) -> None:
        self.events += 1
        if is_validation_evidence(metrics):
            self.events_with_evidence += 1

    def as_result_fields(self) -> dict[str, int]:
        return {
            "validation_events": self.events,
            "validation_events_with_finite_metric": self.events_with_evidence,
        }


def validation_failure_reason(tally: ValidationTally) -> str | None:
    """The reason a run must be reported failed, or ``None`` if validation is sound."""
    if tally.events == 0 or tally.events_with_evidence > 0:
        return None
    return (
        f"Validation ran {tally.events} time(s) and produced no finite metric: "
        "every value was non-finite, empty, or only the failure sentinel / batch "
        "counts. A run whose validation measured nothing has not been validated, "
        "so it is not reported as a success (#181)."
    )
