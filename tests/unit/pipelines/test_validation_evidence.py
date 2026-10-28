"""A run's validation is evidence only if it produced a finite number (#181).

Each shape below is one the training loop's per-event guards let through and
the run-level verdict must catch: the failure sentinel and the batch-count
stamps are finite floats, so a check that only asks "is any value finite?"
reads a validation that measured nothing as a pass.
"""

from __future__ import annotations

import math

import pytest

from spectramr.pipelines.validation_evidence import (
    VALIDATION_FAILURE_SENTINEL,
    ValidationTally,
    is_validation_evidence,
    validation_failure_reason,
)

# Every pass-bookkeeping stamp `_run_validation` writes. They are finite on an
# all-NaN row, so each one left out of NON_EVIDENCE_KEYS turns the cases below red.
_COUNTS = {
    "val_batches_failed": 0.0,
    "val_batches_attempted": 4.0,
    "val_samples_skipped_max": 8.0,
    "val_monitor_samples_skipped": 8.0,
}


@pytest.mark.parametrize(
    "metrics",
    [
        None,
        {},
        {"val_psnr": math.nan, "val_ssim": math.inf, **_COUNTS},
        {VALIDATION_FAILURE_SENTINEL: 1.0},
        {VALIDATION_FAILURE_SENTINEL: 1.0, **_COUNTS},
        {"val_flag": True},
        {"note": "text"},
    ],
    ids=["none", "empty", "all-non-finite", "sentinel", "sentinel+counts", "bool", "str"],
)
def test_measuring_nothing_is_not_evidence(metrics) -> None:
    assert is_validation_evidence(metrics) is False


def test_one_finite_metric_is_evidence() -> None:
    assert is_validation_evidence({"val_psnr": math.nan, "val_ssim": 0.81, **_COUNTS})


def test_a_tally_with_no_events_is_not_judged() -> None:
    """Zero events is owned by `validation_can_fire`, not by this verdict."""
    assert validation_failure_reason(ValidationTally()) is None


def test_nan_predictions_on_every_event_fail_the_run() -> None:
    tally = ValidationTally()
    for _ in range(3):
        tally.record({"val_psnr": math.nan, "val_ssim": math.nan, **_COUNTS})
    reason = validation_failure_reason(tally)
    assert reason is not None and "no finite metric" in reason
    assert tally.as_result_fields() == {
        "validation_events": 3,
        "validation_events_with_finite_metric": 0,
    }


def test_sentinel_only_events_fail_the_run() -> None:
    """The #181(a) evidence: one `validation_error, 1.0` row and success: true."""
    tally = ValidationTally()
    tally.record({VALIDATION_FAILURE_SENTINEL: 1.0})
    assert validation_failure_reason(tally) is not None


def test_one_event_with_evidence_is_enough() -> None:
    tally = ValidationTally()
    tally.record({})
    tally.record({"val_psnr": 28.0})
    assert validation_failure_reason(tally) is None
    assert tally.events_with_evidence == 1
