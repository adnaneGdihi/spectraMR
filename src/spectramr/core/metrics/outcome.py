"""Outcome vocabulary for a metric evaluation: scored / not-applicable / crashed.

A sweep that scores ~164 metrics across ~28 degradations must not conflate three
outcomes:

* **OK** — the metric ran and produced a number.
* **NOT_APPLICABLE** — the metric is *undefined* on this input (an ROI smaller
  than its receptive field, a required mask that was not supplied, a k-space
  measurement that an image-space crop cannot restrict). This is a legitimate,
  expected outcome and carries a machine-readable reason.
* **CRASHED** — the metric raised. This is a **defect**, not a result.

Collapsing all three to a bare ``float("nan")`` — as the sim2rank engine did —
makes a genuine crash indistinguishable from "this metric does not apply here",
so a broken metric reads as an honest N/A and is never fixed. That is pitfall #9
(silent fallback) wearing a numeric disguise: the leaderboard still renders, the
run still exits zero, and the metric simply never scores.

The rule this module enforces: **a number is only ever returned when a metric
actually computed it.** Everything else is NaN *plus a reason*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "MetricNotApplicableError",
    "MetricOutcome",
    "MetricOutcomeReport",
    "MetricStatus",
    "NotApplicableReason",
]


class MetricStatus(StrEnum):
    """How a single metric evaluation ended."""

    OK = "ok"
    NOT_APPLICABLE = "not_applicable"
    CRASHED = "crashed"


class NotApplicableReason(StrEnum):
    """Why a metric is undefined on a given input.

    Every member is a *declared* reason: the metric was never invoked, and the
    caller can explain the NaN without reading a traceback. New reasons are added
    here rather than as free-text so the exclusion table stays groupable.
    """

    IMAGE_TOO_SMALL_FOR_METRIC = "image_too_small_for_metric"
    ROI_TOO_SMALL_SIDE = "roi_min_side_below_metric_floor"
    ROI_TOO_SMALL_AREA = "roi_area_below_metric_floor"
    ROI_NOT_RECTANGULAR = "roi_not_rectangular_for_crop_tier_metric"
    MISSING_REQUIRED_ROI_MASK = "missing_required_roi_mask"
    MEASUREMENT_NOT_ROI_RESTRICTABLE = "physics_measurement_not_roi_restrictable"
    MISSING_MEASUREMENT_CONTEXT = "missing_measurement_context"
    DATA_RANGE_UNRESOLVED = "data_range_unresolved_for_range_sensitive_metric"
    DECLARED_KWARG_UNSUPPORTED = "declared_kwarg_unsupported_by_metric"
    VELOCITY_DIRECTIONS_BELOW_SPATIAL_RANK = "velocity_directions_below_spatial_rank"
    DEGENERATE_VELOCITY_FIELD = "velocity_field_has_no_gradient"


class MetricNotApplicableError(Exception):
    """Raised by a metric that is undefined on the input it was handed.

    This is the *only* exception type a caller may treat as a benign N/A. Any
    other exception escaping a metric is a crash and must be reported as one.
    Raising this instead of returning a fabricated number is the contract: a
    metric never invents a value it could not measure.
    """

    def __init__(self, metric: str, reason: NotApplicableReason, detail: str) -> None:
        self.metric = metric
        self.reason = reason
        self.detail = detail
        super().__init__(f"{metric} is not applicable ({reason}): {detail}")


class MetricContractError(RuntimeError):
    """A metric's registry declaration contradicts its implementation.

    Distinct from :class:`MetricNotApplicableError` (the metric is fine, the
    input is not) and from a computation failure (the metric ran and blew up).
    This is a *wiring* defect: the two halves of the same claim disagree, and
    the only safe response is to stop rather than to guess which half is right.
    It is deliberately not caught by the computer's broad handler -- degrading
    it to NaN would hide the one condition it exists to surface (pitfall #9).
    """


@dataclass(frozen=True, slots=True)
class MetricOutcome:
    """The result of evaluating one metric on one (prediction, target) pair."""

    metric: str
    value: float
    status: MetricStatus
    reason: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status is MetricStatus.OK:
            if math.isnan(self.value):
                raise ValueError(
                    f"{self.metric}: an OK outcome cannot carry NaN. A metric that "
                    "returns NaN has not measured anything -- classify it as CRASHED."
                )
        else:
            if not math.isnan(self.value):
                raise ValueError(
                    f"{self.metric}: a {self.status} outcome must carry NaN, not "
                    f"{self.value!r}. A non-OK outcome has no measurement to report."
                )
            if not self.reason:
                raise ValueError(f"{self.metric}: a {self.status} outcome needs a reason.")

    @classmethod
    def ok(cls, metric: str, value: float) -> MetricOutcome:
        """A genuine measurement. NaN is rejected -- route it to :meth:`crashed`."""
        return cls(metric=metric, value=float(value), status=MetricStatus.OK)

    @classmethod
    def not_applicable(cls, metric: str, reason: NotApplicableReason, detail: str) -> MetricOutcome:
        return cls(
            metric=metric,
            value=math.nan,
            status=MetricStatus.NOT_APPLICABLE,
            reason=str(reason),
            detail=detail,
        )

    @classmethod
    def crashed(cls, metric: str, exc: BaseException) -> MetricOutcome:
        return cls(
            metric=metric,
            value=math.nan,
            status=MetricStatus.CRASHED,
            reason=type(exc).__name__,
            detail=str(exc),
        )

    @classmethod
    def from_value(cls, metric: str, value: float) -> MetricOutcome:
        """Classify a raw metric return: a NaN return is a defect, not a result."""
        v = float(value)
        if math.isnan(v):
            return cls(
                metric=metric,
                value=math.nan,
                status=MetricStatus.CRASHED,
                reason="NaNReturn",
                detail=(
                    f"{metric} returned NaN without raising. A metric that cannot "
                    "measure its input must raise MetricNotApplicableError with a "
                    "reason, never return NaN."
                ),
            )
        return cls.ok(metric, v)


@dataclass(frozen=True, slots=True)
class MetricOutcomeReport:
    """Every outcome from one evaluation pass, with the crashes kept separable."""

    outcomes: tuple[MetricOutcome, ...]

    @property
    def values(self) -> dict[str, float]:
        """``metric -> value`` (NaN for anything that did not score)."""
        return {o.metric: o.value for o in self.outcomes}

    @property
    def scored(self) -> tuple[MetricOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status is MetricStatus.OK)

    @property
    def not_applicable(self) -> tuple[MetricOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status is MetricStatus.NOT_APPLICABLE)

    @property
    def crashed(self) -> tuple[MetricOutcome, ...]:
        return tuple(o for o in self.outcomes if o.status is MetricStatus.CRASHED)

    def raise_if_crashed(self) -> None:
        """Fail loudly when any metric crashed.

        A crash is a defect in the metric or its wiring. A sweep that silently
        NaNs it out reports success while quietly dropping the metric from every
        leaderboard it should have appeared in.
        """
        if not self.crashed:
            return
        lines = [f"  {o.metric}: {o.reason}: {o.detail}" for o in self.crashed]
        raise RuntimeError(
            f"{len(self.crashed)} metric(s) crashed during evaluation:\n" + "\n".join(lines)
        )

    def to_rows(self) -> list[dict[str, object]]:
        """Flatten to table rows (feeds the run's exclusion / `dropped` report)."""
        return [
            {
                "metric": o.metric,
                "status": str(o.status),
                "value": o.value,
                "reason": o.reason or "",
                "detail": o.detail or "",
            }
            for o in self.outcomes
        ]
