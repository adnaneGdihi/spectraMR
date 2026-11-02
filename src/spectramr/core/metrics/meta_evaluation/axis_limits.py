"""Fixed y-limits and the flat-response test shared by every metric-vs-severity figure.

A panel that autoscales makes a metric that moves 0.0003 look as responsive as one that
really reacts, so every panel of one metric takes the same range (min..max of that metric
over every axis or family) and a panel whose metric moves less than its noise is called flat.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

__all__ = ["FLAT_RTOL", "is_flat", "padded_limits", "span_of"]

_MIN_SPAN = 1e-12
#: Fraction of a metric's shared y-range below which a response is flat when it has no floor.
FLAT_RTOL: float = 1e-6


def padded_limits(
    values: Iterable[float] | np.ndarray, pad: float = 0.05
) -> tuple[float, float] | None:
    """``(lo - pad*span, hi + pad*span)`` over the finite ``values``; ``None`` when none are finite.

    A constant metric gets a span of ``max(|v| * 0.05, 1e-12)`` centred on it, so the range is
    never zero-width.
    """
    arr = np.asarray(
        list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64
    )
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        half = max(abs(lo) * 0.05, _MIN_SPAN) / 2.0
        return lo - half, hi + half
    span = hi - lo
    return lo - pad * span, hi + pad * span


def span_of(limits: tuple[float, float]) -> float:
    return float(limits[1] - limits[0])


def is_flat(
    values: np.ndarray, floor: float | None, rtol: float = FLAT_RTOL, ylim_span: float = 0.0
) -> bool:
    """True when the finite range of ``values`` is at most ``max(floor, rtol * |ylim_span|)``.

    ``floor`` is the metric's seed-to-seed noise on the same image; ``ylim_span`` is the shared
    y-range, not the panel's own, so a panel is judged against the scale every sibling uses.
    A ``None`` or non-finite ``floor`` (no seed repeats, or a metric that never scored) means
    no floor: the rtol rule alone decides. A panel with no finite value is flat.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return True
    f = float(floor) if floor is not None and np.isfinite(floor) else 0.0
    return bool(float(arr.max() - arr.min()) <= max(f, rtol * abs(ylim_span)))
