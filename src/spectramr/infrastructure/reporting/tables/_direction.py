r"""Which way a metric improves, for ranking and for the header arrow.

The direction SSOT (:mod:`spectramr.core.metrics.metric_directions`) is asked first.
Two further signals are accepted because logged names carry more than the registry
declares: the longest registry token inside the name (``cjv_t1`` -> ``cjv``), and a
word that only ever means "smaller is better" (``loss``, ``error``). A name with
none of these has **no direction**, and the table shows it without ranking it:
guessing "higher is better" bolds the worst arm of any lower-is-better metric the
registry does not know.
"""

from __future__ import annotations

from spectramr.core.metrics.metric_directions import METRIC_HIGHER_IS_BETTER, resolve_direction

LOWER_WORDS: tuple[str, ...] = (
    "loss",
    "error",
    "rmse",
    "nmse",
    "mae",
    "fid",
    "kid",
    "hd95",
    "ece",
    "nll",
    "lpips",
    "hfen",
    "gmsd",
    "asd",
)


def direction(metric: str, overrides: dict[str, bool] | None = None) -> bool | None:
    """``True`` if higher is better, ``False`` if lower is, ``None`` if nothing says."""
    if overrides and metric in overrides:
        return overrides[metric]
    name = metric.lower()
    # Rows arrive with the split prefix stripped while some keys are declared under
    # their logged name, so both spellings are asked.
    for key in (name, f"val_{name}"):
        declared = resolve_direction(key)
        if declared is not None:
            return declared
    if name in METRIC_HIGHER_IS_BETTER:
        return METRIC_HIGHER_IS_BETTER[name]
    matches = [(k, v) for k, v in METRIC_HIGHER_IS_BETTER.items() if k in name]
    if matches:
        return max(matches, key=lambda kv: len(kv[0]))[1]
    return False if any(word in name for word in LOWER_WORDS) else None


__all__ = ["LOWER_WORDS", "direction"]
