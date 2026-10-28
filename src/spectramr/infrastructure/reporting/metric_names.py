r"""Parse logged metric column names into ``(base, acceleration, aggregate)``.

Training logs name a metric ``<split>_<base>[_<R>x | _mean | _accel_gap]`` --
``val_psnr``, ``val_psnr_8x``, ``val_psnr_mean``. Every report figure and table
needs the *base* metric (to look up its direction, label and the canonical
headline set) and the acceleration factor (for the sweep), so the parse lives
here once and the aggregator stamps its result onto every row.

The split prefix is not re-derived from the name: the aggregator already knows
which file a row came from, and a ``train_`` column inside the validation log is
still a validation row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Prefixes the training/validation/eval logs put on a metric column.
SPLIT_PREFIXES: tuple[str, ...] = ("val_", "train_", "test_")

#: ``_<R>x`` with an integer or decimal R, e.g. ``_2x``, ``_8x``, ``_2.5x``.
_ACCEL_SUFFIX = re.compile(r"_(?P<r>\d+(?:\.\d+)?)x$")

#: Cascade-level aggregates written beside the per-R columns.
AGGREGATE_SUFFIXES: tuple[str, ...] = ("_mean", "_accel_gap")


@dataclass(frozen=True)
class MetricName:
    """One parsed metric column name."""

    raw: str
    base: str
    acceleration: float | None = None
    aggregate: str | None = None

    @property
    def is_headline(self) -> bool:
        """True for the one value a table should show per base metric.

        A plain column (``val_psnr``) and the cascade mean (``val_psnr_mean``)
        both qualify; a per-R column belongs to the acceleration sweep and an
        ``_accel_gap`` is a derived contrast, not a quality score.
        """
        return self.acceleration is None and self.aggregate in (None, "_mean")


def parse_metric_name(name: str) -> MetricName:
    """Split a logged column name into its base metric and qualifiers."""
    stem = str(name)
    for prefix in SPLIT_PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    match = _ACCEL_SUFFIX.search(stem)
    if match is not None:
        return MetricName(name, stem[: match.start()], acceleration=float(match.group("r")))
    for suffix in AGGREGATE_SUFFIXES:
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return MetricName(name, stem[: -len(suffix)], aggregate=suffix)
    return MetricName(name, stem)


def annotate_frame(df):  # type: ignore[no-untyped-def]
    """Add ``base_metric`` / ``acceleration`` / ``headline`` columns to a long frame.

    Parsed once per distinct name rather than per row; the frame is returned
    unchanged when it has no ``metric`` column.
    """
    if df is None or df.empty or "metric" not in df.columns:
        return df
    parsed = {m: parse_metric_name(m) for m in df["metric"].astype(str).unique()}
    names = df["metric"].astype(str)
    df = df.copy()
    df["base_metric"] = names.map(lambda m: parsed[m].base)
    df["acceleration"] = names.map(lambda m: parsed[m].acceleration).astype(float)
    df["headline"] = names.map(lambda m: parsed[m].is_headline)
    return df


__all__ = [
    "AGGREGATE_SUFFIXES",
    "SPLIT_PREFIXES",
    "MetricName",
    "annotate_frame",
    "parse_metric_name",
]
