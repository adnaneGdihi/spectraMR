r"""Accumulate per-case validation metrics into ``per_call_metrics.csv``.

Companion to :class:`ReportCaseRecorder`. Where the recorder keeps a *bounded*
best/median/worst pool of image cases, this sink keeps an *unbounded* (memory-
capped) table of every validation-case observation so the QC group IQM
plots get one point per case rather than only the retained extremes.

Fed from the same detached-CPU validation seam as the recorder
(``feed_report_case_recorder``), so it adds no GPU sync. Each row is one
validation-case observation: the batch's representative sample tagged with its
``step`` plus whatever scalar metrics/losses the run computed for that batch.
It records the metrics the run actually produced — no fixed IQM list.

Each row also carries an optional **context** block: the identity of what was
evaluated, as columns rather than as substructure inside ``case_id``. Cascading
validation feeds this seam once per (batch, acceleration rung), so before the
context block a 45-batch x 3-rung sweep wrote 135 rows sharing three
``case_id`` values — the numbers were all there and nothing said which volume,
which contrast or which acceleration produced any one of them. ``case_id``
keeps its spelling (PNG filenames and downstream parsers are built on it); the
context columns are additive.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spectramr.core.metrics.scalar_transfer import fuse_to_host

logger = logging.getLogger(__name__)

#: Identity columns, in the order they are written. Declared rather than
#: discovered so the CSV's leading columns do not reshuffle between runs
#: because one arm happened to publish a key another did not. A context key
#: outside this tuple is still written — it lands after the declared ones, in
#: sorted order — because refusing an unknown identity would make this
#: component the gatekeeper of a vocabulary it does not own.
CONTEXT_COLUMNS: tuple[str, ...] = (
    "acceleration_level",
    "acceleration_realized",
    "timestep",
    "heldout",
    "contrast",
    "file_id",
    "batch_index",
    "batch_size",
)


class PerCallMetricSink:
    """Collect per-case validation metric rows and flush them to CSV.

    Args:
        enabled: When False every method is a no-op (nothing is written).
        max_rows: Memory cap; once exceeded the oldest row is evicted so a
            long run cannot grow the table without bound.
    """

    def __init__(self, *, enabled: bool, max_rows: int = 100_000) -> None:
        self.enabled = bool(enabled)
        self.max_rows = max(int(max_rows), 1)
        self._rows: list[dict] = []
        #: Which columns arrived as context. Tracked as observed rather than
        #: inferred at write time: a context value and a metric value are both
        #: numbers in the frame, so nothing in the DataFrame can tell them apart
        #: afterwards.
        self._context_keys: set[str] = set()
        #: Every row ever observed; unlike ``n_rows`` it does not stop at
        #: ``max_rows``, so a caller can tell whether a call added one.
        self.n_observed = 0

    @property
    def n_rows(self) -> int:
        return len(self._rows)

    def observe(
        self,
        *,
        case_id: str,
        metrics: dict,
        split: str = "val",
        step: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one validation-case row. Only finite scalar metrics are kept.

        Args:
            context: Identity of what was evaluated — acceleration rung,
                timestep, contrast, file id. Written as its own columns ahead of
                the metrics. Unlike ``metrics``, values are stored **as given**:
                a string stays a string and ``heldout`` stays a bool, because
                coercing them to float would render the flag as ``1.0`` and drop
                the contrast entirely.

        Raises:
            ValueError: when a context key collides with a metric key. One would
                overwrite the other and the CSV would show a number under a name
                that means something else — indistinguishable, on read, from a
                correct row (pitfall #9). A collision is a call-site error, so it
                fires deterministically on the first row rather than corrupting
                the table.
        """
        if not self.enabled:
            return
        row: dict[str, object] = {"case_id": str(case_id), "split": str(split)}
        if step is not None:
            row["step"] = int(step)
        metric_names = {
            str(k)
            for k, v in (metrics or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        for k, v in (context or {}).items():
            if v is None:
                # Absent, not zero. A rung whose timestep could not be resolved
                # must read as blank, not as t=0 — which is a real timestep.
                continue
            key = str(k)
            if key in metric_names or key in ("case_id", "split", "step"):
                raise ValueError(
                    f"per-case context key {key!r} collides with an existing "
                    "column; one value would silently overwrite the other. "
                    "Rename the context key at the call site."
                )
            row[key] = v
            self._context_keys.add(key)
        for k, v in (metrics or {}).items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                row[str(k)] = float(v)
        self._rows.append(row)
        self.n_observed += 1
        if len(self._rows) > self.max_rows:
            self._rows.pop(0)

    def write(self, run_dir: str | Path, filename: str = "per_call_metrics.csv") -> Path | None:
        """Write the accumulated rows to ``<run_dir>/<filename>`` and return it.

        Returns None when disabled or nothing was observed. Column order is
        stable: ``case_id, split, step``, then the identity columns present in
        :data:`CONTEXT_COLUMNS` order (any undeclared ones after, sorted), then
        metric columns sorted by name.
        """
        out = Path(run_dir) / filename
        if not self.enabled or not self._rows:
            return None
        import pandas as pd

        frame = pd.DataFrame(self._rows)
        lead = [c for c in ("case_id", "split", "step") if c in frame.columns]
        declared = [c for c in CONTEXT_COLUMNS if c in self._context_keys and c in frame.columns]
        undeclared = sorted(
            c for c in self._context_keys if c in frame.columns and c not in CONTEXT_COLUMNS
        )
        lead = lead + declared + undeclared
        metric_cols = sorted(c for c in frame.columns if c not in lead)
        frame = frame[lead + metric_cols]
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        logger.info("per-case metrics: wrote %d rows to %s", len(frame), out)
        return out


_SKIPPED_DEVICE_KEYS: set[str] = set()


def feed_from_validation_loop(
    sink: PerCallMetricSink | None,
    metrics: Mapping[str, Any],
    *,
    step: int,
    batch_index: int,
    observed_before: int,
) -> bool:
    """Record a validation batch the strategy did not record itself (#2500).

    The sink is attached to every strategy, but only the strategies whose image
    logging calls ``feed_report_case_recorder`` ever fed it, so most paradigms
    (reconstruction among them) wrote no ``per_call_metrics.csv``. The training
    loop calls this after each ``validation_step``: when the strategy already
    observed a row for the batch (``n_observed`` moved past ``observed_before``)
    nothing is added, so no batch is counted twice. Single-element tensors, on
    any device, are fetched together in one transfer (``val_loss`` arrives on the
    device); a multi-element tensor is skipped.
    """
    if sink is None or not sink.enabled or sink.n_observed > observed_before:
        return False
    scalars: dict[str, float] = {}
    tensors: dict[str, torch.Tensor] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.generic)):
            scalars[str(key)] = float(value)
        elif isinstance(value, torch.Tensor) and value.numel() == 1 and not value.is_complex():
            tensors[str(key)] = value
        elif isinstance(value, torch.Tensor) and key not in _SKIPPED_DEVICE_KEYS:
            _SKIPPED_DEVICE_KEYS.add(key)
            logger.warning(
                "per_call_metrics: %r is a multi-element or complex tensor and is not "
                "recorded per call; return a scalar to record it",
                key,
            )
    if tensors:
        scalars.update(zip(tensors, fuse_to_host(list(tensors.values())), strict=True))
    if not scalars:
        return False
    sink.observe(
        case_id=f"val_step{int(step)}",
        metrics=scalars,
        step=int(step),
        context={"batch_index": int(batch_index)},
    )
    return True
