"""One row per validation batch, carrying the samples' identifiers.

``validation_metrics.csv`` holds one aggregate row per validation pass, which is
enough to select a checkpoint and nothing else: a paired test across arms needs
the same slice scored under each arm. Under ``validation.per_batch_rows: true``
every batch of every pass lands as one row of ``validation_batches.csv`` beside
it, with the identifiers the loader put on the batch (``subject_id``,
``contrast``, ``slice_index``, ``repetition_group``, ``file_id``) and the batch's
metrics. A loader that serves whole volumes (``(B, C, H, W, S)``, the m4raw
route) is scored one depth slice at a time through :func:`expand_depth`, so at
``validation.loader.batch_size: 1`` a row is a slice on every route; a batch
without a depth axis passes through unchanged. Each rank writes its own file,
so a sharded validation loses no slice.

Metric values and tensor identifiers stay on their device until the pass ends
and cross to the host in one transfer per column, so the opt-in costs a few syncs
per pass rather than one per batch.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "IDENTIFIER_KEYS",
    "ValidationBatchRows",
    "batch_identifiers",
    "depth_slices",
    "expand_depth",
    "identifier_strings",
    "validation_batches_csv_for",
    "validation_csv_for",
]

#: Batch fields copied into every row, in column order. A batch of several
#: samples joins the values with ``|``; a field the loader did not put on the
#: batch is written empty rather than guessed.
IDENTIFIER_KEYS: tuple[str, ...] = (
    "subject_id",
    "contrast",
    "slice_index",
    "repetition_group",
    "file_id",
)
_FIXED_COLUMNS: tuple[str, ...] = ("iteration", "epoch", "batch_index", "n_samples")


def validation_csv_for(training_csv: str | Path) -> Path:
    """Where the validation CSV lives, given the training CSV.

    The one path rule shared by the writer (``_run_validation``) and every
    reader. The name swap keeps a suffixed training CSV paired with its
    validation file; any other name falls back to the parent directory.
    """
    training_csv = Path(training_csv)
    if "training_metrics" in training_csv.name:
        return training_csv.with_name(
            training_csv.name.replace("training_metrics", "validation_metrics")
        )
    return training_csv.parent / "validation_metrics.csv"


def validation_batches_csv_for(training_csv: str | Path) -> Path:
    """Where the per-batch validation rows live: beside the validation CSV."""
    return validation_csv_for(training_csv).with_name("validation_batches.csv")


def _count(value: Any) -> int:
    """How many samples a collated field describes, without touching the device."""
    if isinstance(value, torch.Tensor):
        return int(value.shape[0]) if value.ndim else 1
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1


def _strings(value: Any) -> str:
    """A host-side field as one ``|``-joined cell (a scalar is one sample)."""
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    return str(value)


def _read(batch: Any, key: str) -> Any:
    getter = getattr(batch, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except (KeyError, TypeError):
            return None
    return getattr(batch, key, None)


def batch_identifiers(batch: Any, n_samples: int) -> dict[str, Any]:
    """The ``IDENTIFIER_KEYS`` of ``batch``, raw: tensors stay where they are.

    Reads through the batch's mapping protocol, which is the only path that
    reaches a ``TrainingBatch``'s metadata and also serves a plain dict. A key
    that is absent, or whose sample count disagrees with ``n_samples``, is
    written empty: a row must never carry another sample's identity.
    """
    cells: dict[str, Any] = {}
    for key in IDENTIFIER_KEYS:
        value = _read(batch, key)
        if value is None or _count(value) != max(1, n_samples):
            cells[key] = ""
        else:
            cells[key] = value.detach() if isinstance(value, torch.Tensor) else value
    return cells


def _to_host(column: list[Any]) -> list[Any]:
    """One transfer for a column of same-shaped tensors; other values pass through."""
    tensors = [v for v in column if isinstance(v, torch.Tensor)]
    if not tensors:
        return column
    if len(tensors) == len(column) and all(t.shape == tensors[0].shape for t in tensors):
        moved = torch.stack([t.reshape(-1) for t in tensors]).cpu().tolist()
        return [row[0] if len(row) == 1 else row for row in moved]
    return [(v.reshape(-1).cpu().tolist() if isinstance(v, torch.Tensor) else v) for v in column]


def identifier_strings(batch: Any, n_samples: int) -> dict[str, str]:
    """``batch_identifiers`` rendered as the cells ``ValidationBatchRows`` writes.

    One host transfer per tensor field, so this is for a script scoring one
    batch at a time, not for the training loop, where the writer defers the
    transfer to ``flush``.
    """
    cells = batch_identifiers(batch, n_samples)
    return {key: _strings(_to_host([value])[0]) for key, value in cells.items()}


def _depth(batch: Any) -> int:
    """The depth of a volume batch, or 0 when the input has no depth axis."""
    value = _read(batch, "input")
    if isinstance(value, torch.Tensor) and value.ndim == 5:
        return int(value.shape[-1])
    return 0


def _take_slice(value: Any, index: int, depth: int) -> Any:
    """``value[..., index]`` for a 5-D tensor of this depth; anything else as is."""
    if isinstance(value, torch.Tensor) and value.ndim == 5 and value.shape[-1] == depth:
        return value[..., index]
    return value


def depth_slices(batch: Any) -> Iterator[Any]:
    """Yield ``batch`` one depth slice at a time, in the depth-1 collate's layout.

    Every 5-D tensor of the batch (input, target, mask, coil maps, ``input_alt``,
    ``sensitivity_complex``) loses its depth axis the way ``squeeze_depth_dim``
    would have dropped a depth of 1, so a strategy sees exactly what a
    slice-level loader hands it; scalars, strings and lower-rank tensors pass
    through, and ``slice_index`` names the depth index unless the loader already
    put a per-sample one on the batch. A batch without a depth axis is yielded
    once, unchanged.
    """
    depth = _depth(batch)
    if depth == 0:
        yield batch
        return
    n_samples = int(_read(batch, "input").shape[0])
    own_index = _read(batch, "slice_index")
    keep_index = isinstance(own_index, torch.Tensor) and _count(own_index) == n_samples
    for index in range(depth):
        slice_index = torch.full((n_samples,), index, dtype=torch.long)
        if isinstance(batch, Mapping):
            cut = {k: _take_slice(v, index, depth) for k, v in batch.items()}
            if not keep_index:
                cut["slice_index"] = slice_index
            yield cut
            continue
        metadata = {k: _take_slice(v, index, depth) for k, v in batch.metadata.items()}
        if not keep_index:
            metadata["slice_index"] = slice_index
        yield type(batch)(
            input=_take_slice(batch.input, index, depth),
            target=_take_slice(batch.target, index, depth),
            mask=_take_slice(batch.mask, index, depth),
            coil_maps=_take_slice(batch.coil_maps, index, depth),
            axes=batch.axes,
            metadata=metadata,
        )


def expand_depth(loader: Iterable[Any]) -> Iterator[Any]:
    """Iterate ``loader`` with every volume batch expanded into its depth slices."""
    for batch in loader:
        yield from depth_slices(batch)


class ValidationBatchRows:
    """Collect one row per batch of a validation pass, then write them once."""

    def __init__(self, path: str | Path, *, iteration: int, epoch: int) -> None:
        self.path = Path(path)
        self.iteration = int(iteration)
        self.epoch = int(epoch)
        self._rows: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(
        self,
        batch_index: int,
        n_samples: int,
        identifiers: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> None:
        """Record a batch; tensors are kept on their device until ``flush``."""
        # Reduced to the mean here, the same reduction flush applies: a volume-sized
        # metric tensor would otherwise be retained for every batch of the pass.
        kept = {
            k: (v.detach().reshape(-1).float().mean() if isinstance(v, torch.Tensor) else v)
            for k, v in metrics.items()
        }
        self._rows.append((int(batch_index), int(n_samples), dict(identifiers), kept))

    def flush(self) -> int:
        """Write the collected rows and return how many landed.

        Every tensor column of the pass crosses to the host in one transfer; a
        metric that is not a number (a string a strategy reports beside its
        scalars) is written as its text.
        """
        if not self._rows:
            return 0
        names: list[str] = []
        for _, _, _, metrics in self._rows:
            names.extend(k for k in metrics if k not in names)
        host: dict[str, list[Any]] = {}
        for name in names:
            column = [metrics.get(name) for _, _, _, metrics in self._rows]
            host[name] = _to_host(
                [v.reshape(-1).float().mean() if isinstance(v, torch.Tensor) else v for v in column]
            )
        ids: dict[str, list[Any]] = {}
        for key in IDENTIFIER_KEYS:
            ids[key] = _to_host([cells.get(key, "") for _, _, cells, _ in self._rows])
        header = [*_FIXED_COLUMNS, *IDENTIFIER_KEYS, *names]
        rows = []
        for r, (batch_index, n_samples, _, _) in enumerate(self._rows):
            row: dict[str, Any] = {
                "iteration": self.iteration,
                "epoch": self.epoch,
                "batch_index": batch_index,
                "n_samples": n_samples,
            }
            row.update({key: _strings(ids[key][r]) for key in IDENTIFIER_KEYS})
            row.update({name: host[name][r] for name in names})
            rows.append(row)
        self._write(header, rows)
        written = len(rows)
        self._rows.clear()
        return written

    def _write(self, header: list[str], rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._existing_header()
        if existing is None:
            with self.path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            return
        new_names = [c for c in header if c not in existing]
        if new_names:
            self._rewrite_with(existing + new_names, rows)
            return
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=existing, restval="", extrasaction="ignore", lineterminator="\n"
            )
            writer.writerows(rows)

    def _existing_header(self) -> list[str] | None:
        if not self.path.is_file() or self.path.stat().st_size == 0:
            return None
        with self.path.open(newline="") as handle:
            return next(csv.reader(handle), None) or None

    def _rewrite_with(self, header: list[str], rows: list[dict[str, Any]]) -> None:
        """A pass that reports a metric earlier passes did not widens the file once."""
        with self.path.open(newline="") as handle:
            old = list(csv.DictReader(handle))
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, restval="", lineterminator="\n")
            writer.writeheader()
            writer.writerows(old)
            writer.writerows(rows)
