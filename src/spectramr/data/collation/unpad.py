"""Undo the image collate's padding before validation scores a batch slice by slice.

``ImageCollateStrategy`` stacks samples of unequal extent by zero-padding every
non-channel axis, and records what it padded under ``<key>_padding``. Validation
then folds depth into rows (``pipelines/train.py``, ``_preprocess_validation_tensor``)
or cuts it into slices (``validation_rows.depth_slices``), so a padded depth axis
becomes slices of exact zeros that no scanner measured. fastMRI volumes differ in
slice count, so a validation batch of two knee volumes scored the shorter one's
padding as slices, and the reverse sampler's noise estimate refused them as
samples with no measured bin.

:func:`split_padded_batches` hands validation each padded batch as one batch per
sample, every tensor cut back to that sample's own extent. A batch the collate did
not pad passes through unchanged, so routes whose samples already agree in shape
(M4Raw's 18-slice volumes, depth-1 patches) are untouched.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import torch

#: Suffix under which ``ImageCollateStrategy.collate`` records a key's padding.
PADDING_SUFFIX = "_padding"

PadRecord = list[tuple[int, int]] | None


def remove_padding(tensor: torch.Tensor, pad: PadRecord) -> torch.Tensor:
    """Cut one sample's ``(before, after)`` padding off every axis of ``tensor``."""
    if pad is None:
        return tensor
    if len(pad) != tensor.ndim:
        raise ValueError(
            f"padding record covers {len(pad)} axes but the sample has {tensor.ndim} "
            f"(shape {tuple(tensor.shape)}); the record belongs to another tensor."
        )
    index = tuple(
        slice(before, size - after) for (before, after), size in zip(pad, tensor.shape, strict=True)
    )
    return tensor[index]


def _padding_records(batch: Mapping[str, Any]) -> dict[str, list[PadRecord]]:
    """``{key: per-sample record}`` for every key of ``batch`` the collate padded."""
    records: dict[str, list[PadRecord]] = {}
    for name, value in batch.items():
        key = name[: -len(PADDING_SUFFIX)]
        if name.endswith(PADDING_SUFFIX) and key in batch and isinstance(value, list):
            records[key] = value
    return records


def _take(value: Any, row: int, n_samples: int, pad: list[PadRecord] | None) -> Any:
    """Row ``row`` of a batched field, still batched (leading axis of 1)."""
    if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == n_samples:
        piece = value[row]
        if pad is not None:
            piece = remove_padding(piece, pad[row])
        return piece.unsqueeze(0)
    if isinstance(value, (list, tuple)) and len(value) == n_samples:
        return [value[row]]
    return value


def split_padded_batch(batch: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``batch`` as one batch per sample with the collate's padding removed.

    Returns ``[batch]`` when nothing in it was padded. Fields without a batch
    axis (run-level scalars) are shared by every returned batch, and the
    ``<key>_padding`` records are dropped because nothing is padded any more.

    Raises:
        ValueError: The padding records disagree about the batch size.
    """
    records = _padding_records(batch)
    if not any(pad is not None for rec in records.values() for pad in rec):
        return [dict(batch)]
    sizes = {len(rec) for rec in records.values()}
    if len(sizes) != 1:
        raise ValueError(f"padding records disagree about the batch size: {sorted(sizes)}.")
    n_samples = sizes.pop()
    dropped = {f"{key}{PADDING_SUFFIX}" for key in records}
    return [
        {
            name: _take(value, row, n_samples, records.get(name))
            for name, value in batch.items()
            if name not in dropped
        }
        for row in range(n_samples)
    ]


def split_padded_batches(loader: Iterable[Any]) -> Iterator[Any]:
    """Iterate ``loader`` with every padded mapping batch split by :func:`split_padded_batch`."""
    for batch in loader:
        if isinstance(batch, Mapping):
            yield from split_padded_batch(batch)
        else:
            yield batch


__all__ = [
    "PADDING_SUFFIX",
    "remove_padding",
    "split_padded_batch",
    "split_padded_batches",
]
