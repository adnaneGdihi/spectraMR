"""Deterministic file-level train/val index splitting (data SSOT).

One honest split helper shared by every dataset instantiator and manifest
loader so the single-file edge case is handled identically everywhere instead
of drifting into three different wrong behaviours (silent leak, empty train,
warn-and-leak). File-level (never slice-level) splitting avoids the subtler
leak where slices of the same volume land in both train and val.

Policy:

- ``validation_split <= 0`` -> train-only: every item is training, validation
  is empty. This is the explicit escape hatch for single-file smoke / overfit
  sanity runs; set ``validation_split: 0`` in the config to opt in.
- exactly one item with ``validation_split > 0`` -> **raise**. A one-file
  corpus cannot form a train/val split that does not overlap: reusing the file
  for both leaks validation into training, and holding it out empties the
  training set. The caller must supply >= 2 files or choose the train-only
  escape hatch above.
- otherwise -> deterministic non-overlapping split with the last ``n_val``
  items as validation, clamped so ``1 <= n_val <= n - 1`` (both splits
  non-empty).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from os import PathLike
from typing import Any

__all__ = [
    "FILE_KEYS",
    "SUBJECT_KEYS",
    "record_split_key",
    "split_index",
    "split_index_grouped",
    "split_index_three_way",
    "split_index_three_way_grouped",
    "subject_id_from_name",
]

#: Record keys that identify a *subject* (patient), strictest to loosest.
SUBJECT_KEYS = ("subject_id", "patient_id", "subject")
#: Record keys that identify an exact *file*, strictest to loosest.
FILE_KEYS = ("relative_path", "primary_path", "file_id", "path", "file", "input")


def record_split_key(record: Any) -> str:
    """The identity an index item is split on: its subject, else its file.

    A record (a mapping, or an object with those attributes) is read for a subject
    id, then for a file whose name carries a BIDS ``sub-*`` label, then for the file
    itself; a bare path is its ``sub-*`` label or its own name. The same identity the
    leakage audit compares, so a split grouped on it cannot put one subject on both
    sides. An item with none of these is its own group, which splits exactly as a
    by-item split does.
    """
    if isinstance(record, (str, PathLike)):
        name = str(record)
        return subject_id_from_name(name) or name

    def field(key: str) -> Any:
        return record.get(key) if isinstance(record, Mapping) else getattr(record, key, None)

    for k in SUBJECT_KEYS:
        if value := field(k):
            return str(value)
    name = next((str(field(k)) for k in FILE_KEYS if field(k)), None)
    if name is None:
        return f"record-{id(record)}"
    return subject_id_from_name(name) or name


#: The BIDS subject label (``sub-<alnum>``) wherever it appears in a path or
#: file name. Anchored so ``resub-01`` or ``sub-01x`` (a different label) do
#: not match the wrong thing; the label itself is taken whole.
_SUBJECT_TOKEN = re.compile(r"(?<![A-Za-z0-9])(sub-[A-Za-z0-9]+)")


#: ``data.split.type`` values that partition the paired-manifest route from config, not split_hint.
CONFIG_OWNED_SPLIT_TYPES = frozenset({"random", "loso_subject", "subjects"})


def subject_id_from_name(name: str) -> str | None:
    """The subject a file belongs to, read off its BIDS-style name or path.

    ``sub-01/anat/sub-01_T1w.nii.gz`` and ``sub-01_T2w.nii.gz`` both give
    ``sub-01``. Names carrying no ``sub-`` label (M4Raw's ``2022_T101``,
    fastMRI's ``file1000000``) give ``None``: the caller then falls back to the
    file's own identity, which is exactly the pre-2026-09 file-level split.

    Returns:
        The subject label, or ``None`` when the name carries none.
    """
    match = _SUBJECT_TOKEN.search(name)
    return match.group(1) if match else None


def split_index[T](items: Sequence[T], validation_split: float) -> tuple[list[T], list[T]]:
    """Split an ordered index into non-overlapping ``(train, val)`` lists.

    Args:
        items: the ordered index (file records, paths, ...). Copied, never
            mutated.
        validation_split: fraction of ``items`` to hold out for validation.
            ``<= 0`` selects the train-only escape hatch.

    Returns:
        ``(train_items, val_items)`` — disjoint lists whose union is ``items``.

    Raises:
        ValueError: exactly one item with ``validation_split > 0`` (no
            non-overlapping split exists; see module docstring).
    """
    seq = list(items)
    n = len(seq)
    if n == 0:
        return [], []
    if validation_split <= 0.0:
        return seq, []
    if n == 1:
        raise ValueError(
            "Cannot form a non-overlapping train/val split from a single file "
            f"with validation_split={validation_split!r}: reusing it for both "
            "leaks validation into training, and holding it out empties the "
            "training set. Provide >= 2 files, or set validation_split: 0 for "
            "an explicit train-only run."
        )
    n_val = min(max(1, round(n * validation_split)), n - 1)
    return seq[:-n_val], seq[-n_val:]


def split_index_three_way[T](
    items: Sequence[T], validation_split: float, test_split: float
) -> tuple[list[T], list[T], list[T]]:
    """Split an ordered index into disjoint ``(train, val, test)`` lists.

    The **test set is carved out first**, from the tail, and the validation
    fraction is then applied to what remains. Two consequences worth stating,
    because both are easy to get silently wrong:

    * val and test are disjoint by construction -- applying both fractions to the
      whole index independently would overlap them, and an overlap between the
      set you tune on and the set you report is the one leak a held-out split
      exists to prevent;
    * ``validation_split`` keeps meaning "of the training pool", so an arm that
      adds a test set does not silently shrink its validation set as well.

    ``test_split <= 0`` returns an empty test list and delegates entirely to
    :func:`split_index`, so an arm that declares no test set behaves exactly as
    before -- there is no third split unless one was asked for.

    Args:
        items: the ordered index. Copied, never mutated.
        validation_split: fraction of the POST-TEST remainder held out for
            validation.
        test_split: fraction of ``items`` held out for test. ``<= 0`` disables.

    Returns:
        ``(train, val, test)`` — pairwise disjoint, union equal to ``items``.

    Raises:
        ValueError: too few items to form the requested non-overlapping splits.
    """
    seq = list(items)
    n = len(seq)
    if n == 0:
        return [], [], []
    if test_split <= 0.0:
        train, val = split_index(seq, validation_split)
        return train, val, []

    # A test set needs at least one item for each of train/val/test to be
    # non-empty where they were asked for. Refuse rather than silently returning
    # an empty test set the caller would report numbers from.
    minimum = 3 if validation_split > 0 else 2
    if n < minimum:
        raise ValueError(
            f"Cannot form a non-overlapping split of {n} item(s) with "
            f"validation_split={validation_split!r} and test_split={test_split!r}: "
            f"at least {minimum} are needed. Provide more data, or set "
            f"test_split: 0 for a run with no held-out test set."
        )

    n_test = min(max(1, round(n * test_split)), n - minimum + 1)
    remainder, test = seq[:-n_test], seq[-n_test:]
    train, val = split_index(remainder, validation_split)
    return train, val, test


def _group_in_order[T](
    items: Sequence[T], key: Callable[[T], Hashable]
) -> tuple[list[Hashable], dict[Hashable, list[T]]]:
    """Bucket ``items`` by ``key`` keeping first-seen group order and item order."""
    order: list[Hashable] = []
    groups: dict[Hashable, list[T]] = {}
    for item in items:
        k = key(item)
        if k not in groups:
            order.append(k)
            groups[k] = []
        groups[k].append(item)
    return order, groups


def split_index_grouped[T](
    items: Sequence[T], validation_split: float, key: Callable[[T], Hashable]
) -> tuple[list[T], list[T]]:
    """:func:`split_index` over GROUPS, so no group straddles train and val.

    The 2026-09 cohort review found the directory-route splits (BIDS and plain
    NIfTI crawls with no manifest) partitioning a flat *file* list: one
    subject's ``T1w`` could train while its ``T2w`` validated -- the same
    anatomy on both sides, which is the leak a held-out split exists to
    prevent. Grouping by subject before splitting is the fix; a file whose
    name carries no subject label is its own group, so a corpus without labels
    splits exactly as before.

    ``validation_split`` is applied to the number of GROUPS (subjects), which
    is what a fraction of patients means; the returned lists keep the input's
    item order within each group.

    Args:
        items: the ordered index. Copied, never mutated.
        validation_split: fraction of groups held out for validation; ``<= 0``
            selects the train-only escape hatch of :func:`split_index`.
        key: maps an item to its group identity.

    Returns:
        ``(train_items, val_items)`` -- disjoint, union equal to ``items``, and
        every group entirely on one side.
    """
    order, groups = _group_in_order(items, key)
    train_keys, val_keys = split_index(order, validation_split)
    train = [item for k in train_keys for item in groups[k]]
    val = [item for k in val_keys for item in groups[k]]
    return train, val


def split_index_three_way_grouped[T](
    items: Sequence[T],
    validation_split: float,
    test_split: float,
    key: Callable[[T], Hashable],
) -> tuple[list[T], list[T], list[T]]:
    """:func:`split_index_three_way` over GROUPS (see :func:`split_index_grouped`)."""
    order, groups = _group_in_order(items, key)
    train_keys, val_keys, test_keys = split_index_three_way(order, validation_split, test_split)
    return (
        [item for k in train_keys for item in groups[k]],
        [item for k in val_keys for item in groups[k]],
        [item for k in test_keys for item in groups[k]],
    )
