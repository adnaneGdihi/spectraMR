"""Config-owned train/val/test partition of a paired manifest (data SSOT, NN 17).

One owner for the paired-manifest split. When the arm declares a partition
(``split.type`` other than ``auto``/``manifest``, or ``enable_test_split``),
the config decides every subject's split and the manifest's ``split_hint`` is
not read. Everything is grouped by subject id, so a subject's repeat scans
(``sub-X/ses-NN/run-N``) always land in the same split. Unpaired (target-less)
records never enter a scored validation or test partition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectramr.data.split_utils import (
    CONFIG_OWNED_SPLIT_TYPES,
    split_index_three_way_grouped,
    subject_id_from_name,
)

__all__ = [
    "PairedPartition",
    "build_partition_stamp",
    "declares_paired_test",
    "declares_test_set_in",
    "holdout_site_on_paired_route",
    "is_config_owned",
    "manifest_sha256",
    "partition_paired_records",
    "resolve_paired_partition",
    "resolve_paired_test_ids",
]


@dataclass(frozen=True)
class PairedPartition:
    """The applied three-way split plus the provenance stamp describing it."""

    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    stamp: dict[str, Any]


def manifest_sha256(path: Path) -> str:
    """SHA-256 of the manifest bytes, so a run names the exact manifest it split."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_config_owned(data: Any) -> bool:
    """Whether the config, not the manifest's ``split_hint``, partitions this arm."""
    split_type = data.split.type
    if split_type in CONFIG_OWNED_SPLIT_TYPES:
        return True
    return split_type == "auto" and bool(data.enable_test_split)


def _read(block: Any, name: str) -> Any:
    # The audit's corpus walk holds only the raw YAML mapping; the loaders hold the typed block.
    return block.get(name) if isinstance(block, Mapping) else getattr(block, name, None)


def declares_paired_test(data: Any) -> bool:
    """Whether the config declares a held-out test partition on the paired route."""
    return bool(_read(data, "enable_test_split")) or bool(
        _read(_read(data, "split"), "test_subjects")
    )


def declares_test_set_in(data: Any) -> bool:
    """Whether a ``data`` block (typed or raw mapping) names a held-out test set.

    A test manifest, or a config-owned test partition of a paired manifest. The one
    predicate the loaders and the audit witness both read.
    """
    source = _read(data, "source")
    if _read(source, "test_index_path"):
        return True
    return bool(_read(source, "paired_manifest_path")) and declares_paired_test(data)


def holdout_site_on_paired_route(data: Any) -> bool:
    """Whether a ``data`` block (typed or raw) sets ``split.holdout_site`` on a paired manifest.

    The paired route raises on it at load under every ``split.type``; the audit reads this
    predicate so it refuses the same arms before a job is queued.
    """
    source = _read(data, "source")
    return bool(_read(source, "paired_manifest_path")) and bool(
        _read(_read(data, "split"), "holdout_site")
    )


def _subject_of(record: dict[str, Any]) -> str:
    sid = record.get("subject_id") or subject_id_from_name(
        str(record.get("primary_path") or record.get("file_id") or "")
    )
    if not sid:
        raise ValueError(
            f"paired record {record.get('file_id') or record.get('primary_path')!r} carries no "
            "subject id: a config-owned partition cannot keep its repeat scans together. "
            "Name the sub-* directory or stamp subject_id in the manifest."
        )
    record["subject_id"] = str(sid)
    return str(sid)


def _subjects(records: list[dict[str, Any]]) -> list[str]:
    return sorted({_subject_of(r) for r in records})


def build_partition_stamp(
    *,
    route: str,
    strategy: str,
    manifest_sha: str | None,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The applied split, for ``provenance.json``: subject ids, counts and manifest digest."""

    def ids(items: list[dict[str, Any]]) -> list[str]:
        return sorted({str(r["subject_id"]) for r in items if r.get("subject_id")})

    stamp: dict[str, Any] = {
        "route": route,
        "strategy": strategy,
        "manifest_sha256": manifest_sha,
        "records": {"train": len(train), "val": len(val), "test": len(test)},
        "subjects": {"train": ids(train), "val": ids(val), "test": ids(test)},
    }
    if extra:
        stamp.update(extra)
    return stamp


def partition_paired_records(
    records: list[dict[str, Any]],
    data: Any,
    *,
    manifest_sha: str | None = None,
) -> PairedPartition:
    """Partition pool records (every pairing status, no split_hint filter) by the config.

    Raises:
        ValueError: On any declaration this route cannot honour (holdout_site, a
            conflicting manifest split, unknown subject ids, an empty split).
    """
    split = data.split
    if split.holdout_site:
        raise ValueError(
            "data.split.holdout_site is site-level and is not applied on the paired-manifest "
            "route (its records carry subject ids, not sites). Use split.type='loso_subject' "
            "with holdout_subject or loso_fold."
        )
    test_declared = declares_paired_test(data)
    strategy = "random" if split.type == "auto" else split.type
    if split.type == "manifest" and test_declared:
        raise ValueError(
            "split.type='manifest' takes its partition from the manifest split_hint, but "
            "data.enable_test_split / split.test_subjects also declare one. Pick one owner."
        )
    if strategy == "manifest":
        raise ValueError("split.type='manifest' is not a config-owned partition.")
    for r in records:
        _subject_of(r)
    paired = [r for r in records if r.get("pairing_status") == "paired"]
    unpaired = [r for r in records if r.get("pairing_status") != "paired"]
    test_ids = set(split.test_subjects or [])

    if strategy == "random":
        test_frac = float(data.test_split) if data.enable_test_split else 0.0
        train, val, test = split_index_three_way_grouped(
            paired, float(split.validation_fraction), test_frac, key=_subject_of
        )
        if split.validation_fraction > 0 and not val:
            raise ValueError("validation_fraction carves no validation subject from the pool.")
    elif strategy == "loso_subject":
        subjects = _subjects(paired)
        holdout = split.holdout_subject
        if holdout is None:
            fold = int(split.loso_fold or 0)
            if fold >= len(subjects):
                raise ValueError(
                    f"loso_fold={fold} but only {len(subjects)} subjects survive filtering "
                    f"({subjects}); folds are 0..{len(subjects) - 1}."
                )
            holdout = subjects[fold]
        if holdout not in subjects:
            raise ValueError(
                f"holdout_subject={holdout!r} is not among the subjects that survive "
                f"filtering: {subjects}"
            )
        _check_known(test_ids, subjects, "test_subjects")
        if holdout in test_ids:
            raise ValueError(f"subject {holdout!r} is both the validation and a test subject.")
        val = [r for r in paired if _subject_of(r) == holdout]
        test = [r for r in paired if _subject_of(r) in test_ids]
        train = [r for r in paired if _subject_of(r) not in ({holdout} | test_ids)]
    else:  # "subjects"
        val_ids = set(split.val_subjects or [])
        subjects = _subjects(paired)
        _check_known(val_ids, subjects, "val_subjects")
        _check_known(test_ids, subjects, "test_subjects")
        val = [r for r in paired if _subject_of(r) in val_ids]
        test = [r for r in paired if _subject_of(r) in test_ids]
        train = [r for r in paired if _subject_of(r) not in (val_ids | test_ids)]

    if test_declared and not test:
        raise ValueError(
            "a held-out test partition is declared but carves no paired test record; "
            "check data.test_split / split.test_subjects against the manifest subjects."
        )
    held_out = {_subject_of(r) for r in val + test}
    unpaired_admitted = [r for r in unpaired if _subject_of(r) not in held_out]
    if unpaired_admitted and data.pairing.unpaired_training:
        # A held-out subject's unpaired sessions would leak its anatomy into training.
        train = train + unpaired_admitted
    if not train:
        raise ValueError("config-owned partition leaves the training split empty.")
    return PairedPartition(
        train=train,
        val=val,
        test=test,
        stamp=build_partition_stamp(
            route="config_owned",
            strategy=split.type,
            manifest_sha=manifest_sha,
            train=train,
            val=val,
            test=test,
            extra={
                "unpaired_records": len(unpaired),
                "unpaired_held_out_dropped": len(unpaired) - len(unpaired_admitted),
                "unpaired_in_train": bool(data.pairing.unpaired_training),
            },
        ),
    )


def _check_known(ids: set[str], subjects: list[str], field: str) -> None:
    missing = sorted(ids - set(subjects))
    if missing:
        raise ValueError(f"split.{field} names subjects absent from the paired manifest: {missing}")


def resolve_paired_partition(manifest_path: Path, data: Any) -> PairedPartition:
    """Read the pool from ``manifest_path`` and partition it by ``data`` (config-owned)."""
    from spectramr.data.metadata.index_builder import IndexBuilder

    pool = IndexBuilder.load_paired_bids_manifest(
        manifest_path=manifest_path, split=None, config=data
    )
    return partition_paired_records(pool, data, manifest_sha=manifest_sha256(manifest_path))


def resolve_paired_test_ids(data: Any, manifest_path: Path) -> list[str]:
    """Subject ids of the declared test partition (pinned into the test-pass config)."""
    part = resolve_paired_partition(manifest_path, data)
    return list(part.stamp["subjects"]["test"])
