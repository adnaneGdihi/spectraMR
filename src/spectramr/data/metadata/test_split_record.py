"""``test_split.json``: how a config-owned test partition was carved, so a later run re-derives it.

An arm with ``data.source.test_index_path`` names its test set by file. An arm whose test set is
carved from a paired manifest by the config (``split.type`` random / loso_subject / subjects) has
no such file: the set exists only as the output of the partition. The first run writes this
record into its run directory; every later run on that directory (``infer --test-split``, a
validate-only re-score, a resume) re-derives the partition and raises if it differs, so the
numbers it reports are on the subjects the run was trained without.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spectramr.core.run_artifacts import INFERENCE_SUBDIR

TEST_SPLIT_RECORD = "test_split.json"


def derived_test_split_record(data: Any, stamp: dict[str, Any] | None) -> dict[str, Any] | None:
    """The record for ``stamp`` (``DataPipelineDirector.split_provenance``), or ``None``.

    ``None`` unless the test set is config-owned and non-empty: a manifest-named test set
    needs no recipe. ``seed`` is ``None`` because the partition draws no random numbers; it
    carves the test subjects from the tail of the manifest's subject order.
    """
    if not stamp or stamp.get("route") != "config_owned":
        return None
    test_ids = list((stamp.get("subjects") or {}).get("test") or [])
    if not test_ids:
        return None
    split = data.split
    return {
        "strategy": stamp.get("strategy"),
        "seed": None,
        "order": "manifest subject order; test carved from the tail, then validation",
        "validation_fraction": float(split.validation_fraction),
        "test_fraction": float(data.test_split) if data.enable_test_split else 0.0,
        "holdout_subject": split.holdout_subject,
        "loso_fold": split.loso_fold,
        "test_subjects_declared": list(split.test_subjects or []),
        "manifest_path": str(data.source.paired_manifest_path),
        "manifest_sha256": stamp.get("manifest_sha256"),
        "test_subjects": sorted(test_ids),
    }


def write_or_verify_test_split(run_dir: Path, record: dict[str, Any] | None) -> None:
    """Write ``record`` on the first run; on any later run, raise if the test set changed.

    A campaign inference run writes into ``<arm>/inference/``; its partition is verified against
    the training run's record one level up, which must exist.

    Raises:
        ValueError: The run directory's recorded test subjects or manifest digest differ from
            the ones this run derived, or an inference run finds no training record.
    """
    if record is None:
        return
    run_dir = Path(run_dir)
    path = run_dir / TEST_SPLIT_RECORD
    if run_dir.name == INFERENCE_SUBDIR:
        trained = run_dir.parent / TEST_SPLIT_RECORD
        if not trained.is_file():
            raise ValueError(
                f"{trained} is missing: the training run recorded no test partition, so this "
                "inference run cannot show it grades the subjects the arm was trained without."
            )
        _verify(trained, record)
        path.write_text(json.dumps(record, indent=2, sort_keys=True))
        return
    if not path.is_file():
        path.write_text(json.dumps(record, indent=2, sort_keys=True))
        return
    _verify(path, record)


def _verify(path: Path, record: dict[str, Any]) -> None:
    stored = json.loads(path.read_text())
    for key in ("test_subjects", "manifest_sha256"):
        if stored.get(key) != record[key]:
            raise ValueError(
                f"{path}: the recorded {key} differs from the one this run derived "
                f"({stored.get(key)!r} vs {record[key]!r}). The test set is not the one this run "
                "was trained without; restore the manifest and split settings it recorded."
            )


__all__ = ["TEST_SPLIT_RECORD", "derived_test_split_record", "write_or_verify_test_split"]
