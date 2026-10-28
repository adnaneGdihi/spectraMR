"""Build the held-out M4Raw *motion* subset manifest (campaign arm C-2).

The M4Raw release [Lyu et al., Sci Data 2023] ships a set of deliberately
motion-corrupted acquisitions (``inter-scan_motion/`` / ``intra-scan_motion/``)
that Lyu et al. exclude from the standard training split. The VF campaign
(bundles P2/P3/P5) needs exactly those subjects as a *held-out* test set to
measure motion robustness — the network must never have trained on them.

This module is an offline manifest builder (CLAUDE.md pitfall #11: anything that
turns files into a dataset/manifest lives under :mod:`spectramr.data`). It reads an
existing M4Raw metadata manifest (the v3.0 JSON the repo already ships under
``data/manifests/m4raw_motion.json``), keeps only the records that belong to the
declared motion subjects, and writes a fresh v3.0 manifest the experiment YAMLs
reference via ``data.index_path``.

The subject-filtering logic — :func:`filter_motion_subjects` — is pure and
unit-tested on a tiny synthetic metadata structure; it needs no real data.

Run::

    python -m spectramr.data.manifests.build_m4raw_motion_subset \
        --source-manifest data/manifests/m4raw_motion.json \
        --output data/manifests/m4raw_motion_subset_held_out.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# M4Raw filenames look like ``<subjectID>_<CONTRAST><rep>.h5`` (e.g.
# ``2022061401_FLAIR01.h5``); the subject id is the leading numeric token.
_SUBJECT_ID_RE = re.compile(r"^(?P<subject>\d+)_")

# Path tokens M4Raw uses to flag deliberately motion-corrupted acquisitions.
DEFAULT_MOTION_TOKENS: tuple[str, ...] = (
    "inter-scan_motion",
    "intra-scan_motion",
    "motion",
)


def subject_id_from_record(record: dict[str, Any]) -> str | None:
    """Extract the M4Raw subject id from a manifest record.

    Args:
        record: A v3.0 manifest record with at least a ``filename`` (or
            ``file_id``) key, e.g. ``{"filename": "2022061401_T101.h5"}``.

    Returns:
        The leading numeric subject token (``"2022061401"``), or ``None`` if the
        filename does not match the expected M4Raw pattern.
    """
    name = record.get("filename") or record.get("file_id") or ""
    match = _SUBJECT_ID_RE.match(str(name))
    return match.group("subject") if match else None


def _record_is_motion(
    record: dict[str, Any],
    motion_tokens: tuple[str, ...],
) -> bool:
    """True if the record's relative path carries a motion token."""
    rel = str(record.get("relative_path") or record.get("filename") or "").lower()
    return any(token.lower() in rel for token in motion_tokens)


def filter_motion_subjects(
    records: list[dict[str, Any]],
    motion_subjects: set[str] | None = None,
    motion_tokens: tuple[str, ...] = DEFAULT_MOTION_TOKENS,
) -> list[dict[str, Any]]:
    """Keep only the records belonging to motion-corrupted subjects.

    Two complementary selection rules are OR-ed together so the builder works
    whether motion is encoded in the directory layout (path tokens) or supplied
    as an explicit subject allow-list:

    1. **Path-token rule.** Any record whose ``relative_path`` contains one of
       ``motion_tokens`` (e.g. ``inter-scan_motion/``) is a motion record.
    2. **Subject-allow-list rule.** If ``motion_subjects`` is given, any record
       whose subject id is in that set is kept (even if its path lacks a token).

    Args:
        records: The full list of v3.0 manifest records.
        motion_subjects: Optional explicit set of subject ids to retain. When
            ``None``, selection falls back to the path-token rule alone.
        motion_tokens: Path substrings that mark a motion acquisition.

    Returns:
        The filtered list of records (a new list; input is not mutated),
        deduplicated by ``filename`` and ordered as in the input.
    """
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        subject = subject_id_from_record(record)
        by_subject = motion_subjects is not None and subject in motion_subjects
        by_token = _record_is_motion(record, motion_tokens)
        if not (by_subject or by_token):
            continue
        key = str(record.get("filename") or record.get("file_id") or id(record))
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    return kept


def build_motion_subset_manifest(
    source_manifest: str | Path,
    output: str | Path,
    motion_subjects: set[str] | None = None,
    motion_tokens: tuple[str, ...] = DEFAULT_MOTION_TOKENS,
    dataset_name: str = "m4raw_motion_subset_held_out",
) -> dict[str, Any]:
    """Read an M4Raw metadata manifest and write the held-out motion subset.

    Args:
        source_manifest: Path to an existing v3.0 M4Raw manifest (the repo ships
            ``data/manifests/m4raw_motion.json``).
        output: Path the held-out manifest is written to.
        motion_subjects: Optional explicit subject allow-list (see
            :func:`filter_motion_subjects`).
        motion_tokens: Path substrings that mark a motion acquisition.
        dataset_name: ``dataset_name`` field of the written manifest.

    Returns:
        The manifest dict that was written.
    """
    source_path = Path(source_manifest)
    with source_path.open("r", encoding="utf-8") as fh:
        source = json.load(fh)

    records = source.get("records", [])
    kept = filter_motion_subjects(
        records, motion_subjects=motion_subjects, motion_tokens=motion_tokens
    )

    manifest = {
        "manifest_version": "3.0",
        "dataset_name": dataset_name,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "data_root": source.get("data_root", ""),
        "file_type": source.get("file_type", "h5"),
        "held_out": True,
        "selection": {
            "motion_tokens": list(motion_tokens),
            "motion_subjects": sorted(motion_subjects) if motion_subjects else [],
        },
        "total_records": len(kept),
        "records": kept,
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        default="data/manifests/m4raw_motion.json",
        help="Existing v3.0 M4Raw manifest to filter.",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/m4raw_motion_subset_held_out.json",
        help="Destination for the held-out motion-subset manifest.",
    )
    parser.add_argument(
        "--motion-subjects",
        nargs="*",
        default=None,
        help="Optional explicit subject-id allow-list (space separated).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    subjects = set(args.motion_subjects) if args.motion_subjects else None
    manifest = build_motion_subset_manifest(
        source_manifest=args.source_manifest,
        output=args.output,
        motion_subjects=subjects,
    )
    logger.info(
        "Wrote %s held-out motion records to %s",
        manifest["total_records"],
        args.output,
    )


if __name__ == "__main__":  # pragma: no cover
    # Standalone entrypoint: attach a handler to this module's logger only; the root
    # logger belongs to infrastructure/logging/log_policy.py.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    main()
