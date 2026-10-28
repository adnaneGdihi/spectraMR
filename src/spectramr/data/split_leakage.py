"""Train/val split-leakage analysis (data SSOT).

This module is the data-layer half of the audit guard
``ConfigHealthChecker.check_train_val_split_leakage``. Reading a manifest
file into records is a data concern (pitfall #7/#11: file→records code lives
under ``src/spectramr/data/``), so the raw scan lives here and the infrastructure
checker only *consumes* the report — it never opens a manifest itself.

Why a dedicated lightweight reader instead of ``parse_fastmri_index`` /
``read_external_manifest``: those are the eager runtime loaders — they resolve
every on-disk path and RAISE on a missing file / empty ``records[]`` (correct
for a real run). The audit must instead *tolerate* absence (manifests are
gitignored and only present on the cluster) and scan only record identity, not
voxels. So we read the JSON and pull the subject/file keys, nothing more.

Leak model
----------
The one statically-detectable, high-value leak is a subject (or an exact
on-disk file) present in BOTH splits. The framework's own random/LOSO/directory
splits partition a *single* index via :func:`spectramr.data.split_utils.split_index`
(file-level, deterministic), so those are disjoint by construction — only a
subject straddling the boundary, or the SAME subject appearing at both the LOSO
holdout site and elsewhere, can leak. The ``manifest`` split reads two
independently-authored files with no code enforcing disjointness — that is where
the real leaks live (the 2026-07-07 mrixfields cluster failure).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spectramr.data.split_utils import (
    FILE_KEYS as _FILE_KEYS,
    SUBJECT_KEYS as _SUBJECT_KEYS,
    record_split_key,
    split_index_grouped,
)

__all__ = [
    "SplitLeakageReport",
    "analyze_paired_partition_leakage",
    "analyze_split_leakage",
    "analyze_test_split_leakage",
    "read_manifest_records",
]

#: Cap on how many overlapping identities we surface (keep messages bounded).
_OVERLAP_SAMPLE_CAP = 20



@dataclass(frozen=True)
class SplitLeakageReport:
    """Outcome of a train/val overlap analysis.

    Attributes
    ----------
    status:
        ``"leak"`` (overlap found), ``"clean"`` (no overlap), or ``"skipped"``
        (indeterminate — manifests absent locally, or a strategy resolved only
        at runtime). ``"skipped"`` is neither pass nor fail.
    mode:
        Which split regime was analysed: ``"explicit_manifest"``,
        ``"single_index"``, ``"loso"``, ``"directory"``, or ``"unknown"``.
    key_kind:
        ``"subject"`` or ``"file"`` — which identity flagged the leak
        (``None`` when clean/skipped).
    overlap:
        A bounded sample of the overlapping identities.
    n_train, n_val:
        Record counts per split (0 when skipped).
    detail:
        Human-readable one-line explanation.
    """

    status: str
    mode: str
    key_kind: str | None
    overlap: tuple[str, ...]
    n_train: int
    n_val: int
    detail: str


def _first_key(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = record.get(k)
        if v:
            return str(v)
    return None


def _subject_ids(records: list[dict[str, Any]]) -> set[str]:
    return {s for r in records if (s := _first_key(r, _SUBJECT_KEYS)) is not None}


def _file_ids(records: list[dict[str, Any]]) -> set[str]:
    return {s for r in records if (s := _first_key(r, _FILE_KEYS)) is not None}


def read_manifest_records(path: str | Path) -> list[dict[str, Any]] | None:
    """Return the record list of a JSON manifest, or ``None`` if unreadable.

    Tolerant by design (the audit must not crash on a gitignored/absent
    manifest): a missing file, unparseable JSON, or an unrecognised shape all
    yield ``None`` (→ the caller reports ``skipped``). Accepts a bare top-level
    list, or a dict wrapping the records under ``records`` / ``files`` /
    ``samples`` (the shapes emitted across this repo's manifest generators).
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("records", "files", "samples"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                records = candidate
                break
        else:
            return None
    else:
        return None
    return [r for r in records if isinstance(r, dict)]


def _resolve(path: str) -> str:
    """Resolve a config path via ``PathResolver``, tolerating any failure.

    The analyzer runs at audit time on hosts where cluster-env resolution may
    be unavailable; falling back to the literal path keeps it hermetic (an
    absolute test path resolves to itself).
    """
    try:
        from spectramr.data.metadata.path_resolver import PathResolver

        return str(PathResolver.resolve(path))
    except Exception:
        return path


def _overlap_report(
    mode: str,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    sides: tuple[str, str] = ("train", "validation"),
) -> SplitLeakageReport:
    """Compare two record lists on subject identity first, then file identity.

    ``sides`` names the two lists in the detail text; the held-out analysis
    passes ``("training pool", "held-out test")``.
    """
    left, right = sides
    subj_overlap = sorted(_subject_ids(train) & _subject_ids(val))
    if subj_overlap:
        return SplitLeakageReport(
            status="leak",
            mode=mode,
            key_kind="subject",
            overlap=tuple(subj_overlap[:_OVERLAP_SAMPLE_CAP]),
            n_train=len(train),
            n_val=len(val),
            detail=(
                f"{len(subj_overlap)} subject id(s) appear in BOTH the {left} "
                f"and {right} split — patient-level data leakage."
            ),
        )
    file_overlap = sorted(_file_ids(train) & _file_ids(val))
    if file_overlap:
        return SplitLeakageReport(
            status="leak",
            mode=mode,
            key_kind="file",
            overlap=tuple(file_overlap[:_OVERLAP_SAMPLE_CAP]),
            n_train=len(train),
            n_val=len(val),
            detail=(
                f"{len(file_overlap)} file path(s) appear in BOTH splits — the "
                f"same on-disk volume is used for {left} and {right}."
            ),
        )
    return SplitLeakageReport(
        status="clean",
        mode=mode,
        key_kind=None,
        overlap=(),
        n_train=len(train),
        n_val=len(val),
        detail=f"{left} and {right} splits are disjoint.",
    )


def _skipped(mode: str, detail: str) -> SplitLeakageReport:
    return SplitLeakageReport(
        status="skipped",
        mode=mode,
        key_kind=None,
        overlap=(),
        n_train=0,
        n_val=0,
        detail=detail,
    )


#: Dataset types whose directory crawl ``IndexBuilder.build_nifti_index`` owns.
#: Only those are replayed here; any other directory route is reported as
#: not replayed, never as "disjoint by construction".
_NIFTI_ROUTES = frozenset(
    {"nifti", "nifti_paired", "paired_nifti", "paired_mri", "bids", "bids_paired"}
)


def _directory_report(data: Any) -> SplitLeakageReport:
    """Replay the loader's own directory crawl for train and val, then compare.

    Until 2026-09-02 this route was skipped as "folder-disjoint by
    construction" -- and the premise was false: ``IndexBuilder`` split the
    crawled FILE list, so a subject's T1w could train while its T2w validated,
    and the guard for that leak returned green on exactly that route. The
    crawl needs the data on disk, so a host without it still skips, with the
    honest reason.
    """
    source = getattr(data, "source", None)
    raw_root = getattr(source, "root", None)
    if not raw_root:
        return _skipped("directory", "no data.source.root declared; nothing to crawl.")
    dataset_type = str(getattr(data, "dataset_type", "") or "")
    if dataset_type not in _NIFTI_ROUTES:
        return _skipped(
            "directory",
            f"directory crawl for dataset_type={dataset_type!r} is not replayed by this "
            "analyzer (only the NIfTI/BIDS routes are); split disjointness is UNVERIFIED here.",
        )
    root = _resolve(str(raw_root))
    if not Path(root).exists():
        return _skipped(
            "directory",
            f"data root {root!r} not present on this host (the crawl runs on the cluster).",
        )
    try:
        from spectramr.data.metadata.index_builder import IndexBuilder

        train = IndexBuilder.build_nifti_index(data, "train")
        val = IndexBuilder.build_nifti_index(data, "val")
    except Exception as exc:  # the analyzer never raises; it reports
        return _skipped("directory", f"directory crawl could not be replayed here: {exc}")
    if not train and not val:
        return _skipped("directory", f"directory crawl under {root!r} found no records.")
    return _overlap_report("directory", train, val)


def analyze_test_split_leakage(config: Any) -> SplitLeakageReport:
    """Analyse a declared HELD-OUT test manifest against the training pool.

    The pool is every record the run can see while training or selecting a
    checkpoint: ``index_path`` plus ``validation_index_path``. A subject (or
    file) present in both the pool and ``test_index_path`` makes the reported
    number a training-set number. Skips, with the reason, when no test set is
    declared or the manifests are not on this host; never raises.
    """
    data = getattr(config, "data", None)
    if data is None:
        return _skipped("unknown", "no data section.")
    source = getattr(data, "source", None)
    test_path = getattr(source, "test_index_path", None)
    if not test_path:
        return _skipped(
            "held_out_test",
            "no data.source.test_index_path declared: the arm reports on its "
            "validation (checkpoint-selection) set.",
        )
    test = read_manifest_records(_resolve(str(test_path)))
    if test is None:
        return _skipped(
            "held_out_test",
            "test manifest not present locally (gitignored — the audit runs for real on the cluster).",
        )
    pool: list[dict[str, Any]] = []
    for key in ("index_path", "validation_index_path"):
        path = getattr(source, key, None)
        if not path:
            continue
        records = read_manifest_records(_resolve(str(path)))
        if records is None:
            return _skipped(
                "held_out_test",
                f"{key} manifest not present locally; cannot compare the test set against it.",
            )
        pool.extend(records)
    if not pool:
        return _skipped(
            "held_out_test",
            "no train/validation manifest declared to compare the test set against.",
        )
    return _overlap_report("held_out_test", pool, test, sides=("training pool", "held-out test"))


def partition_leakage_report(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> SplitLeakageReport:
    """Subject/file overlap across every pair of a train/val/test partition.

    Repeat scans of one subject (``sub-X/ses-NN``) share ``subject_id``, so a
    subject straddling two splits is caught even when its files differ.
    """
    pairs = [
        ("train", "validation", train, val),
        ("train", "held-out test", train, test),
        ("validation", "held-out test", val, test),
    ]
    last: SplitLeakageReport | None = None
    for left_name, right_name, left, right in pairs:
        if not left or not right:
            continue
        report = _overlap_report(
            "paired_partition", left, right, sides=(left_name, right_name)
        )
        if report.status == "leak":
            return report
        last = report
    if last is None:
        return _skipped("paired_partition", "fewer than two non-empty splits to compare.")
    return last


def analyze_paired_partition_leakage(data: Any) -> SplitLeakageReport:
    """Replay the paired-manifest route's real partition and check it for leaks.

    A config-owned partition is replayed through the loader's own function. On
    the ``split_hint`` route the train and validation records are read by the
    same reader the loader uses.
    """
    from spectramr.data.metadata.index_builder import IndexBuilder
    from spectramr.data.metadata.paired_partition import (
        is_config_owned,
        resolve_paired_partition,
    )

    path = _resolve(str(data.source.paired_manifest_path))
    if not Path(path).exists():
        return _skipped(
            "paired_partition",
            "paired manifest not present locally (gitignored — runs on cluster).",
        )
    try:
        if is_config_owned(data):
            part = resolve_paired_partition(Path(path), data)
            return partition_leakage_report(part.train, part.val, part.test)
        train = IndexBuilder.load_paired_bids_manifest(Path(path), split="train", config=data)
        val = IndexBuilder.load_paired_bids_manifest(Path(path), split="val", config=data)
    except ValueError as exc:
        # The loader raises on the same declaration at load time; the audit reports why.
        return _skipped("paired_partition", f"partition does not resolve: {exc}")
    return partition_leakage_report(train, val, [])


def analyze_split_leakage(config: Any) -> SplitLeakageReport:
    """Analyse a config's train/val split for subject/file leakage.

    Branches every split strategy the framework supports (mirroring the
    resolution cascade in ``ManifestLoader.load_fastmri_splits``):

    * two manifests (``index_path`` + ``validation_index_path``) → compare the
      two record sets directly (the real leak surface);
    * single index + ``holdout_site`` → LOSO: a subject at both the holdout
      site and elsewhere leaks;
    * single index, no holdout → ``random``: replay the deterministic
      subject-grouped split and check for a subject straddling the boundary;
    * directory crawl with no index files → replay ``IndexBuilder``'s crawl for
      both splits and compare (skipped only when the data is not on this host).

    Never raises and never false-positives: an indeterminate case (absent
    manifest, runtime-only directory split) returns ``status="skipped"``.
    """
    data = getattr(config, "data", None)
    if data is None:
        return _skipped("unknown", "no data section.")

    if data.source.paired_manifest_path:
        return analyze_paired_partition_leakage(data)

    index_path = data.source.index_path
    val_index_path = data.source.validation_index_path
    holdout_site = data.split.holdout_site
    validation_split = data.split.validation_fraction
    if validation_split is None:
        validation_split = 0.1

    # 1. Two independently-authored manifests — the primary leak surface.
    if index_path and val_index_path:
        train = read_manifest_records(_resolve(str(index_path)))
        val = read_manifest_records(_resolve(str(val_index_path)))
        if train is None or val is None:
            return _skipped(
                "explicit_manifest",
                "train and/or validation manifest not present locally "
                "(gitignored — the audit runs for real on the cluster).",
            )
        return _overlap_report("explicit_manifest", train, val)

    # Everything below partitions a SINGLE index file. Without one, the split
    # is a runtime directory crawl: replay it for both splits and compare.
    if not index_path:
        return _directory_report(data)

    records = read_manifest_records(_resolve(str(index_path)))
    if records is None:
        return _skipped(
            "single_index",
            "index manifest not present locally (gitignored — runs on cluster).",
        )

    # 2. LOSO — partition by site membership; a subject at both the holdout
    #    site and elsewhere leaks (mirrors the loader's site-match cascade).
    if holdout_site:
        holdout = str(holdout_site).lower()
        held, rest = [], []
        for r in records:
            meta = r.get("metadata", {}) or {}
            sites = [
                str(s).lower()
                for s in (
                    meta.get("site"),
                    meta.get("institution"),
                    meta.get("institutionName"),
                    meta.get("systemVendor"),
                )
                if s
            ]
            (held if any(holdout in s for s in sites) else rest).append(r)
        return _overlap_report("loso", rest, held)

    # 3. Random — replay the subject-grouped split every route makes, check for
    #    a subject straddling the held-out boundary.
    if float(validation_split) <= 0.0:
        return SplitLeakageReport(
            status="clean",
            mode="single_index",
            key_kind=None,
            overlap=(),
            n_train=len(records),
            n_val=0,
            detail="validation_split<=0: train-only run, no validation set.",
        )
    if len({record_split_key(r) for r in records}) < 2:
        return _skipped(
            "single_index",
            "fewer than 2 subjects; the runtime split raises on this itself.",
        )
    train, val = split_index_grouped(records, float(validation_split), key=record_split_key)
    return _overlap_report("single_index", train, val)
