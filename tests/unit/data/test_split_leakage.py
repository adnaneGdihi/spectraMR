"""Unit tests for the train/val split-leakage analyzer (data SSOT).

``analyze_split_leakage`` is the data-layer half of the audit's
``check_train_val_split_leakage`` guard. It must:

* Detect the real, statically-detectable leak — the SAME subject (or the
  same on-disk file) present in BOTH the train manifest and the val
  manifest (the two-independent-files ``manifest`` split, which no code
  guarantees disjoint — the 2026-07-07 mrixfields cluster failure family).
* Support EVERY split strategy without crashing or false-positiving:
  ``random`` (single-index, deterministic ``split_index`` — only a subject
  straddling the boundary leaks), ``loso`` (a subject appearing at both the
  holdout site and elsewhere), ``directory``/``auto`` (folder-disjoint by
  construction — skip with info).
* Skip (never fail, never falsely pass) when the manifest files are absent
  locally — they are gitignored and regenerated on the cluster, so the
  check runs for real there.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from spectramr.data.split_leakage import (
    SplitLeakageReport,
    analyze_split_leakage,
    read_manifest_records,
)
from tests.utils.data_config_stub import DataConfigStub


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({"data_root": ".", "records": records}))


def _rec(subject_id: str, *, field: float = 3.0, contrast: str = "T1w") -> dict[str, Any]:
    rel = f"{subject_id}/{contrast}/{field}/vol.nii.gz"
    return {
        "relative_path": rel,
        "file_id": rel,
        "field_strength": field,
        "contrast": contrast,
        "subject_id": subject_id,
        "pairing_group": f"{subject_id}|{contrast}",
    }


def _cfg(
    *,
    index_path: str | None = None,
    validation_index_path: str | None = None,
    validation_split: float = 0.1,
    holdout_site: str | None = None,
    split_strategy: str = "manifest",
    dataset_type: str | None = None,
) -> Any:
    # The SHARED stub, not a hand-rolled namespace. This one had been migrated
    # for `data.split.*` but not for `data.source.*`, so every reader of
    # `data.source.index_path` hit "'SimpleNamespace' object has no attribute
    # 'source'". DataConfigStub carries every sub-block and routes the flat
    # legacy kwargs from RENAMES, so it cannot fall behind the next fold the way
    # a hand-written shape does.
    return SimpleNamespace(
        data=DataConfigStub(
            dataset_type=dataset_type,
            index_path=index_path,
            validation_index_path=validation_index_path,
            split=SimpleNamespace(
                validation_fraction=validation_split,
                holdout_site=holdout_site,
                type=split_strategy,
            ),
        )
    )


# --------------------------------------------------------------------------- #
# read_manifest_records
# --------------------------------------------------------------------------- #
def test_read_manifest_records_records_wrapper(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    _write_manifest(p, [_rec("s0"), _rec("s1")])
    recs = read_manifest_records(p)
    assert recs is not None and len(recs) == 2
    assert {r["subject_id"] for r in recs} == {"s0", "s1"}


def test_read_manifest_records_bare_list(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text(json.dumps([_rec("s0")]))
    recs = read_manifest_records(p)
    assert recs is not None and recs[0]["subject_id"] == "s0"


def test_read_manifest_records_missing_returns_none(tmp_path: Path) -> None:
    assert read_manifest_records(tmp_path / "nope.json") is None


# --------------------------------------------------------------------------- #
# two-manifest (explicit) — the primary leak case
# --------------------------------------------------------------------------- #
def test_explicit_manifest_subject_overlap_is_leak(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    _write_manifest(train, [_rec("subjA"), _rec("subjB"), _rec("subjC")])
    _write_manifest(val, [_rec("subjC"), _rec("subjD")])  # subjC leaks
    rep = analyze_split_leakage(_cfg(index_path=str(train), validation_index_path=str(val)))
    assert isinstance(rep, SplitLeakageReport)
    assert rep.status == "leak"
    assert rep.mode == "explicit_manifest"
    assert rep.key_kind == "subject"
    assert "subjC" in rep.overlap


def test_explicit_manifest_disjoint_is_clean(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    _write_manifest(train, [_rec("Training_vol0"), _rec("Training_vol1")])
    # mrixfields-correct: val subjects are split-namespaced, never colliding.
    _write_manifest(val, [_rec("Validating_vol0"), _rec("Validating_vol1")])
    rep = analyze_split_leakage(_cfg(index_path=str(train), validation_index_path=str(val)))
    assert rep.status == "clean"
    assert rep.overlap == ()


def test_explicit_manifest_file_overlap_is_leak_even_if_subject_differs(
    tmp_path: Path,
) -> None:
    # Same physical file re-labelled with a different subject_id in val: the
    # subject sets are disjoint but the file identity collides — still a leak.
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    shared = _rec("Training_vol0")
    dup = dict(shared)
    dup["subject_id"] = "Validating_vol0"  # relabelled, same relative_path
    _write_manifest(train, [shared, _rec("Training_vol1")])
    _write_manifest(val, [dup])
    rep = analyze_split_leakage(_cfg(index_path=str(train), validation_index_path=str(val)))
    assert rep.status == "leak"
    assert rep.key_kind == "file"
    assert shared["relative_path"] in rep.overlap


def test_explicit_manifest_absent_files_skipped(tmp_path: Path) -> None:
    # Gitignored locally — must skip (not crash, not falsely pass).
    rep = analyze_split_leakage(
        _cfg(
            index_path=str(tmp_path / "train.json"),
            validation_index_path=str(tmp_path / "val.json"),
        )
    )
    assert rep.status == "skipped"


# --------------------------------------------------------------------------- #
# single-index random — support the random strategy
# --------------------------------------------------------------------------- #
def test_single_index_keeps_a_straddling_subject_on_one_side(tmp_path: Path) -> None:
    # A by-file split held out the LAST n_val records, so s2 landed on both sides
    # (the 49-subject leak on the fastMRI knee HSSC arms). The split is grouped
    # on the subject now, and the audit replays that split.
    idx = tmp_path / "idx.json"
    recs = [
        _rec("s0", field=1.5),
        _rec("s1", field=1.5),
        _rec("s2", field=1.5),
        _rec("s2", field=3.0),  # s2 also in the held-out tail
    ]
    _write_manifest(idx, recs)
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), validation_split=0.25, split_strategy="random")
    )
    assert rep.status == "clean"
    assert rep.mode == "single_index"


def test_a_self_indexed_creator_keeps_a_straddling_subject_on_one_side(
    tmp_path: Path,
) -> None:
    # npy_slice builds and splits its own index; it now groups on the subject too,
    # and the audit replays that split.
    idx = tmp_path / "idx.json"
    recs = [
        _rec("s0", field=1.5),
        _rec("s1", field=1.5),
        _rec("s2", field=1.5),
        _rec("s2", field=3.0),
    ]
    _write_manifest(idx, recs)
    rep = analyze_split_leakage(
        _cfg(
            index_path=str(idx),
            validation_split=0.25,
            split_strategy="random",
            dataset_type="npy_slice",
        )
    )
    assert rep.status == "clean"


def test_single_index_with_one_subject_is_skipped_not_raised(tmp_path: Path) -> None:
    idx = tmp_path / "idx.json"
    _write_manifest(idx, [_rec("s0"), _rec("s0", field=3.0)])
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), validation_split=0.2, split_strategy="random")
    )
    assert rep.status == "skipped"


def test_single_index_clean_when_subjects_contiguous(tmp_path: Path) -> None:
    idx = tmp_path / "idx.json"
    recs = [_rec(f"s{i}") for i in range(10)]  # every subject unique
    _write_manifest(idx, recs)
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), validation_split=0.2, split_strategy="random")
    )
    assert rep.status == "clean"


def test_single_index_train_only_split_is_clean(tmp_path: Path) -> None:
    idx = tmp_path / "idx.json"
    _write_manifest(idx, [_rec("s0"), _rec("s0")])  # duplicate subject, but no val
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), validation_split=0.0, split_strategy="random")
    )
    assert rep.status == "clean"  # train-only: nothing to leak into


# --------------------------------------------------------------------------- #
# loso — support leave-one-site-out
# --------------------------------------------------------------------------- #
def test_loso_subject_across_sites_is_leak(tmp_path: Path) -> None:
    idx = tmp_path / "idx.json"
    a = _rec("patX")
    a["metadata"] = {"site": "siteA"}
    b = _rec("patX", field=7.0)
    b["metadata"] = {"site": "siteB"}  # same patient, holdout site
    c = _rec("patY")
    c["metadata"] = {"site": "siteA"}
    _write_manifest(idx, [a, b, c])
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), holdout_site="siteB", split_strategy="loso")
    )
    assert rep.status == "leak"
    assert rep.mode == "loso"
    assert "patX" in rep.overlap


def test_loso_disjoint_sites_clean(tmp_path: Path) -> None:
    idx = tmp_path / "idx.json"
    a = _rec("patX")
    a["metadata"] = {"site": "siteA"}
    b = _rec("patY")
    b["metadata"] = {"site": "siteB"}
    _write_manifest(idx, [a, b])
    rep = analyze_split_leakage(
        _cfg(index_path=str(idx), holdout_site="siteB", split_strategy="loso")
    )
    assert rep.status == "clean"


# --------------------------------------------------------------------------- #
# directory crawl — replayed when the data is here, skipped honestly when not
# (cohort review 2026-09-02, T0.2: the old "folder-disjoint by construction"
# skip returned green on the one route that split a flat FILE list)
# --------------------------------------------------------------------------- #
def test_directory_split_without_a_data_root_is_skipped(tmp_path: Path) -> None:
    rep = analyze_split_leakage(_cfg(split_strategy="directory"))
    assert rep.status == "skipped"
    assert rep.mode in {"directory", "unknown"}


def _dir_cfg(root: Path, *, dataset_type: str = "nifti", validation_split: float = 0.5) -> Any:
    return SimpleNamespace(
        data=DataConfigStub(
            data_root=str(root),
            data_layout="flat",
            dataset_type=dataset_type,
            index_path=None,
            validation_index_path=None,
            holdout_site=None,
            validation_split=validation_split,
            datasets=None,
            contrasts=None,
            target_contrasts=None,
        )
    )


def _bids_named_files(root: Path) -> None:
    for s in (1, 2, 3):
        for c in ("T1w", "T2w"):
            (root / f"sub-{s:02d}_{c}.nii.gz").write_bytes(b"")


def test_directory_crawl_is_replayed_and_clean_when_grouped_by_subject(tmp_path: Path) -> None:
    _bids_named_files(tmp_path)
    rep = analyze_split_leakage(_dir_cfg(tmp_path))
    assert rep.status == "clean", rep.detail
    assert rep.mode == "directory"
    assert rep.n_train + rep.n_val == 6


def test_directory_crawl_that_straddles_a_subject_is_a_leak(tmp_path: Path, monkeypatch) -> None:
    """The planted violation: the pre-review file-level split, replayed."""
    from spectramr.data.metadata.index_builder import IndexBuilder

    _bids_named_files(tmp_path)

    def _file_level(config, split):  # sub-02 straddles
        recs = [
            {
                "primary_path": f"sub-{s:02d}_{c}.nii.gz",
                "file_id": f"sub-{s:02d}_{c}",
                "subject_id": f"sub-{s:02d}",
            }
            for s in (1, 2, 3)
            for c in ("T1w", "T2w")
        ]
        return recs[:3] if split == "train" else recs[3:]

    monkeypatch.setattr(IndexBuilder, "build_nifti_index", staticmethod(_file_level))
    rep = analyze_split_leakage(_dir_cfg(tmp_path))
    assert rep.status == "leak"
    assert rep.key_kind == "subject"
    assert rep.overlap == ("sub-02",)


def test_directory_crawl_with_absent_root_is_skipped_with_the_reason(tmp_path: Path) -> None:
    rep = analyze_split_leakage(_dir_cfg(tmp_path / "not_here"))
    assert rep.status == "skipped"
    assert "not present on this host" in rep.detail


def test_directory_crawl_for_a_non_nifti_route_is_reported_unverified(tmp_path: Path) -> None:
    """A route this analyzer cannot replay is never called disjoint by construction."""
    _bids_named_files(tmp_path)
    rep = analyze_split_leakage(_dir_cfg(tmp_path, dataset_type="kspace"))
    assert rep.status == "skipped"
    assert "UNVERIFIED" in rep.detail


# --------------------------------------------------------------------------- #
# held-out test manifest vs the training pool (cohort review 2026-09-02, T0.3)
# --------------------------------------------------------------------------- #
from spectramr.data.split_leakage import analyze_test_split_leakage  # noqa: E402


def _held_out_cfg(train: str | None, val: str | None, test: str | None) -> Any:
    return SimpleNamespace(
        data=SimpleNamespace(
            source=SimpleNamespace(
                index_path=train, validation_index_path=val, test_index_path=test
            )
        )
    )


def test_held_out_subject_in_training_pool_is_a_leak(tmp_path: Path) -> None:
    """The planted violation: sub-02 trains AND is reported on."""
    train, val, test = tmp_path / "train.json", tmp_path / "val.json", tmp_path / "test.json"
    _write_manifest(train, [_rec("sub-01"), _rec("sub-02")])
    _write_manifest(val, [_rec("sub-03")])
    _write_manifest(test, [_rec("sub-02"), _rec("sub-04")])
    rep = analyze_test_split_leakage(_held_out_cfg(str(train), str(val), str(test)))
    assert rep.status == "leak" and rep.mode == "held_out_test"
    assert rep.key_kind == "subject" and rep.overlap == ("sub-02",)
    assert "held-out test" in rep.detail


def test_held_out_subject_in_validation_is_also_a_leak(tmp_path: Path) -> None:
    """Validation selects the checkpoint, so it is part of the pool."""
    train, val, test = tmp_path / "train.json", tmp_path / "val.json", tmp_path / "test.json"
    _write_manifest(train, [_rec("sub-01")])
    _write_manifest(val, [_rec("sub-03")])
    _write_manifest(test, [_rec("sub-03")])
    rep = analyze_test_split_leakage(_held_out_cfg(str(train), str(val), str(test)))
    assert rep.status == "leak" and rep.overlap == ("sub-03",)


def test_held_out_disjoint_is_clean(tmp_path: Path) -> None:
    train, val, test = tmp_path / "train.json", tmp_path / "val.json", tmp_path / "test.json"
    _write_manifest(train, [_rec("sub-01"), _rec("sub-02")])
    _write_manifest(val, [_rec("sub-03")])
    _write_manifest(test, [_rec("sub-04")])
    rep = analyze_test_split_leakage(_held_out_cfg(str(train), str(val), str(test)))
    assert rep.status == "clean" and rep.n_train == 3 and rep.n_val == 1


def test_held_out_undeclared_is_skipped_with_the_reason() -> None:
    rep = analyze_test_split_leakage(_held_out_cfg("train.json", None, None))
    assert rep.status == "skipped" and "validation (checkpoint-selection) set" in rep.detail


def test_held_out_absent_manifest_is_skipped(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    _write_manifest(train, [_rec("sub-01")])
    rep = analyze_test_split_leakage(
        _held_out_cfg(str(train), None, str(tmp_path / "missing.json"))
    )
    assert rep.status == "skipped" and "not present locally" in rep.detail


def _scan(subject: str, ses: str) -> dict:
    return {
        "subject_id": subject,
        "primary_path": f"/data/sub-{subject}/ses-{ses}/anat/input.nii.gz",
        "file_id": f"sub-{subject}_ses-{ses}",
    }


def test_partition_leakage_clean_when_subjects_are_disjoint():
    from spectramr.data.split_leakage import partition_leakage_report

    report = partition_leakage_report(
        [_scan("01", "01"), _scan("01", "02")],
        [_scan("02", "01"), _scan("02", "02")],
        [_scan("03", "01")],
    )
    assert report.status == "clean"


def test_partition_leakage_flags_subject_in_two_splits():
    from spectramr.data.split_leakage import partition_leakage_report

    report = partition_leakage_report(
        [_scan("01", "01"), _scan("02", "01")],
        [_scan("02", "02")],
        [_scan("03", "01")],
    )
    assert report.status == "leak"
    assert "02" in report.overlap


def test_partition_leakage_flags_repeat_scan_straddling_the_boundary():
    """ses-01 in train and ses-02 in validation are different files of one subject."""
    from spectramr.data.split_leakage import partition_leakage_report

    report = partition_leakage_report(
        [_scan("05", "01"), _scan("06", "01")],
        [_scan("05", "02")],
        [_scan("07", "01")],
    )
    assert report.status == "leak"
    assert report.key_kind == "subject"
    assert "05" in report.overlap


def test_partition_leakage_flags_test_overlapping_validation():
    from spectramr.data.split_leakage import partition_leakage_report

    report = partition_leakage_report(
        [_scan("01", "01")],
        [_scan("07", "01")],
        [_scan("07", "02")],
    )
    assert report.status == "leak"
    assert "held-out test" in report.detail
    assert "07" in report.overlap
