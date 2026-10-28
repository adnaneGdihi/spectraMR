"""Config-owned paired partition: subject-grouped, leak-free, unpaired never scored."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from spectramr.config.schemas.data import DataPairingConfigSchema, DataSplitConfigSchema
from spectramr.data.metadata.paired_partition import (
    build_partition_stamp,
    holdout_site_on_paired_route,
    is_config_owned,
    partition_paired_records,
    resolve_paired_partition,
)
from spectramr.data.split_leakage import partition_leakage_report

SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]


def _rec(subject: str, ses: str = "01", run: int = 1, status: str = "paired") -> dict:
    stem = f"/data/sub-{subject}/ses-{ses}/anat/run-{run}"
    return {
        "subject_id": subject,
        "file_id": f"sub-{subject}_ses-{ses}_run-{run}",
        "primary_path": f"{stem}/input.nii.gz",
        "target_path": f"{stem}/target.nii.gz" if status == "paired" else None,
        "pairing_status": status,
        "split_hint": "train",
    }


def _pool(unpaired_subjects: tuple[str, ...] = ()) -> list[dict]:
    """Two repeat scans per subject (ses-01 and ses-02), plus targetless records on some."""
    records = []
    for s in SUBJECTS:
        records.append(_rec(s, "01"))
        records.append(_rec(s, "02"))
    for s in unpaired_subjects:
        records.append(_rec(s, "03", status="unpaired_ulf"))
    return records


def _data(*, split: dict, enable_test: bool = False, test_frac: float = 0.0, unpaired_train=False):
    return SimpleNamespace(
        split=DataSplitConfigSchema(**split),
        pairing=DataPairingConfigSchema(unpaired_training=unpaired_train),
        enable_test_split=enable_test,
        test_split=test_frac,
    )


def _subjects_of(items: list[dict]) -> set[str]:
    return {r["subject_id"] for r in items}


def _assert_subject_disjoint(part) -> None:
    train, val, test = (_subjects_of(x) for x in (part.train, part.val, part.test))
    assert not (train & val) and not (train & test) and not (val & test)


def test_random_fraction_keeps_repeat_scans_together():
    data = SimpleNamespace(
        split=DataSplitConfigSchema(type="random", validation_fraction=0.2),
        pairing=DataPairingConfigSchema(),
        enable_test_split=True,
        test_split=0.2,
    )
    part = partition_paired_records(_pool(), data)
    _assert_subject_disjoint(part)
    for subject in SUBJECTS:
        splits = {
            name
            for name, items in (("train", part.train), ("val", part.val), ("test", part.test))
            if any(r["subject_id"] == subject for r in items)
        }
        assert len(splits) == 1, f"subject {subject} straddles {splits}"


def test_loso_subject_holds_out_one_subject_and_test_subjects_are_disjoint():
    data = _data(
        split={"type": "loso_subject", "loso_fold": 3, "test_subjects": ["08", "09"]},
    )
    part = partition_paired_records(_pool(), data)
    assert _subjects_of(part.val) == {"04"}
    assert _subjects_of(part.test) == {"08", "09"}
    _assert_subject_disjoint(part)
    assert len(part.val) == 2  # both repeat scans of the held-out subject


def test_explicit_subject_lists_partition_and_stamp_lists():
    data = _data(split={"type": "subjects", "val_subjects": ["02"], "test_subjects": ["07"]})
    part = partition_paired_records(_pool(), data, manifest_sha="ab" * 32)
    assert part.stamp["subjects"] == {
        "train": sorted(set(SUBJECTS) - {"02", "07"}),
        "val": ["02"],
        "test": ["07"],
    }
    assert part.stamp["records"] == {"train": 16, "val": 2, "test": 2}
    assert part.stamp["manifest_sha256"] == "ab" * 32


def test_unpaired_records_never_enter_val_or_test():
    data = _data(split={"type": "subjects", "val_subjects": ["02"], "test_subjects": ["07"]})
    part = partition_paired_records(_pool(unpaired_subjects=("02", "07")), data)
    assert all(r["pairing_status"] == "paired" for r in part.val + part.test)
    assert not any(r["pairing_status"] != "paired" for r in part.train)


def test_unpaired_enter_train_only_when_declared():
    data = _data(
        split={"type": "subjects", "val_subjects": ["02"]},
        unpaired_train=True,
    )
    part = partition_paired_records(_pool(unpaired_subjects=("03",)), data)
    assert any(r["pairing_status"] == "unpaired_ulf" for r in part.train)


def test_unpaired_sessions_of_held_out_subjects_stay_out_of_train():
    data = _data(
        split={"type": "subjects", "val_subjects": ["02"], "test_subjects": ["07"]},
        unpaired_train=True,
    )
    part = partition_paired_records(_pool(unpaired_subjects=("02", "07", "03")), data)
    _assert_subject_disjoint(part)
    assert {r["subject_id"] for r in part.train if r["pairing_status"] != "paired"} == {"03"}
    assert part.stamp["unpaired_held_out_dropped"] == 2


def test_holdout_site_is_refused_on_paired_route():
    data = _data(split={"type": "random", "validation_fraction": 0.2, "holdout_site": "siteA"})
    with pytest.raises(ValueError, match="loso_subject"):
        partition_paired_records(_pool(), data)


def test_unknown_subject_id_raises():
    data = _data(split={"type": "subjects", "val_subjects": ["99"]})
    with pytest.raises(ValueError, match="absent from the paired manifest"):
        partition_paired_records(_pool(), data)


def test_test_subject_overlapping_loso_holdout_raises():
    data = _data(split={"type": "loso_subject", "holdout_subject": "05", "test_subjects": ["05"]})
    with pytest.raises(ValueError, match="both the validation and a test subject"):
        partition_paired_records(_pool(), data)


def test_declared_test_with_no_paired_test_record_raises():
    data = _data(
        split={"type": "subjects", "val_subjects": ["02"]},
        enable_test=True,
        test_frac=0.2,
    )
    with pytest.raises(ValueError, match="carves no paired test record"):
        partition_paired_records(_pool(), data)


def test_record_without_subject_raises_instead_of_grouping_by_file():
    pool = _pool()
    pool.append({"file_id": "orphan", "primary_path": "/x/orphan.nii", "pairing_status": "paired"})
    data = _data(split={"type": "subjects", "val_subjects": ["02"]})
    with pytest.raises(ValueError, match="carries no subject id"):
        partition_paired_records(pool, data)


def test_is_config_owned_ignores_validation_fraction_on_auto():
    auto = _data(split={"type": "auto", "validation_fraction": 0.2})
    assert not is_config_owned(auto)
    assert is_config_owned(_data(split={"type": "loso_subject", "loso_fold": 0}))
    assert is_config_owned(_data(split={"type": "auto"}, enable_test=True, test_frac=0.1))


def test_resolve_reads_manifest_pool_and_stamps_its_digest(tmp_path):
    manifest = tmp_path / "paired.json"
    manifest.write_text(
        json.dumps(
            {"manifest_version": "5.0", "records": _pool(unpaired_subjects=("03",))},
        )
    )
    data = _data(split={"type": "subjects", "val_subjects": ["02"], "test_subjects": ["07"]})
    part = resolve_paired_partition(manifest, data)
    assert part.stamp["subjects"]["test"] == ["07"]
    assert len(part.stamp["manifest_sha256"]) == 64
    assert partition_leakage_report(part.train, part.val, part.test).status == "clean"


def test_build_partition_stamp_counts_each_split():
    stamp = build_partition_stamp(
        route="split_hint",
        strategy="auto",
        manifest_sha=None,
        train=[{"subject_id": "01"}],
        val=[{"subject_id": "02"}, {"subject_id": "02"}],
        test=[],
    )
    assert stamp["records"] == {"train": 1, "val": 2, "test": 0}
    assert stamp["subjects"]["val"] == ["02"]


def test_holdout_site_on_paired_route_reads_typed_and_raw_blocks() -> None:
    raw = {"source": {"paired_manifest_path": "m.json"}, "split": {"holdout_site": ["0001"]}}
    assert holdout_site_on_paired_route(raw) is True
    assert holdout_site_on_paired_route({**raw, "source": {"index_path": "i.json"}}) is False
    typed = SimpleNamespace(
        source=SimpleNamespace(paired_manifest_path="m.json"),
        split=DataSplitConfigSchema(type="loso", holdout_site="sub-0001"),
    )
    assert holdout_site_on_paired_route(typed) is True
