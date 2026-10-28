"""Unit tests for IndexBuilder train/val split delegation (WS1).

The NIfTI flat-directory split now routes through the shared ``split_index``
SSOT, so its single-file handling matches every other loader: raise on one file
with validation_split>0 (no empty train), and train-only when validation_split
is 0. Volume mode (``variant != '2d_slices'``) never loads voxels before the
split, so these tests need only empty placeholder files.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.data.metadata.index_builder import IndexBuilder


def _touch(dir_path: Path, n: int) -> None:
    for i in range(n):
        (dir_path / f"vol_{i}.nii.gz").write_bytes(b"")


def _config(root: Path, validation_split: float) -> SimpleNamespace:
    # The shared stub, not a bare namespace: every kwarg below is a flat legacy
    # spelling that folded (`data_root` -> `source.root`, `validation_split` ->
    # `split.validation_fraction`, `contrasts` -> `pairing.contrasts`), and
    # DataConfigStub routes them from RENAMES instead of restating the shape.
    return DataConfigStub(
        data_root=str(root),
        data_layout="flat",
        dataset_type="nifti",
        holdout_site=None,
        validation_split=validation_split,
        datasets=None,
        contrasts=None,
        target_contrasts=None,
    )


def test_single_nifti_with_split_raises(tmp_path):
    _touch(tmp_path, 1)
    with pytest.raises(ValueError, match="single file"):
        IndexBuilder.build_nifti_index(_config(tmp_path, 0.2), split="train")


def test_single_nifti_train_only_returns_the_file(tmp_path):
    _touch(tmp_path, 1)
    train = IndexBuilder.build_nifti_index(_config(tmp_path, 0.0), split="train")
    val = IndexBuilder.build_nifti_index(_config(tmp_path, 0.0), split="val")
    assert len(train) == 1
    assert len(val) == 0  # train-only: validation is empty


def test_multi_nifti_split_is_disjoint(tmp_path):
    _touch(tmp_path, 10)
    train = IndexBuilder.build_nifti_index(_config(tmp_path, 0.2), split="train")
    val = IndexBuilder.build_nifti_index(_config(tmp_path, 0.2), split="val")
    train_paths = {r["primary_path"] for r in train}
    val_paths = {r["primary_path"] for r in val}
    assert train_paths.isdisjoint(val_paths)
    assert len(val_paths) == 2 and len(train_paths) == 8


# ---------------------------------------------------------------------------
# Leave-one-SUBJECT-out partitioning (2026-07-26)
# ---------------------------------------------------------------------------
import json as _json  # noqa: E402
import types as _types  # noqa: E402


def _loso_manifest(tmp_path):
    """Three subjects, two contrasts, so the fold index must address the
    subjects that SURVIVE filtering rather than every subject in the file."""
    records = []
    for sub in ("0003", "0001", "0002"):  # deliberately unsorted on disk
        for contrast in ("T1w", "T2w"):
            records.append(
                {
                    "subject_id": sub,
                    "contrast": contrast,
                    "pairing_status": "paired",
                    "split_hint": "train",
                    "primary_path": f"{sub}_{contrast}_ulf.nii.gz",
                    "target_path": f"{sub}_{contrast}_hf.nii.gz",
                    "file_id": f"{sub}_{contrast}",
                }
            )
    # a fourth subject present ONLY in T2w — the trap that shipped an empty
    # validation set when the fold indexed pre-filter subjects
    records.append(
        {
            "subject_id": "0009",
            "contrast": "T2w",
            "pairing_status": "paired",
            "split_hint": "train",
            "primary_path": "0009_T2w_ulf.nii.gz",
            "target_path": "0009_T2w_hf.nii.gz",
            "file_id": "0009_T2w",
        }
    )
    p = tmp_path / "loso.json"
    p.write_text(_json.dumps({"manifest_version": "4.0", "records": records}))
    return str(p)


def _loso_cfg(**kw):
    base = {
        "split_strategy": "loso_subject",
        "loso_fold": None,
        "holdout_subject": None,
        "allow_unpaired": False,
        "contrasts": ["T1w"],
        "target_contrasts": ["T1w"],
        "bidirectional_mode": "ulf_to_hf",
        "hf_resolution": None,
    }
    base.update(kw)
    # Every key here folded into `data.split.*` or `data.pairing.*`; the stub
    # routes them so the reader finds them where it actually looks.
    return DataConfigStub(**base)


def _load(path, split, **kw):
    """The LOSO partition is owned by ``paired_partition``; the reader only pools records."""
    from spectramr.config.schemas.data import DataPairingConfigSchema, DataSplitConfigSchema
    from spectramr.data.metadata.paired_partition import partition_paired_records

    pool = IndexBuilder.load_paired_bids_manifest(path, None, _loso_cfg(**kw))
    data = SimpleNamespace(
        split=DataSplitConfigSchema(
            type="loso_subject",
            loso_fold=kw.get("loso_fold"),
            holdout_subject=kw.get("holdout_subject"),
        ),
        pairing=DataPairingConfigSchema(),
        enable_test_split=False,
        test_split=0.0,
    )
    part = partition_paired_records(pool, data)
    return {"train": part.train, "val": part.val, "test": part.test}[split]


def test_loso_folds_partition_by_subject_without_overlap(tmp_path) -> None:
    """No subject may appear in both splits, and every subject must be held out
    exactly once across the folds."""
    path = _loso_manifest(tmp_path)
    held = set()
    for fold in range(3):  # 3 subjects survive the T1w filter
        train = _load(path, "train", loso_fold=fold)
        val = _load(path, "val", loso_fold=fold)
        tr = {r["subject_id"] for r in train}
        va = {r["subject_id"] for r in val}
        assert len(va) == 1, f"fold {fold} held out {va}"
        assert not (tr & va), f"fold {fold} leaks {tr & va}"
        held |= va
    assert held == {"0001", "0002", "0003"}


def test_fold_indexes_subjects_that_survive_filtering(tmp_path) -> None:
    """The fold list is built AFTER the pairing and contrast filters.

    Computing it before saw every subject in the file — including one present
    only in a contrast the arm filtered out — so folds selected subjects with no
    matching record and the validation set came back EMPTY.
    """
    path = _loso_manifest(tmp_path)
    for fold in range(3):
        assert _load(path, "val", loso_fold=fold), f"fold {fold} has no val records"
    # 0009 exists only in T2w and must not be reachable under a T1w filter
    with pytest.raises(ValueError, match="not among the subjects"):
        _load(path, "val", holdout_subject="0009")


def test_fold_order_is_the_sorted_subject_id_not_manifest_order(tmp_path) -> None:
    """Otherwise the same fold means a different subject after a regeneration."""
    path = _loso_manifest(tmp_path)
    assert {r["subject_id"] for r in _load(path, "val", loso_fold=0)} == {"0001"}


def test_out_of_range_fold_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"folds are 0\.\.2"):
        _load(_loso_manifest(tmp_path), "val", loso_fold=7)


def test_explicit_holdout_subject_is_honoured(tmp_path) -> None:
    path = _loso_manifest(tmp_path)
    assert {r["subject_id"] for r in _load(path, "val", holdout_subject="0002")} == {"0002"}


def test_split_hint_still_drives_the_default_strategy(tmp_path) -> None:
    """loso_subject must not change behaviour for arms that do not ask for it."""
    path = _loso_manifest(tmp_path)
    from spectramr.data.metadata.index_builder import IndexBuilder

    cfg = _loso_cfg(split_strategy="manifest")
    assert IndexBuilder.load_paired_bids_manifest(path, "train", cfg)
    assert not IndexBuilder.load_paired_bids_manifest(path, "val", cfg)


# ---------------------------------------------------------------------------
# manifest_roles admits a subject on what LOADED, not what was PROMISED (B7)
# ---------------------------------------------------------------------------
# ``load_from_manifest_roles``'s H5 branch ``continue``s past a failed read, but
# the has_input/has_target guard was computed from ``role_samples`` -- the
# manifest's promise -- rather than ``subject_dict``, what actually loaded. So a
# manifest naming an input whose file will not open still produced a
# ``tio.Subject``, with no input image in it. The failure then surfaced hours
# later inside the training loop, naming neither the file nor the role. The
# guard also accepted ``has_input or has_target``, admitting a target-only
# subject that can never feed a forward pass.
import logging as _logging  # noqa: E402
import pickle as _pickle  # noqa: E402

from tests.utils.data_config_stub import DataConfigStub  # noqa: E402


def _good_h5(path: Path) -> Path:
    """A loadable H5: complex ``(H, W, D)`` k-space, which
    ``FastMRIH5Strategy.load_torchio_tensor`` stacks to the ``[2, H, W, D]``
    4-D tensor ``tio.ScalarImage`` requires (a 2-D dataset yields 3-D and is
    rejected, which would make every 'good' file a silent second failure)."""
    import h5py
    import numpy as np

    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=np.zeros((4, 4, 2), dtype=np.complex64))
    return path


def _broken_h5(path: Path) -> Path:
    """An ``.h5`` the reader cannot open -- h5py raises on the file signature.
    This is the real defect's shape: the path resolves, the file exists, and
    only the read fails."""
    path.write_bytes(b"not an HDF5 file")
    return path


def _manifest(path: Path, records: list[dict]) -> str:
    with open(path, "wb") as f:
        _pickle.dump(records, f)
    return str(path)


def _roles_config(input_manifest: str, target_manifest: str | None = None):
    """A ``data:`` stand-in for ``load_from_manifest_roles``.

    ``DataConfigStub`` carries every phase-9 sub-block at its REAL schema
    defaults -- the function walks ``config.source.root``, ``config.pairing``
    and ``config.split``, so a flat namespace models a shape no live config
    produces. ``validation_split=0`` selects the explicit train-only path, so
    every surviving subject lands in the train list.
    """
    roles = _types.SimpleNamespace(
        inputs=[{"manifest": input_manifest, "key": "input"}],
        targets=([{"manifest": target_manifest, "key": "target"}] if target_manifest else []),
        auxiliary=[],
    )
    return DataConfigStub(manifest_roles=roles, validation_split=0.0)


def _two_subject_manifests(tmp_path, *, subj_b_input_loads: bool):
    """subjA always loads; subjB's INPUT is the variable. Both carry a good
    target, so a target-only subjB is admitted by the pre-fix guard under
    *both* of its halves -- the strongest available witness.

    File ids avoid the ``_gt``/``_rss``/``_normalized``/``_compressed``/
    ``_reconstructed`` substrings, which the base-id derivation strips with
    ``str.replace`` (anywhere in the name, not just as a suffix).
    """
    b_input = _good_h5 if subj_b_input_loads else _broken_h5
    inputs = _manifest(
        tmp_path / "inputs.pkl",
        [
            {
                "filename": "subjA.h5",
                "path": str(_good_h5(tmp_path / "subjA.h5")),
                "format": "h5",
            },
            {
                "filename": "subjB.h5",
                "path": str(b_input(tmp_path / "subjB.h5")),
                "format": "h5",
            },
        ],
    )
    targets = _manifest(
        tmp_path / "targets.pkl",
        [
            {
                "filename": "subjA_gt.h5",
                "path": str(_good_h5(tmp_path / "subjA_gt.h5")),
                "format": "h5",
            },
            {
                "filename": "subjB_gt.h5",
                "path": str(_good_h5(tmp_path / "subjB_gt.h5")),
                "format": "h5",
            },
        ],
    )
    return _roles_config(inputs, targets)


def test_subject_whose_input_fails_to_load_is_dropped(tmp_path) -> None:
    """Pre-fix, subjB came back as a Subject carrying only its target."""
    config = _two_subject_manifests(tmp_path, subj_b_input_loads=False)
    train, _val = IndexBuilder.load_from_manifest_roles(config, None, None)

    assert [s["file_id"] for s in train] == ["subjA"], (
        "a subject whose input H5 failed to open was admitted anyway"
    )
    # The defect was never 'subject missing' -- it was 'subject present with no
    # input image'. Assert the survivor actually carries one, so the guard
    # cannot be satisfied by dropping everything.
    assert "input" in train[0], "the surviving subject has no input image"


def test_dropped_subject_is_named_in_a_counted_warning(tmp_path, caplog) -> None:
    """The census must name the count AND the file, or the operator learns
    only that training failed, not which record to fix."""
    config = _two_subject_manifests(tmp_path, subj_b_input_loads=False)
    with caplog.at_level(_logging.WARNING):
        IndexBuilder.load_from_manifest_roles(config, None, None)

    assert "Dropped 1 of 2" in caplog.text
    assert "subjB" in caplog.text
    assert "input failed to load" in caplog.text


def test_input_without_target_is_kept_but_warned(tmp_path, caplog) -> None:
    """An input with no target still feeds a forward pass (unpaired /
    self-supervised arms are legitimate), so it must NOT be dropped -- but it
    cannot supervise a paired objective, so it must be counted."""
    inputs = _manifest(
        tmp_path / "inputs.pkl",
        [
            {
                "filename": "subjA.h5",
                "path": str(_good_h5(tmp_path / "subjA.h5")),
                "format": "h5",
            }
        ],
    )
    config = _roles_config(inputs)  # no target manifest at all

    with caplog.at_level(_logging.WARNING):
        train, _val = IndexBuilder.load_from_manifest_roles(config, None, None)

    assert [s["file_id"] for s in train] == ["subjA"]
    assert "input but no target" in caplog.text
    assert "subjA" in caplog.text


def test_every_input_failing_raises_with_read_and_dropped_counts(tmp_path) -> None:
    """The bare 'No valid subjects' message told the operator nothing about
    whether the manifest was empty or every file failed to open."""
    inputs = _manifest(
        tmp_path / "inputs.pkl",
        [
            {
                "filename": f"subj{n}.h5",
                "path": str(_broken_h5(tmp_path / f"subj{n}.h5")),
                "format": "h5",
            }
            for n in ("A", "B")
        ],
    )
    targets = _manifest(
        tmp_path / "targets.pkl",
        [
            {
                "filename": f"subj{n}_gt.h5",
                "path": str(_good_h5(tmp_path / f"subj{n}_gt.h5")),
                "format": "h5",
            }
            for n in ("A", "B")
        ],
    )
    config = _roles_config(inputs, targets)

    with pytest.raises(ValueError) as excinfo:
        IndexBuilder.load_from_manifest_roles(config, None, None)

    message = str(excinfo.value)
    assert "2 manifest record(s) were read" in message
    assert "2 were dropped" in message


def test_loadable_input_and_target_subject_is_returned(tmp_path) -> None:
    """Guard against over-tightening: the happy path must still build the
    Subject, with both images mounted."""
    config = _two_subject_manifests(tmp_path, subj_b_input_loads=True)
    train, _val = IndexBuilder.load_from_manifest_roles(config, None, None)

    assert sorted(s["file_id"] for s in train) == ["subjA", "subjB"]
    for subject in train:
        assert "input" in subject and "target" in subject


# ---------------------------------------------------------------------------
# Subject-grouped directory splits (cohort review 2026-09-02, T0.2)
#
# Planted violation: with BIDS-style names, 3 subjects x 2 contrasts and a
# 50 % hold-out, the old file-level split put sub-02's T1w in train and its
# T2w in val. Each directory route must now keep a subject whole and stamp
# ``subject_id`` on its records so ``split_leakage`` can see it.
# ---------------------------------------------------------------------------


def _bids_files(root: Path, subjects=(1, 2, 3), contrasts=("T1w", "T2w")) -> list[Path]:
    made = []
    for s in subjects:
        for c in contrasts:
            p = root / f"sub-{s:02d}_{c}.nii.gz"
            p.write_bytes(b"")
            made.append(p)
    return made


def _subjects(records: list[dict]) -> set[str]:
    return {r["subject_id"] for r in records}


def test_flat_nifti_split_keeps_a_subject_on_one_side(tmp_path):
    _bids_files(tmp_path)
    cfg = _config(tmp_path, 0.5)
    train = IndexBuilder.build_nifti_index(cfg, split="train")
    val = IndexBuilder.build_nifti_index(cfg, split="val")
    assert all("subject_id" in r for r in train + val)
    assert not (_subjects(train) & _subjects(val)), "a subject straddles train/val"
    assert len(train) + len(val) == 6


def test_flat_nifti_without_labels_is_the_file_level_split(tmp_path):
    """No ``sub-`` label -> every file is its own group -> the pre-review split."""
    _touch(tmp_path, 10)
    cfg = _config(tmp_path, 0.2)
    val = IndexBuilder.build_nifti_index(cfg, split="val")
    assert len(val) == 2
    assert all("subject_id" not in r for r in val)


def test_bids_layout_split_keeps_a_subject_on_one_side(tmp_path):
    for s in (1, 2, 3):
        anat = tmp_path / f"sub-{s:02d}" / "anat"
        anat.mkdir(parents=True)
        for c in ("T1w", "T2w"):
            (anat / f"sub-{s:02d}_{c}.nii.gz").write_bytes(b"")
    cfg = DataConfigStub(
        data_root=str(tmp_path),
        data_layout="bids",
        dataset_type="nifti",
        holdout_site=None,
        validation_split=0.5,
        datasets=None,
        contrasts=None,
        target_contrasts=None,
    )
    train = IndexBuilder.build_nifti_index(cfg, split="train")
    val = IndexBuilder.build_nifti_index(cfg, split="val")
    assert len(train) + len(val) == 6
    assert not (_subjects(train) & _subjects(val))
    assert _subjects(val) == {"sub-02", "sub-03"} or _subjects(val) == {"sub-03"}


def test_paired_nifti_split_keeps_a_subject_on_one_side(tmp_path):
    src, tgt = tmp_path / "source", tmp_path / "target"
    src.mkdir()
    tgt.mkdir()
    for s in (1, 2, 3):
        for c in ("T1w", "T2w"):
            (src / f"sub-{s:02d}_{c}.nii.gz").write_bytes(b"")
            (tgt / f"sub-{s:02d}_{c}.nii.gz").write_bytes(b"")
    cfg = DataConfigStub(
        data_root=str(tmp_path),
        data_layout="flat",
        dataset_type="nifti_paired",
        holdout_site=None,
        validation_split=0.5,
        datasets=None,
        contrasts=None,
        target_contrasts=None,
    )
    train = IndexBuilder.build_nifti_index(cfg, split="train")
    val = IndexBuilder.build_nifti_index(cfg, split="val")
    assert len(train) + len(val) == 6
    assert all(r.get("target_path") for r in train + val)
    assert not (_subjects(train) & _subjects(val))
