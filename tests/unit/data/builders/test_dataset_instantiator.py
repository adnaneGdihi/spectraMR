"""Unit tests for DatasetInstantiator dispatch.

Regression cover for BD-003: an unknown ``dataset_type`` used to fall through
every named dispatch branch (m4raw, contrast_aware_paired, nifti/*,
preprocessed, synthetic, ..., image/folder) and silently land on the
FastMRI/Universal K-Space loader (step 9). A YAML typo or an unregistered new
type therefore produced wrong data loading with no error — a NN#3 / pitfall #9
silent fallback. The fix guards step 9 with an explicit ``_KSPACE_TYPES`` set
and raises ``ValueError`` for anything else.
"""

from types import SimpleNamespace

import pytest

from spectramr.config.schemas.data import DataConfigSchema as _DataSchema
from spectramr.data.builders.dataset_instantiator import DatasetInstantiator
from tests.utils.data_config_stub import DataConfigStub as _Cfg


def _stub_config(dataset_type: str) -> SimpleNamespace:
    """Minimal config stub touching only the attributes ``create_datasets``
    reads before reaching the step-9 dispatch guard."""
    return _Cfg(
        dataset_type=dataset_type,
        # _has_valid_manifest_roles → False (no manifest_roles attr present)
        index_path=None,
    )


def test_create_datasets_unknown_type_raises() -> None:
    """An unrecognised dataset_type must raise ValueError with the offending
    type name in the message — never silently fall through to FastMRI."""
    config = _stub_config("unknown_xyz")
    with pytest.raises(ValueError, match="unknown_xyz"):
        DatasetInstantiator.create_datasets(
            config,
            train_index=[],
            val_index=[],
            train_transforms=None,
            val_transforms=None,
        )


def test_create_datasets_unknown_type_message_is_actionable() -> None:
    """The raised message must enumerate known types so a misconfigured YAML
    is fixable from the error alone."""
    config = _stub_config("kSpace")  # casing typo of a known type
    with pytest.raises(ValueError, match="is not recognised"):
        DatasetInstantiator.create_datasets(
            config,
            train_index=[],
            val_index=[],
            train_transforms=None,
            val_transforms=None,
        )


# ── target_mode threading (M4Raw NEX averaging) ───────────────────────────────


class _StubDS:
    def __len__(self) -> int:
        return 0


def _m4raw_cfg(**overrides) -> SimpleNamespace:
    # The two NEX defaults are sourced FROM the schema, never restated. A stub
    # that omits a field the schema declares is the `SimpleNamespace` trap that
    # kept `rule_spatial_rank` and the SFC wrapper green while both were dead
    # (PR #644): the stub agrees with whatever the reader asks for, so it can
    # certify a branch a real config can never take.
    base = {
        "kspace_percentile": 0.99,
        "normalize_kspace": True,
        "single_contrast": True,
        "log_scaling": False,
        "validation_split": 0,
        "target_mode": _DataSchema.model_fields["target_mode"].default,
        "val_target_mode": _DataSchema.model_fields["val_target_mode"].default,
        "r2r_alpha": _DataSchema.model_fields["r2r_alpha"].default,
        "nex_target_exclude_input": _DataSchema.model_fields["nex_target_exclude_input"].default,
        "nex_fallback": _DataSchema.model_fields["nex_fallback"].default,
        "use_repetitions": _DataSchema.model_fields["use_repetitions"].default,
        "slice_level_records": _DataSchema.model_fields["slice_level_records"].default,
    }
    base.update(overrides)
    return _Cfg(**base)


def _capture_m4raw(monkeypatch) -> dict:
    import spectramr.data.datasets.m4raw_dataset as m4raw_mod

    cap: dict = {}
    monkeypatch.setattr(
        m4raw_mod,
        "M4RawRepetitionDataset",
        lambda _h5, **kw: (cap.update(kw), _StubDS())[1],
    )
    return cap


def test_m4raw_threads_target_mode(monkeypatch) -> None:
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(target_mode="phase_aligned_mean"),
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert cap["target_mode"] == "phase_aligned_mean"


def test_m4raw_threads_the_schema_default_when_the_arm_is_silent(monkeypatch) -> None:
    """An arm that declares nothing gets the SCHEMA default, read from the schema.

    This replaces a test that passed a stub with no ``target_mode`` attribute at
    all and asserted the reader's ``getattr(..., "complex_mean")`` fallback. That
    fallback is unreachable in production -- ``target_mode`` is a declared field,
    so every real ``DataConfigSchema`` has it -- which made the test pin a branch
    only a test double could take. Worse, the fallback restated ``complex_mean``
    as a literal, so a future schema default change would have left the reader
    silently disagreeing with the schema.
    """
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(),  # declares neither NEX knob -> schema defaults
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert cap["target_mode"] == _DataSchema.model_fields["target_mode"].default
    assert cap["target_mode"] == "complex_mean"  # anti-vacuity: pin today's value


def test_m4raw_threads_nex_target_exclude_input(monkeypatch) -> None:
    """The leave-one-out NEX flag reaches the dataset, not just the schema."""
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(nex_target_exclude_input=True),
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert cap["nex_target_exclude_input"] is True


def test_m4raw_threads_nex_fallback(monkeypatch) -> None:
    """The leave-one-out fallback policy reaches the dataset (cohort review T0.1)."""
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(nex_target_exclude_input=True, nex_fallback="all_reps"),
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert cap["nex_fallback"] == "all_reps"


def test_m4raw_threads_slice_level_records(monkeypatch) -> None:
    """The slice-level index knob reaches the dataset (#1757)."""
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(slice_level_records=True),
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert cap["slice_level_records"] is True


def test_m4raw_slice_level_records_defaults_off_from_the_schema(monkeypatch) -> None:
    """An arm that says nothing keeps the per-group index: the value threaded
    is the schema default, read from the schema."""
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(),
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert cap["slice_level_records"] == _DataSchema.model_fields["slice_level_records"].default
    assert cap["slice_level_records"] is False  # anti-vacuity: pin today's value


def test_m4raw_use_repetitions_none_means_the_route_default_true(monkeypatch) -> None:
    """The schema default is None ("the route decides"); on the m4raw route that
    is True. This pins the zero-behaviour-change contract for the 46 corpus arms
    that never declare the knob: before the wiring the literal was ``True``."""
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(),  # use_repetitions absent -> schema default None
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert _DataSchema.model_fields["use_repetitions"].default is None  # anti-vacuity
    assert cap["use_repetitions"] is True


@pytest.mark.parametrize("declared", [True, False])
def test_m4raw_honours_an_explicit_use_repetitions(monkeypatch, declared) -> None:
    """An explicit value is read, not overwritten by the old literal (#668)."""
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(use_repetitions=declared),
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert cap["use_repetitions"] is declared


@pytest.mark.parametrize(
    "missing",
    [
        "target_mode",
        "val_target_mode",
        "r2r_alpha",
        "nex_target_exclude_input",
        "nex_fallback",
        "use_repetitions",
    ],
)
def test_m4raw_nex_knob_missing_from_receiver_fails_loud(monkeypatch, missing) -> None:
    """A receiver without the field must RAISE, never resolve to a default.

    The reader used to be ``getattr(config, "target_mode", "complex_mean")``. If a
    later phase decomposes these leaves into a sub-block (``data.nex.*``), that
    form hands back ``complex_mean`` -- which ``.claude/rules/data.md`` documents
    as destructive for M4Raw, because complex-averaging phase-incoherent
    repetitions cancels signal rather than averaging it. The run would train, and
    smoke-pass, against a corrupted reference. This is the same silent-reader
    shape that killed ``rule_spatial_rank`` and the SFC wrapper on this branch;
    the guard is that the attribute read now raises instead.
    """
    _capture_m4raw(monkeypatch)
    cfg = _m4raw_cfg()
    delattr(cfg, missing)
    with pytest.raises(AttributeError, match=missing):
        DatasetInstantiator._create_m4raw_repetition(
            cfg,
            [{"primary_path": "/x/a.h5"}],
            [{"primary_path": "/x/b.h5"}],
            None,
            None,
        )


def test_m4raw_threads_r2r_alpha(monkeypatch) -> None:
    """``data.r2r_alpha`` must reach the dataset, and reach BOTH splits.

    Nothing downstream raises on a wrong alpha -- it silently shifts noise
    between the two R2R halves -- so a dropped forward is invisible at runtime.
    """
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(
            target_mode="r2r", r2r_alpha=0.25, val_target_mode="phase_aligned_mean"
        ),
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    # ``cap`` holds the LAST construction, i.e. the val dataset -- so this also
    # pins that val is not left on the default while train gets the declared
    # value (they are two separate call sites). alpha forwards to BOTH splits
    # even though only the training split reads it, because the two datasets are
    # otherwise built from identical kwargs; the target MODE is the one thing
    # that differs.
    assert cap["r2r_alpha"] == 0.25
    assert cap["target_mode"] == "phase_aligned_mean"


def test_m4raw_threads_the_r2r_alpha_default_when_the_arm_is_silent(monkeypatch) -> None:
    cap = _capture_m4raw(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(),
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert cap["r2r_alpha"] == _DataSchema.model_fields["r2r_alpha"].default
    assert cap["r2r_alpha"] == 1.0  # anti-vacuity: pin today's value


def _capture_m4raw_both(monkeypatch) -> list[dict]:
    """Record BOTH constructions, in order: train first, val second.

    ``_capture_m4raw`` keeps only the last kwargs, so it cannot see the two
    splits diverging -- which is exactly what ``val_target_mode`` does.
    """
    import spectramr.data.datasets.m4raw_dataset as m4raw_mod

    calls: list[dict] = []
    monkeypatch.setattr(
        m4raw_mod,
        "M4RawRepetitionDataset",
        lambda _h5, **kw: (calls.append(kw), _StubDS())[1],
    )
    return calls


def test_val_target_mode_reaches_the_val_split_only(monkeypatch) -> None:
    """The training split trains on r2r; the validation split scores against NEX.

    This is the whole point of the knob. An r2r validation target is redrawn
    every ``__getitem__``, so validating against it scores the noise draw and
    ``track_best_metric`` keeps the luckiest epoch. If this forward were dropped
    the run would still train, still log a falling loss, and still write
    checkpoints -- picked on noise. Nothing downstream raises.
    """
    calls = _capture_m4raw_both(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(target_mode="r2r", val_target_mode="phase_aligned_mean"),
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert len(calls) == 2, "expected one train and one val construction"
    train_kw, val_kw = calls
    assert train_kw["target_mode"] == "r2r"
    assert val_kw["target_mode"] == "phase_aligned_mean"


def test_val_target_mode_none_serves_the_training_mode_to_both_splits(monkeypatch) -> None:
    """The default must not silently change what every existing arm validates on.

    111 M4Raw arms declare ``target_mode`` and none declares the new knob, so
    ``None`` has to mean exactly "what it did before".
    """
    calls = _capture_m4raw_both(monkeypatch)
    DatasetInstantiator._create_m4raw_repetition(
        _m4raw_cfg(target_mode="phase_aligned_mean"),  # val_target_mode -> schema default
        [{"primary_path": "/x/a.h5"}],
        [{"primary_path": "/x/b.h5"}],
        None,
        None,
    )
    assert len(calls) == 2
    train_kw, val_kw = calls
    assert train_kw["target_mode"] == "phase_aligned_mean"
    assert val_kw["target_mode"] == "phase_aligned_mean"
    assert _DataSchema.model_fields["val_target_mode"].default is None  # anti-vacuity


def test_the_m4raw_stub_declares_every_nex_field_the_schema_declares() -> None:
    """The stub must not omit a field a real config always carries (anti-F16).

    Without this, ``_m4raw_cfg`` could drift back to omitting a NEX knob and the
    threading tests above would certify a code path production cannot reach.
    """
    cfg = _m4raw_cfg()
    for field in ("target_mode", "val_target_mode", "r2r_alpha", "nex_target_exclude_input"):
        assert field in _DataSchema.model_fields, f"{field} left the schema"
        assert hasattr(cfg, field), (
            f"the m4raw stub omits `{field}`, which DataConfigSchema declares -- "
            "a stub that is missing a real field lets a dead branch test green"
        )


# ── spec E2: bart_kspace dispatch ─────────────────────────────────────────────


def _write_bart(directory, name, arr):
    import numpy as np

    dims = list(arr.shape)
    (directory / f"{name}.hdr").write_text("# Dimensions\n" + " ".join(str(d) for d in dims) + "\n")
    arr.astype(np.complex64).flatten(order="F").tofile(directory / f"{name}.cfl")


def test_bart_kspace_requires_enabled() -> None:
    """dataset_type='bart_kspace' with data.bart.enabled=False must raise."""
    config = _Cfg(
        dataset_type="bart_kspace",
        index_path=None,
        bart=SimpleNamespace(enabled=False),
    )
    with pytest.raises(ValueError, match=r"data\.bart\.enabled"):
        DatasetInstantiator.create_datasets(
            config,
            train_index=[],
            val_index=[],
            train_transforms=None,
            val_transforms=None,
        )


def test_bart_kspace_builds_train_val_datasets(tmp_path) -> None:
    """A valid bart config + on-disk .cfl/.hdr yields two BartKspaceDataset."""
    import numpy as np

    from spectramr.config.schemas.data import BartConfigSchema
    from spectramr.data.datasets.bart_dataset import BartKspaceDataset

    for name in ("a", "b"):
        _write_bart(tmp_path, name, np.ones((4, 4, 1), dtype=np.complex64))

    config = _Cfg(
        dataset_type="bart_kspace",
        index_path=None,
        data_root=str(tmp_path),
        split_strategy="random",
        validation_split=0.5,
        bart=BartConfigSchema(
            enabled=True,
            bart_dim_map={"readout": 0, "phase": 1, "coil": 2},
            sampling="cartesian",
        ),
    )
    train_ds, val_ds = DatasetInstantiator.create_datasets(
        config,
        train_index=[],
        val_index=[],
        train_transforms=None,
        val_transforms=None,
    )
    assert isinstance(train_ds, BartKspaceDataset)
    assert isinstance(val_ds, BartKspaceDataset)
    assert len(train_ds) >= 1 and len(val_ds) >= 1


def test_bart_kspace_manifest_branch_honors_file_pattern(tmp_path) -> None:
    """F3: ``dataset_type='bart_kspace'`` with ``index_path`` + ``bart.file_pattern``
    loads ONLY the matching arrays from the manifest, not the radial siblings."""
    import json
    from pathlib import Path

    import numpy as np

    from spectramr.config.schemas.data import BartConfigSchema

    root = tmp_path / "raw"
    root.mkdir()
    for name in ("scan_05_kspace", "scan_05_b1map", "scan_06_b1map"):
        _write_bart(root, name, np.ones((4, 4, 1), dtype=np.complex64))

    mf = tmp_path / "demo.json"
    mf.write_text(
        json.dumps(
            {
                "manifest_version": "3.0",
                "dataset_name": "demo",
                "data_root": str(root),
                "file_type": "bart",
                "total_records": 3,
                "records": [
                    {"relative_path": "scan_05_kspace.cfl"},
                    {"relative_path": "scan_05_b1map.cfl"},
                    {"relative_path": "scan_06_b1map.cfl"},
                ],
                "status": "downloaded",
            }
        )
    )

    config = _Cfg(
        dataset_type="bart_kspace",
        index_path=str(mf),
        data_root=str(root),
        split_strategy="random",
        validation_split=0.5,
        bart=BartConfigSchema(
            enabled=True,
            bart_dim_map={"readout": 0, "phase": 1, "coil": 2},
            sampling="cartesian",
            file_pattern="b1map",
        ),
    )
    train_ds, val_ds = DatasetInstantiator.create_datasets(
        config,
        train_index=[],
        val_index=[],
        train_transforms=None,
        val_transforms=None,
    )
    # index records are dicts {"path", "shape"} (F5); read the path back
    combined = [r["path"] for r in (list(train_ds.index) + list(val_ds.index))]
    assert combined, "filtered index is empty"
    assert all("b1map" in Path(p).stem for p in combined)
    assert all("kspace" not in Path(p).stem for p in combined)


# ── spec E5: bids_paired dispatch ─────────────────────────────────────────────


def test_bids_paired_requires_enabled() -> None:
    """dataset_type='bids_paired' with data.bids_paired.enabled=False must raise."""
    config = _Cfg(
        dataset_type="bids_paired",
        index_path=None,
        bids_paired=SimpleNamespace(enabled=False),
    )
    with pytest.raises(ValueError, match=r"data\.bids_paired\.enabled"):
        DatasetInstantiator.create_datasets(
            config,
            train_index=[],
            val_index=[],
            train_transforms=None,
            val_transforms=None,
        )


def test_png_paired_requires_enabled() -> None:
    """dataset_type='png_paired' with data.png_paired.enabled=False must raise."""
    config = _Cfg(
        dataset_type="png_paired",
        num_virtual_coils=4,
        index_path=None,
        png_paired=SimpleNamespace(enabled=False),
    )
    with pytest.raises(ValueError, match=r"data\.png_paired\.enabled"):
        DatasetInstantiator.create_datasets(
            config,
            train_index=[],
            val_index=[],
            train_transforms=None,
            val_transforms=None,
        )


def test_regroup_mrixfields_multi_source_is_group_aware_and_shares_n() -> None:
    # REGRESSION (B-1.1): the upstream flat RECORD split shatters a subject's field group
    # across train/val (each side then lacks complete tuples -> raises). The mrixfields
    # multi_source path re-splits on WHOLE pairing groups and derives ONE shared source-field
    # set so train/val have identical consensus arity N (and no subject leaks across splits).
    fields = [0.1, 1.5, 3.0, 5.0, 7.0]
    full = [
        {
            "subject_id": s,
            "contrast": "T1w",
            "pairing_group": f"{s}|T1w",
            "primary_path": f"{s}_{f}",
            "field_strength": f,
        }
        for s in ("s1", "s2", "s3", "s4")
        for f in fields
    ]
    # simulate the upstream flat record slice (which would split mid-group)
    split = int(0.8 * len(full))
    train_idx, val_idx = full[:split], full[split:]
    tr, va, src_fields = DatasetInstantiator._regroup_mrixfields_multi_source(
        train_idx, val_idx, target=7.0, val_split=0.25
    )
    assert src_fields == [0.1, 1.5, 3.0, 5.0]  # shared set from the FULL index
    # whole groups land on one side only (no subject leakage)
    tr_subj = {r["subject_id"] for r in tr}
    va_subj = {r["subject_id"] for r in va}
    assert tr_subj and va_subj and not (tr_subj & va_subj)
    # each split holds complete 5-field groups -> all records present per subject
    assert len(tr) == len(tr_subj) * 5 and len(va) == len(va_subj) * 5


def test_create_mrixfields_ulf_source_regroups_so_val_keeps_ulf_field() -> None:
    """REGRESSION (2026-06-22): ``ulf_source`` used the raw upstream flat split.

    On the field-SORTED ``mrixfields2026_train.json`` (all 0.1 T first ... 7 T last) a
    contiguous 90/10 slice strands EVERY 0.1 T source in train, so validation has no
    0.1 T -> ``MRIxFieldsPairedDataset`` raises ``produced 0 pairs ... fields present=[5,7]``.
    The group-aware re-split (previously applied only to ``multi_source``) must cover
    ``ulf_source`` too, so each split keeps COMPLETE field groups (incl. 0.1 T) and
    pairing succeeds. Pre-fix this call raised; post-fix val holds real ULF->HF pairs.
    """
    fields = [0.1, 1.5, 3.0, 5.0, 7.0]
    # field-SORTED full index, mirroring the real manifest record ordering
    full = [
        {
            "subject_id": s,
            "contrast": "T1w",
            "pairing_group": f"{s}|T1w",
            "primary_path": f"{s}_{f}",
            "field_strength": f,
        }
        for f in fields
        for s in ("s1", "s2", "s3", "s4")
    ]
    split = int(0.8 * len(full))
    train_idx, val_idx = full[:split], full[split:]
    # the flat val slice is field-degenerate: only the top fields, no 0.1 T source
    assert 0.1 not in {r["field_strength"] for r in val_idx}
    cfg = _Cfg(
        dataset_type="mrixfields",
        mrixfields_pairing_policy="ulf_source",
        mrixfields_target_field=None,
        expose_field_strength_target=True,
        # This test exercises group-aware RE-SPLITTING, not slicing. central keeps
        # construction lazy so the fake paths are never loaded; all_slices (the default)
        # would foreground-scan every volume at construction and hit the missing files.
        mrixfields_slice_mode="central",
        validation_split=0.25,
    )
    train_ds, val_ds = DatasetInstantiator._create_mrixfields(cfg, train_idx, val_idx, None, None)
    # group-aware re-split rescued val: complete groups incl. 0.1 T -> ulf_source pairs
    assert len(val_ds) > 0, "val regrouped to complete groups must yield ULF->HF pairs"
    assert len(train_ds) > 0


def test_create_mrixfields_fixed_target_regroups_val() -> None:
    """``fixed_target`` is field-pinned too: the group-aware re-split must keep the
    pinned target field present in BOTH splits (same stranding hazard as ulf_source)."""
    fields = [0.1, 1.5, 3.0, 5.0, 7.0]
    full = [
        {
            "subject_id": s,
            "contrast": "T1w",
            "pairing_group": f"{s}|T1w",
            "primary_path": f"{s}_{f}",
            "field_strength": f,
        }
        for f in fields
        for s in ("s1", "s2", "s3", "s4")
    ]
    split = int(0.8 * len(full))
    train_idx, val_idx = full[:split], full[split:]
    cfg = _Cfg(
        dataset_type="mrixfields",
        mrixfields_pairing_policy="fixed_target",
        mrixfields_target_field=7.0,
        expose_field_strength_target=True,
        # Tests regrouping, not slicing; central keeps construction lazy (see the
        # ulf_source test above) so the fake paths are never loaded.
        mrixfields_slice_mode="central",
        validation_split=0.25,
    )
    train_ds, val_ds = DatasetInstantiator._create_mrixfields(cfg, train_idx, val_idx, None, None)
    assert len(val_ds) > 0 and len(train_ds) > 0


# ──────────────────────────────────────────────────────────────────────
# bidirectional_mode for paired NIfTI (2026-06-22 inert-knob fix)
# ──────────────────────────────────────────────────────────────────────
#
# The paired manifest stores primary_path=ULF, target_path=HF. ``hf_to_ulf`` is
# a genuine HF→ULF translation direction (swap both arms); ``hf_to_hf`` is the
# stage-1 autoencode mode (drop the ULF arm so target≡input=HF). That knob was
# only honored by ``contrast_aware_paired``; ``nifti_paired`` ignored it, so a
# stage-1 VAE silently trained ULF→HF. These pin the swap + autoencode rewrites.


def test_swap_paired_arms_swaps_primary_and_target() -> None:
    idx = [
        {
            "primary_path": "/d/sub_ulf.nii",
            "target_path": "/d/sub_hf.nii",
            "file_id": "sub",
        }
    ]
    out = DatasetInstantiator._swap_paired_arms(idx)
    assert out[0]["primary_path"] == "/d/sub_hf.nii"  # input arm ← HF
    assert out[0]["target_path"] == "/d/sub_ulf.nii"  # target arm ← ULF
    # original is not mutated (shallow copy)
    assert idx[0]["primary_path"] == "/d/sub_ulf.nii"


def test_swap_paired_arms_swaps_field_and_contrast_labels() -> None:
    idx = [
        {
            "primary_path": "u",
            "target_path": "h",
            "input_field": 0.064,
            "target_field": 3.0,
            "input_contrast": "ulf_FLAIR",
            "target_contrast": "hf_FLAIR",
        }
    ]
    out = DatasetInstantiator._swap_paired_arms(idx)
    assert out[0]["input_field"] == 3.0 and out[0]["target_field"] == 0.064
    assert out[0]["input_contrast"] == "hf_FLAIR"
    assert out[0]["target_contrast"] == "ulf_FLAIR"


def test_swap_paired_arms_missing_target_raises() -> None:
    """Fail loud (#9/#15) rather than silently no-op the knob."""
    with pytest.raises(ValueError, match="requires every paired record"):
        DatasetInstantiator._swap_paired_arms([{"primary_path": "u", "file_id": "x"}])


def _capture_universal(monkeypatch):
    """Patch UniversalMRIDataset in the instantiator module to capture the
    index it is handed (avoids real NIfTI IO)."""
    captured = {}

    class _Capture:
        def __init__(self, index, **kwargs):
            captured.setdefault("indices", []).append(index)
            captured.setdefault("kwargs", []).append(kwargs)

    import spectramr.data.builders.dataset_instantiator as di

    monkeypatch.setattr(di, "UniversalMRIDataset", _Capture)
    return captured


# ── multi_contrast → contrast_idx wiring (de-facade the FiLM path, 2026-07-08) ─
# The nifti_paired ULF→HF arms carry a per-arm data.multi_contrast.contrast_map,
# but the instantiator never threaded it into UniversalMRIDataset — so no
# contrast_idx was emitted and every FiLM / contrast-guidance arm silently
# conditioned on contrast 0 (pitfall #16). These pin the wiring: an enabled block
# forwards the map; a disabled / absent block forwards None (contrast-agnostic
# arms — ulf_physics, unpaired_ulf — are untouched, no crash on their T1w/ADC).


def test_create_nifti_universal_threads_contrast_map_when_enabled(monkeypatch):
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(
        dataset_type="nifti_paired",
        multi_contrast=SimpleNamespace(enabled=True, contrast_map={"T1w": 0, "T2w": 1, "FLAIR": 2}),
    )
    idx = [
        {
            "primary_path": "/d/ulf.nii",
            "target_path": "/d/hf.nii",
            "file_id": "s",
            "contrast": "T2w",
        }
    ]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    for kw in captured["kwargs"]:
        assert kw["contrast_map"] == {"T1w": 0, "T2w": 1, "FLAIR": 2}


def test_create_nifti_universal_no_contrast_map_when_disabled(monkeypatch):
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(
        dataset_type="nifti_paired",
        multi_contrast=SimpleNamespace(
            enabled=False, contrast_map={"T1w": 0, "T2w": 1, "FLAIR": 2}
        ),
    )
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    for kw in captured["kwargs"]:
        assert kw.get("contrast_map") is None


def test_create_nifti_universal_no_contrast_map_when_block_absent(monkeypatch):
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(dataset_type="nifti_paired")  # no multi_contrast at all
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    for kw in captured["kwargs"]:
        assert kw.get("contrast_map") is None


# ── mrixfields_target_field → field_strength_target wiring (unblock the ─────────
# field-conditioned ULF→HF task2 strategies on the paired-ULF dataset, 2026-07-08).
# ulf_dps / monotone_field / field_conditioned_inr / generative_refiner /
# ulf_redegrad_tta read batch['field_strength_target'] directly and RAISE on its
# absence; nifti_paired never emitted it. The paired ULF task is a fixed 64mT→3T
# translation, so the target field is the config constant data.mrixfields_target_field.
# These pin the wiring: value forwarded when the knob is set + exposed; None (no
# stamp) when the knob is unset (field-agnostic arms) or exposure is off.


def test_create_nifti_universal_threads_target_field_when_set(monkeypatch):
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(
        dataset_type="nifti_paired",
        mrixfields_target_field=3.0,
        expose_field_strength_target=True,
    )
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    for kw in captured["kwargs"]:
        assert kw["target_field_strength"] == 3.0


def test_create_nifti_universal_no_target_field_when_exposure_off(monkeypatch):
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(
        dataset_type="nifti_paired",
        mrixfields_target_field=3.0,
        expose_field_strength_target=False,
    )
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    for kw in captured["kwargs"]:
        assert kw.get("target_field_strength") is None


def test_create_nifti_universal_no_target_field_when_knob_absent(monkeypatch):
    # Field-agnostic arms (geomamba baselines, unpaired_ulf) leave the knob unset:
    # default expose=True but no value → no stamp, so they stay untouched.
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(dataset_type="nifti_paired")
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    for kw in captured["kwargs"]:
        assert kw.get("target_field_strength") is None


def test_create_nifti_universal_hf_to_ulf_routes_hf_to_input(monkeypatch) -> None:
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(dataset_type="nifti_paired", bidirectional_mode="hf_to_ulf")
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    # both train and val datasets were handed the swapped index
    for got in captured["indices"]:
        assert got[0]["primary_path"] == "/d/hf.nii"  # input ← HF


def test_create_nifti_universal_default_ulf_to_hf_does_not_swap(monkeypatch) -> None:
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(dataset_type="nifti_paired")  # no bidirectional_mode → ulf_to_hf
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    assert captured["indices"][0][0]["primary_path"] == "/d/ulf.nii"  # unchanged


def test_create_nifti_universal_unknown_mode_raises(monkeypatch) -> None:
    _capture_universal(monkeypatch)
    cfg = _Cfg(dataset_type="nifti_paired", bidirectional_mode="hf_to_ulf_wrong")
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    with pytest.raises(ValueError, match="not recognised"):
        DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)


# ── hf_to_hf / ulf_to_ulf single-field autoencode (2026-07 ldm triage) ─────────
# A stage-1 VAE must reconstruct ONE field; a translation direction would train
# a degradation net that corrupts the frozen stage-2 latent. These modes drop the
# opposite arm (target_path=None → self-supervised branch aliases target=input).


def test_autoencode_field_target_arm_promotes_hf_and_drops_target() -> None:
    idx = [
        {
            "primary_path": "/d/ulf.nii",
            "target_path": "/d/hf.nii",
            "file_id": "s",
            "input_field": 0.064,
            "target_field": 3.0,
        }
    ]
    out = DatasetInstantiator._autoencode_field(idx, arm="target")
    assert out[0]["primary_path"] == "/d/hf.nii"  # input ← HF
    assert out[0]["target_path"] is None  # ULF arm dropped
    assert out[0]["input_field"] == 3.0  # provenance: input is HF
    assert "target_field" not in out[0]
    assert idx[0]["primary_path"] == "/d/ulf.nii"  # original not mutated


def test_autoencode_field_primary_arm_keeps_ulf_and_drops_target() -> None:
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    out = DatasetInstantiator._autoencode_field(idx, arm="primary")
    assert out[0]["primary_path"] == "/d/ulf.nii"  # input stays ULF
    assert out[0]["target_path"] is None  # HF arm dropped


def test_autoencode_field_target_arm_missing_target_raises() -> None:
    with pytest.raises(ValueError, match="requires every record"):
        DatasetInstantiator._autoencode_field(
            [{"primary_path": "/d/ulf.nii", "file_id": "x"}], arm="target"
        )


def test_create_nifti_universal_hf_to_hf_autoencodes_hf(monkeypatch) -> None:
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(dataset_type="nifti_paired", bidirectional_mode="hf_to_hf")
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    for got in captured["indices"]:
        assert got[0]["primary_path"] == "/d/hf.nii"  # input ← HF
        assert got[0]["target_path"] is None  # ULF dropped


def test_create_nifti_universal_ulf_to_ulf_autoencodes_ulf(monkeypatch) -> None:
    captured = _capture_universal(monkeypatch)
    cfg = _Cfg(dataset_type="nifti_paired", bidirectional_mode="ulf_to_ulf")
    idx = [{"primary_path": "/d/ulf.nii", "target_path": "/d/hf.nii", "file_id": "s"}]
    DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    for got in captured["indices"]:
        assert got[0]["primary_path"] == "/d/ulf.nii"  # input stays ULF
        assert got[0]["target_path"] is None  # HF dropped


def test_create_nifti_universal_hf_to_hf_missing_target_raises(monkeypatch) -> None:
    _capture_universal(monkeypatch)
    cfg = _Cfg(dataset_type="nifti_paired", bidirectional_mode="hf_to_hf")
    idx = [{"primary_path": "/d/ulf.nii", "file_id": "s"}]  # no target_path
    with pytest.raises(ValueError, match="requires every record"):
        DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)


# ── slice_2d wrapping (3D-volume → per-2D-slice, 2026-06-22) ──────────────────


def _patch_volumetric_universal(monkeypatch):
    """Patch UniversalMRIDataset with a volumetric stub that exposes .index."""

    class _StubVol:
        def __init__(self, index, **kwargs):
            self.index = index

        def __len__(self):
            return len(self.index)

        def __getitem__(self, i):  # pragma: no cover — shape metadata avoids load
            raise AssertionError("should not load when records carry shape")

    import spectramr.data.builders.dataset_instantiator as di

    monkeypatch.setattr(di, "UniversalMRIDataset", _StubVol)


def test_create_nifti_universal_slice_2d_wraps(monkeypatch) -> None:
    _patch_volumetric_universal(monkeypatch)
    from spectramr.data.datasets.slice_dataset import SliceVolumeDataset

    cfg = _Cfg(dataset_type="nifti_paired", slice_2d=True)
    idx = [{"primary_path": "u", "target_path": "h", "file_id": "s", "shape": [1, 4, 4, 3]}]
    tr, va = DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    assert isinstance(tr, SliceVolumeDataset) and isinstance(va, SliceVolumeDataset)
    assert len(tr) == 3  # depth-3 volume → 3 slices


def test_create_nifti_universal_slice_2d_default_does_not_wrap(monkeypatch) -> None:
    _patch_volumetric_universal(monkeypatch)
    from spectramr.data.datasets.slice_dataset import SliceVolumeDataset

    cfg = _Cfg(dataset_type="nifti_paired")  # slice_2d absent → False
    idx = [{"primary_path": "u", "target_path": "h", "file_id": "s", "shape": [1, 4, 4, 3]}]
    tr, _ = DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    assert not isinstance(tr, SliceVolumeDataset)


def test_create_nifti_universal_slice_2d_slab_depth_from_patch(monkeypatch) -> None:
    # patch_size depth 3 → 3-slice slabs for a 3D slab model
    _patch_volumetric_universal(monkeypatch)
    cfg = _Cfg(dataset_type="nifti_paired", slice_2d=True, patch_size=(128, 128, 3))
    idx = [{"primary_path": "u", "target_path": "h", "file_id": "s", "shape": [1, 4, 4, 6]}]
    tr, _ = DatasetInstantiator._create_nifti_universal(cfg, idx, idx, None, None)
    assert tr.slab_depth == 3
    assert len(tr) == 2  # depth 6 / slab 3 = 2 windows


# ── WS1: single-file train/val split — honest handling ────────────────────────


def _m4raw_config(validation_split: float) -> SimpleNamespace:
    """Config stub covering only the attributes ``_create_m4raw_repetition``
    reads before the single-file guard fires (no dataset is constructed)."""
    return _Cfg(
        dataset_type="m4raw",
        num_virtual_coils=4,
        kspace_percentile=99.0,
        normalize_kspace=True,
        single_contrast=False,
        coil_processing_mode="svd",
        log_scaling=False,
        validation_split=validation_split,
    )


def test_m4raw_empty_val_with_split_raises_not_leak() -> None:
    """An empty validation split with validation_split>0 must RAISE, not
    silently reuse the training files for validation (the old
    ``val_h5 or train_h5`` leak)."""
    config = _m4raw_config(validation_split=0.2)
    with pytest.raises(ValueError, match="leak"):
        DatasetInstantiator.create_datasets(
            config,
            train_index=[{"primary_path": "/nonexistent/a.h5"}],
            val_index=[],
            train_transforms=None,
            val_transforms=None,
        )


def test_m4raw_empty_val_train_only_does_not_raise_on_guard() -> None:
    """validation_split==0 is the train-only escape hatch: the leak guard must
    NOT fire (construction may still fail later on missing files, which is a
    different, honest failure)."""
    config = _m4raw_config(validation_split=0.0)
    try:
        DatasetInstantiator.create_datasets(
            config,
            train_index=[{"primary_path": "/nonexistent/a.h5"}],
            val_index=[],
            train_transforms=None,
            val_transforms=None,
        )
    except ValueError as exc:
        assert "leak" not in str(exc), "train-only must not trip the leak guard"
    except Exception:
        # Missing-file / dataset-construction errors are acceptable here — we
        # only assert the leak guard did not reject a legitimate train-only run.
        pass


def test_create_mrixfields_threads_rescale_per_image_to_BOTH_splits() -> None:
    """The renorm knob must reach train AND val, and default off when absent (#15).

    A knob wired into the train dataset but not the val one is the classic
    half-wiring bug: training would keep the corpus scale while validation
    renormalised, so the metrics would grade against a different reference than
    the one the model was fit to. Assert both, plus the schema-sourced default
    for a config stub that predates the field.
    """
    fields = [0.1, 3.0, 7.0]
    full = [
        {
            "subject_id": s,
            "contrast": "T1w",
            "pairing_group": f"{s}|T1w",
            "primary_path": f"{s}_{f}",
            "field_strength": f,
        }
        for f in fields
        for s in ("s1", "s2", "s3", "s4")
    ]
    split = int(0.8 * len(full))

    def _build(**extra):
        cfg = _Cfg(
            dataset_type="mrixfields",
            # ulf_source (like the sibling re-split test): the flat 80/20 slice of a
            # field-sorted index leaves val holding only 7 T, and the field-pinned
            # policies are the ones that regroup it back into complete groups.
            mrixfields_pairing_policy="ulf_source",
            mrixfields_target_field=None,
            expose_field_strength_target=True,
            mrixfields_slice_mode="central",  # keep construction lazy (no real files)
            validation_split=0.25,
            **extra,
        )
        return DatasetInstantiator._create_mrixfields(cfg, full[:split], full[split:], None, None)

    # Absent from the stub -> schema default (False), not a crash.
    train_ds, val_ds = _build()
    assert train_ds._rescale_per_image is False
    assert val_ds._rescale_per_image is False

    # Explicitly enabled -> reaches BOTH splits.
    train_ds, val_ds = _build(mrixfields_rescale_per_image=True)
    assert train_ds._rescale_per_image is True
    assert val_ds._rescale_per_image is True


class TestDatasetTypeVocabularyIsReachable:
    """Every canonical ``dataset_type`` must reach a construction branch.

    ``graph_mri`` was canonical, sat in ``_SELF_INDEXED_DATASET_TYPES`` and in
    both collation maps, and had NO branch here -- so declaring it fell through
    to the final "not recognised" raise, from a message that did not list it
    either. This class is the total oracle that makes that state impossible:
    parametrised over the SSOT, so a canonical type with no branch fails at once.

    Behavioural rather than a source grep -- but every ``_create_*`` is stubbed,
    so it asserts which branch the dispatcher CHOSE without constructing
    anything. The first cut called the real constructors and ran for minutes:
    the ``pde_synthetic`` branch actually solves PDEs.
    """

    @staticmethod
    def _cfg(dataset_type):
        from types import SimpleNamespace

        return SimpleNamespace(
            dataset_type=dataset_type,
            coils=SimpleNamespace(num_virtual_coils=4),
            source=SimpleNamespace(index_path=None),
            manifest_roles=None,
        )

    @staticmethod
    def _stub_creators(monkeypatch):
        from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

        for attr in [a for a in dir(DatasetInstantiator) if a.startswith("_create_")]:
            monkeypatch.setattr(
                DatasetInstantiator,
                attr,
                lambda *a, **k: ("train", "val"),
                raising=False,
            )

    def test_every_canonical_type_reaches_a_branch(self, monkeypatch):
        from spectramr.config.schemas.data import CANONICAL_DATASET_TYPES
        from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

        self._stub_creators(monkeypatch)
        unreachable = []
        for dt in CANONICAL_DATASET_TYPES:
            try:
                DatasetInstantiator.create_datasets(self._cfg(dt), [], [], None, None)
            except ValueError as exc:
                if "is not recognised" in str(exc):
                    unreachable.append(dt)
            except Exception:
                # Any OTHER failure means the branch was entered, which is all
                # this invariant claims.
                pass
        assert not unreachable, (
            f"canonical dataset_type(s) with no construction branch: {unreachable}"
        )

    def test_an_unknown_type_still_raises(self, monkeypatch):
        from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

        self._stub_creators(monkeypatch)
        with pytest.raises(ValueError, match="is not recognised"):
            DatasetInstantiator.create_datasets(
                self._cfg("definitely_not_a_type"), [], [], None, None
            )

    def test_the_error_message_is_derived_from_the_ssot(self, monkeypatch):
        """It was hand-written, and wrong in BOTH directions.

        It omitted ``mrixfields`` and ``oracle_bssfp`` -- 87 arms' worth of real,
        servable types -- while advertising ten alias spellings that can never
        reach this point.
        """
        from spectramr.config.schemas.data import (
            CANONICAL_DATASET_TYPES,
            DATASET_TYPE_ALIASES,
        )
        from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

        self._stub_creators(monkeypatch)
        with pytest.raises(ValueError) as exc:
            DatasetInstantiator.create_datasets(
                self._cfg("definitely_not_a_type"), [], [], None, None
            )
        msg = str(exc.value)
        for t in CANONICAL_DATASET_TYPES:
            assert t in msg, f"canonical type {t!r} missing from the error message"
        for alias in DATASET_TYPE_ALIASES:
            if alias not in CANONICAL_DATASET_TYPES:
                assert f" {alias}," not in msg, f"unreachable alias {alias!r} advertised"


class TestCineHeterogeneousFrameGuard:
    """A cine cohort whose volumes disagree on frame count cannot be batched.

    Frames occupy the TorchIO channel slot, so unequal frame counts are
    unequal channel counts and torchio's collate raises ``stack expects each
    tensor to be equal size`` from inside the loader -- naming neither cine
    nor the knob that would fix it. The guard says it at build time instead.
    """

    @staticmethod
    def _cfg(batch_size: int, total_frames: int | None, root):
        from spectramr.config.schemas.data import DataConfigSchema

        return DataConfigSchema.model_validate(
            {
                "dataset_type": "cine",
                "source": {"root": str(root)},
                "temporal": {
                    "enabled": True,
                    "target_source": "self",
                    "total_frames": total_frames,
                },
                "loader": {"batch_size": batch_size},
            }
        )

    def test_batched_cine_without_total_frames_raises(self, tmp_path):
        from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

        with pytest.raises(ValueError) as exc:
            DatasetInstantiator.create_datasets(self._cfg(4, None, tmp_path), [], [], None, None)
        msg = str(exc.value)
        assert "total_frames" in msg
        # Both escapes must be named, plus why the third does not exist yet.
        assert "batch_size: 1" in msg
        assert "temporal_sampler" in msg
        # And it must not read as "your cohort is heterogeneous" -- the builder
        # cannot know that without opening every volume. It is asking for an
        # assertion, and a homogeneous cohort hits this too.
        assert "declaration requirement" in msg

    def test_batch_size_one_is_allowed_without_total_frames(self, tmp_path):
        """Serial serving handles a heterogeneous cohort fine -- do not block it."""
        from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

        with pytest.raises(FileNotFoundError, match="Cine index empty"):
            DatasetInstantiator.create_datasets(self._cfg(1, None, tmp_path), [], [], None, None)

    def test_total_frames_lifts_the_batch_restriction(self, tmp_path):
        """Declaring the count is the assertion that makes stacking safe."""
        from spectramr.data.builders.dataset_instantiator import DatasetInstantiator

        with pytest.raises(FileNotFoundError, match="Cine index empty"):
            DatasetInstantiator.create_datasets(self._cfg(4, 12, tmp_path), [], [], None, None)


def test_create_mrixfields_multi_field_regroups_group_aware() -> None:
    """REGRESSION: ``multi_field`` must be in ``_create_mrixfields``'s regroup tuple.

    Driving ``_create_mrixfields`` is the point. A sibling test that called
    ``_regroup_mrixfields_multi_source`` directly proved the FUNCTION groups correctly
    and stayed green when ``"multi_field"`` was deleted from the tuple at the call site
    -- the three-of-four-sites shape this policy's ``_TUPLE_POLICIES`` owner exists to
    prevent, one file further out.

    Two claims, one call:

    1. **The regroup runs.** On the field-SORTED manifest a flat 80/20 record slice puts
       every field below 7 T in train, so val carries no 0.1 T. ``_build_multi_field``
       raises on a declared field no record holds, so an un-regrouped val cannot build
       at all -- ``len(val_ds) > 0`` is only reachable through the regroup.
    2. **It runs with ``group_by_subject=True``.** Two contrasts per subject make the
       coarse (``subject_id``) and fine (``subject|contrast``) keys distinguishable: on
       the fine key a subject's T1w can land in train and its T2w in val.

       ``validation_split`` is load-bearing here and was chosen by measurement, not by
       taste. ``_regroup_mrixfields_multi_source`` splits GROUP KEYS in manifest order,
       so at 0.25 the eight fine keys cut cleanly between subjects (val = ``s4|T1w`` +
       ``s4|T2w``) and BOTH keys keep subjects whole -- the assertion below passes under
       the fine key too, and the test is blind. At 0.4 the boundary lands mid-subject
       (val = ``s3|T2w``, ``s4|*``), stranding ``s3``. Measured across 0.2-0.5, only
       0.375 and 0.4 discriminate. The coarse key spans at NO fraction, which is the
       invariant; this fraction is simply one where the fine key visibly does not.
    """
    fields = [0.1, 1.5, 3.0, 5.0, 7.0]
    full = [
        {
            "subject_id": s,
            "contrast": c,
            "pairing_group": f"{s}|{c}",
            "primary_path": f"{s}_{c}_{f}",
            "field_strength": f,
        }
        for f in fields  # field-major: mirrors mrixfields2026_train.json's ordering
        for s in ("s1", "s2", "s3", "s4")
        for c in ("T1w", "T2w")
    ]
    cut = int(0.8 * len(full))
    train_idx, val_idx = full[:cut], full[cut:]
    assert {r["field_strength"] for r in val_idx} == {7.0}, "fixture must strand 0.1 T"
    # Second precondition, for claim 2: at this fraction the FINE key must strand a
    # subject, or the disjointness assertion below passes under both keys and cannot
    # tell them apart. Executed rather than asserted in prose -- a rounding change in
    # split_index would otherwise re-blind the test silently (0.25 measured blind).
    val_split = 0.4  # see the docstring: 0.25 is measured blind. ONE owner -- the
    # precondition below and the cfg further down must probe the SAME fraction, or
    # re-blinding the cfg leaves the precondition testing a fraction nothing uses.
    _ft, _fv, _ = DatasetInstantiator._regroup_mrixfields_multi_source(
        train_idx, val_idx, None, val_split, group_by_subject=False, explicit_val=False
    )
    assert {r["subject_id"] for r in _ft} & {r["subject_id"] for r in _fv}, (
        "fixture no longer discriminates group_by_subject; re-measure the fraction"
    )

    cfg = _Cfg(
        dataset_type="mrixfields",
        mrixfields_pairing_policy="multi_field",
        mrixfields_fields=[1.5, 3.0, 5.0, 7.0],
        mrixfields_heldout_fields=[0.1],
        mrixfields_target_field=None,
        # central keeps construction lazy so the fake paths are never opened; all_slices
        # (the default) foreground-scans every volume and would hit the missing files.
        mrixfields_slice_mode="central",
        validation_split=val_split,
    )
    train_ds, val_ds = DatasetInstantiator._create_mrixfields(cfg, train_idx, val_idx, None, None)
    assert len(train_ds) > 0
    assert len(val_ds) > 0, "val can only build a stack if the regroup restored 0.1 T"

    def _subjects(ds) -> set[str]:
        return {recs[0]["subject_id"] for recs, _judge in ds._tuples}

    assert not (_subjects(train_ds) & _subjects(val_ds)), "subject leaked across the split"
