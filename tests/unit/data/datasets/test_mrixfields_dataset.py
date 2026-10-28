"""Tests for MRIxFieldsPairedDataset (MICCAI MRIxFields2026 cross-field pairs)."""

from __future__ import annotations

import pytest
import torch

from spectramr.data.datasets.mrixfields_dataset import (
    MRIxFieldsPairedDataset,
    _default_nifti_loader,
)


def _records() -> list[dict]:
    base = {
        "subject_id": "s1",
        "contrast": "T1w",
        "pairing_group": "s1|T1w",
        "shape": [8, 8, 1],
    }
    return [
        {**base, "primary_path": "f01.nii", "field_strength": 0.1},
        {**base, "primary_path": "f30.nii", "field_strength": 3.0},
        {**base, "primary_path": "f70.nii", "field_strength": 7.0},
    ]


def _fake_loader(path: str) -> torch.Tensor:
    seed = sum(ord(c) for c in path)
    g = torch.Generator().manual_seed(seed)
    return torch.rand(8, 8, 1, generator=g)


def _stub_nifti_header(monkeypatch, depth: int = 1) -> None:
    """Stub the header-only depth peek and reset the slice cache.

    ``_default_nifti_loader`` reads the NIfTI header to locate the central
    slice, so tests driving it with a synthetic path need a stubbed header.
    """
    import sys
    import types

    from spectramr.data.datasets import mrixfields_dataset as mod

    fake_nib = types.ModuleType("nibabel")
    fake_nib.load = lambda _p: types.SimpleNamespace(shape=(8, 8, depth))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nibabel", fake_nib)
    mod._load_central_slice.cache_clear()


def test_all_pairs_count_and_keys() -> None:
    ds = MRIxFieldsPairedDataset(
        _records(), pairing_policy="all_pairs", image_loader=_fake_loader
    )
    assert len(ds) == 6  # 3 fields -> 3*2 ordered pairs
    s = ds[0]
    for k in (
        "input",
        "target",
        "field_strength",
        "field_strength_target",
        "contrast_id",
        "subject_id",
    ):
        assert k in s
    assert s["input"].shape == (1, 8, 8)
    assert float(s["input"].min()) >= 0.0
    assert float(s["input"].max()) <= 1.0


def test_fixed_target_only_targets_pinned_field() -> None:
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="fixed_target",
        target_field=7.0,
        image_loader=_fake_loader,
    )
    assert len(ds) == 2  # 2 non-target sources -> 2 pairs to 7T
    for i in range(len(ds)):
        assert float(ds[i]["field_strength_target"]) == 7.0


def test_ulf_source_only_sources_pinned() -> None:
    ds = MRIxFieldsPairedDataset(
        _records(), pairing_policy="ulf_source", image_loader=_fake_loader
    )
    assert len(ds) == 2  # source 0.1 -> {3.0, 7.0}
    for i in range(len(ds)):
        assert abs(float(ds[i]["field_strength"]) - 0.1) < 1e-6


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        MRIxFieldsPairedDataset(
            _records(), pairing_policy="bogus", image_loader=_fake_loader
        )


def test_fixed_target_requires_target_field() -> None:
    with pytest.raises(ValueError):
        MRIxFieldsPairedDataset(
            _records(), pairing_policy="fixed_target", image_loader=_fake_loader
        )


def test_input_target_differ() -> None:
    ds = MRIxFieldsPairedDataset(
        _records(), pairing_policy="all_pairs", image_loader=_fake_loader
    )
    s = ds[0]
    assert not torch.allclose(s["input"], s["target"])


def _tio_compose():
    """A real ``tio.Compose`` (shape-preserving) — the production transform type.

    The dataset pipeline injects a ``tio.Compose`` (built for a ``tio.Subject``).
    Applying ANY tio transform to a RAW dict raises ``RuntimeError: ... a value
    for "include" must be specified`` — the 2026-06-20 crash that took out every
    transform-bearing mrixfields2026 arm. The dataset must wrap image keys in a
    ``tio.Subject`` first (mirroring contrast_aware / oracle_bssfp).
    """
    import torchio as tio

    return tio.Compose([tio.RescaleIntensity(out_min_max=(0.0, 1.0))])


def test_transform_applied_via_subject_wrapping_pair_policy() -> None:
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        image_loader=_fake_loader,
        transform=_tio_compose(),
    )
    s = ds[0]  # must NOT raise the tio "include" RuntimeError
    assert isinstance(s, dict)
    assert s["input"].shape == (1, 8, 8)
    assert s["target"].shape == (1, 8, 8)
    # non-image scalar/id keys ride through untouched
    assert s["subject_id"] == "s1"
    assert "field_strength_target" in s


def test_transform_applied_via_subject_wrapping_multi_source() -> None:
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="multi_source",
        target_field=7.0,
        image_loader=_fake_loader,
        transform=_tio_compose(),
    )
    item = ds[0]  # must NOT raise; the [N,1,H,W] sources stack survives the round-trip
    assert isinstance(item, dict)
    assert item["sources"].shape == (2, 1, 8, 8)
    assert item["input"].shape == (1, 8, 8)
    assert item["target"].shape == (1, 8, 8)


def _fake_loader_4d(path: str) -> torch.Tensor:
    """Mimic ``NiftiStrategy.load``: a 4-D channel-first volume ``[C, H, W, D]``.

    The real loader returns 4-D volumes, but the inline ``_fake_loader`` returns
    3-D ``[H, W, 1]`` — so the 4-D path (which crashed every NIfTI-backed arm with
    ``Input tensor must be 4D, but it is 5D``) was never exercised in tests.
    """
    seed = sum(ord(c) for c in path)
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 8, 8, 5, generator=g)  # [C=1, H=8, W=8, D=5]


def test_to_chw_unit_collapses_4d_volume_to_2d_slice() -> None:
    # REGRESSION (2026-06-20): ``NiftiStrategy.load`` returns 4-D ``[C,H,W,D]``;
    # ``_to_chw_unit`` must collapse it to a single ``[1,H,W]`` 2-D magnitude slice
    # (first channel, central depth). Previously a 4-D load fell through every branch
    # unchanged -> ``_apply_transform``'s ``unsqueeze(-1)`` produced a 5-D tensor that
    # ``tio.ScalarImage`` rejects, taking out every mrixfields2026 arm at batch 1.
    out = MRIxFieldsPairedDataset._to_chw_unit(torch.rand(1, 8, 8, 5))
    assert out.shape == (1, 8, 8)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_4d_volume_loader_through_transform_pair_policy() -> None:
    # The REAL crash path: a 4-D volume loader + an injected ``tio.Compose``. The inline
    # 3-D ``_fake_loader`` never reproduced it; this drives the 4-D loader end-to-end.
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        image_loader=_fake_loader_4d,
        transform=_tio_compose(),
    )
    s = ds[0]  # must NOT raise 'Input tensor must be 4D, but it is 5D'
    assert s["input"].shape == (1, 8, 8)
    assert s["target"].shape == (1, 8, 8)


def test_4d_volume_loader_through_transform_multi_source() -> None:
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="multi_source",
        target_field=7.0,
        image_loader=_fake_loader_4d,
        transform=_tio_compose(),
    )
    item = ds[
        0
    ]  # the [N,1,H,W] sources stack from 4-D volumes must survive the round-trip
    assert item["sources"].shape == (2, 1, 8, 8)
    assert item["input"].shape == (1, 8, 8)
    assert item["target"].shape == (1, 8, 8)


def test_multi_source_emits_stacked_tuple() -> None:
    # B-1.1 travelling-volunteer tuple: one item per subject-contrast group carrying ALL
    # source fields (< target) stacked as 'sources' against the shared target.
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="multi_source",
        target_field=7.0,
        image_loader=_fake_loader,
    )
    assert len(ds) == 1  # one (subject,contrast) group
    assert ds._source_fields == [0.1, 3.0]  # distinct non-target fields, sorted
    item = ds[0]
    assert item["sources"].shape == (2, 1, 8, 8)  # N=2 sources stacked
    assert float(item["field_strength_target"]) == 7.0
    assert float(item["field_strength"]) == pytest.approx(
        0.1
    )  # first-source compat alias
    assert (
        "field_strengths" not in item
    )  # per-source vector NOT emitted (renderer field-invariant)
    assert torch.equal(
        item["input"], item["sources"][0]
    )  # input = first source (compat)
    assert item["target"].shape == (1, 8, 8)


def test_multi_source_skips_incomplete_groups() -> None:
    # A second subject missing the 3.0T source cannot form the full tuple -> skipped.
    recs = [
        *_records(),
        {
            "subject_id": "s2",
            "contrast": "T1w",
            "pairing_group": "s2|T1w",
            "primary_path": "s2_f01.nii",
            "field_strength": 0.1,
        },
        {
            "subject_id": "s2",
            "contrast": "T1w",
            "pairing_group": "s2|T1w",
            "primary_path": "s2_f70.nii",
            "field_strength": 7.0,
        },
    ]
    ds = MRIxFieldsPairedDataset(
        recs, pairing_policy="multi_source", target_field=7.0, image_loader=_fake_loader
    )
    assert len(ds) == 1 and ds._skipped_groups == 1  # only s1 is complete


def test_multi_source_requires_target_field() -> None:
    with pytest.raises(ValueError, match="requires target_field"):
        MRIxFieldsPairedDataset(
            _records(), pairing_policy="multi_source", image_loader=_fake_loader
        )


def test_multi_source_needs_two_sources() -> None:
    one_src = [
        {
            "subject_id": "s1",
            "contrast": "T1w",
            "pairing_group": "s1|T1w",
            "primary_path": "a",
            "field_strength": 3.0,
        },
        {
            "subject_id": "s1",
            "contrast": "T1w",
            "pairing_group": "s1|T1w",
            "primary_path": "b",
            "field_strength": 7.0,
        },
    ]
    with pytest.raises(ValueError, match=">=2 distinct source fields"):
        MRIxFieldsPairedDataset(
            one_src,
            pairing_policy="multi_source",
            target_field=7.0,
            image_loader=_fake_loader,
        )


def test_fixed_target_raises_on_no_matching_target() -> None:
    # REGRESSION (#12): fixed_target that matches nothing (e.g. target 7.0 on a proxy with no
    # 7T column) must FAIL FAST like multi_source, not return a silently-empty dataset.
    with pytest.raises(ValueError, match="produced 0 pairs"):
        MRIxFieldsPairedDataset(
            _records(),
            pairing_policy="fixed_target",
            target_field=99.0,
            image_loader=_fake_loader,
        )


def test_zero_pairs_message_points_at_the_split_not_missing_data() -> None:
    # REGRESSION (2026-06-22): the 0-pairs hint used to say "the local proxy manifest has
    # no 7T column", which mis-blamed the DATA. The real cause is usually a flat train/val
    # slice stranding the pinned field — the message must steer to the group-aware re-split.
    # _records() has ONE 3-field group (max_group>1), so the split-blame branch fires.
    with pytest.raises(ValueError, match="re-split GROUP-AWARE"):
        MRIxFieldsPairedDataset(
            _records(),
            pairing_policy="fixed_target",
            target_field=99.0,
            image_loader=_fake_loader,
        )


def test_zero_pairs_singleton_groups_points_at_manifest_regeneration() -> None:
    # REGRESSION (2026-07-07 cluster failure): the common cause is a STALE unpaired-by-
    # design prospective manifest — every pairing group holds a single field because the
    # gitignored manifest was built before ordinal pairing. Then blaming the split is
    # wrong; the message must steer to REGENERATING via the committed generator.
    singletons = [
        {
            "subject_id": "s0",
            "contrast": "T1w",
            "pairing_group": "s0|T1w",
            "primary_path": "a.nii",
            "field_strength": 0.1,
        },
        {
            "subject_id": "s1",
            "contrast": "T1w",
            "pairing_group": "s1|T1w",
            "primary_path": "b.nii",
            "field_strength": 7.0,
        },
    ]
    with pytest.raises(ValueError, match="build_mrixfields2026_manifest"):
        MRIxFieldsPairedDataset(
            singletons,
            pairing_policy="fixed_target",
            target_field=7.0,  # present in the index but never co-resident with a source
            image_loader=_fake_loader,
        )


def test_multi_source_raises_on_duplicate_field() -> None:
    # Travelling-volunteer invariant: one volume per (subject,contrast,field). A duplicate
    # field is a malformed manifest -> fail fast (vs a silent dict-collapse that drops data).
    dup = [
        {
            "subject_id": "s1",
            "contrast": "T1w",
            "pairing_group": "s1|T1w",
            "primary_path": "a",
            "field_strength": 3.0,
        },
        {
            "subject_id": "s1",
            "contrast": "T1w",
            "pairing_group": "s1|T1w",
            "primary_path": "b",
            "field_strength": 3.0,
        },  # duplicate 3.0T
        {
            "subject_id": "s1",
            "contrast": "T1w",
            "pairing_group": "s1|T1w",
            "primary_path": "c",
            "field_strength": 0.1,
        },
        {
            "subject_id": "s1",
            "contrast": "T1w",
            "pairing_group": "s1|T1w",
            "primary_path": "d",
            "field_strength": 7.0,
        },
    ]
    with pytest.raises(ValueError, match="duplicate field"):
        MRIxFieldsPairedDataset(
            dup,
            pairing_policy="multi_source",
            target_field=7.0,
            image_loader=_fake_loader,
        )


def test_multi_source_honors_expose_target_field() -> None:
    # The expose_field_strength_target knob must gate emission for multi_source too (#15).
    off = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="multi_source",
        target_field=7.0,
        expose_target_field=False,
        image_loader=_fake_loader,
    )[0]
    assert "field_strength_target" not in off


# ---------------------------------------------------------------------------
# _default_nifti_loader: the NiftiStrategy returns a dict, not a bare tensor
# ---------------------------------------------------------------------------
def test_default_nifti_loader_extracts_tensor_from_strategy_dict(monkeypatch) -> None:
    """``_default_nifti_loader`` must extract the image tensor from the
    NiftiStrategy dict return (``{"data": tensor, "affine":..., "metadata":...}``),
    not tensorise the whole dict.

    Regression (2026-06-20): every NIfTI-backed mrixfields2026 arm crashed in the
    DataLoader with ``RuntimeError: Could not infer dtype of dict`` because the
    loader passed the dict straight to ``torch.as_tensor``. All other dataset
    tests inject a fake ``image_loader``, so the real default loader was never
    exercised — hence the bug shipped. See
    ``src/spectramr/data/io_strategies.py::NiftiStrategy.load``.
    """
    import spectramr.data.io_strategies as io_mod

    expected = torch.zeros(1, 8, 8, 1)

    class _DictStrategy:
        def load(self, path, metadata=None):
            return {"data": expected, "affine": [[1, 0, 0, 0]], "metadata": {}}

    monkeypatch.setattr(
        io_mod.IOStrategyFactory, "get", staticmethod(lambda name: _DictStrategy())
    )
    _stub_nifti_header(monkeypatch)
    out = _default_nifti_loader("/fake.nii.gz")
    assert isinstance(out, torch.Tensor)
    assert out.shape == expected.shape


def test_default_nifti_loader_raises_clear_error_on_keyless_dict(monkeypatch) -> None:
    """A dict return with no ``image``/``data`` key must raise a self-describing
    error naming the path + the keys present — not the cryptic torch dtype error."""
    import spectramr.data.io_strategies as io_mod

    class _BadStrategy:
        def load(self, path, metadata=None):
            return {"affine": [[1, 0, 0, 0]], "metadata": {}}

    monkeypatch.setattr(
        io_mod.IOStrategyFactory, "get", staticmethod(lambda name: _BadStrategy())
    )
    _stub_nifti_header(monkeypatch)
    with pytest.raises(ValueError, match=r"no 'image'/'data' key"):
        _default_nifti_loader("/fake.nii.gz")


def test_multi_source_shared_source_fields_pins_uniform_n() -> None:
    # An explicit source_fields set (passed by the instantiator from the FULL manifest) pins
    # N regardless of this split's own field coverage -> train/val get the same arity.
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="multi_source",
        target_field=7.0,
        source_fields=[0.1, 3.0],
        image_loader=_fake_loader,
    )
    assert ds[0]["sources"].shape[0] == 2


def test_parse_fastmri_index_retains_mrixfields_keys(tmp_path) -> None:
    # REGRESSION (cohort-wide CRITICAL): dataset_type='mrixfields' routes through
    # parse_fastmri_index, which previously dropped field_strength/subject_id/contrast/
    # pairing_group -> every mrixfields arm KeyError'd at construction on the real path.
    # The inline-record tests never exercised the parser. Drive a real manifest THROUGH it.
    import json

    from spectramr.data.datasets.universal_dataset import parse_fastmri_index

    manifest = {
        "data_root": "",
        "records": [
            {
                "path": f"s1_{f}.nii",
                "file_id": f"s1_{f}",
                "field_strength": f,
                "contrast": "T1w",
                "subject_id": "s1",
                "pairing_group": "s1|T1w",
            }
            for f in (0.1, 3.0, 7.0)
        ],
    }
    mpath = tmp_path / "mrixfields_manifest.json"
    mpath.write_text(json.dumps(manifest))
    parsed = parse_fastmri_index(str(mpath))
    for rec in parsed:
        assert "field_strength" in rec and "subject_id" in rec
        assert "contrast" in rec and "pairing_group" in rec
    # and the parsed records build a working multi_source dataset
    ds = MRIxFieldsPairedDataset(
        parsed,
        pairing_policy="multi_source",
        target_field=7.0,
        image_loader=_fake_loader,
    )
    assert len(ds) == 1 and ds[0]["sources"].shape[0] == 2


def test_expose_target_field_gates_emission() -> None:
    # The expose_field_strength_target knob (wired from data config) gates whether
    # field_strength_target is emitted (pitfall #15: the knob must be read).
    on = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        expose_target_field=True,
        image_loader=_fake_loader,
    )[0]
    off = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        expose_target_field=False,
        image_loader=_fake_loader,
    )[0]
    assert "field_strength_target" in on
    assert "field_strength_target" not in off


def _unpaired_records() -> list[dict]:
    """One field per volunteer, distinct subjects — the retrospective cohort shape."""
    base = {"contrast": "T1w", "shape": [8, 8, 1]}
    return [
        {
            **base,
            "subject_id": "r1",
            "pairing_group": "r1|T1w",
            "primary_path": "u01.nii",
            "field_strength": 0.1,
        },
        {
            **base,
            "subject_id": "r2",
            "pairing_group": "r2|T1w",
            "primary_path": "u30.nii",
            "field_strength": 3.0,
        },
        {
            **base,
            "subject_id": "r3",
            "pairing_group": "r3|T1w",
            "primary_path": "u70.nii",
            "field_strength": 7.0,
        },
    ]


def test_prior_policy_identity_for_unpaired_singletons() -> None:
    # Unpaired pool (retrospective): each singleton becomes an IDENTITY pair so the
    # field-conditioned prior trains on every image as its own target. No target_field
    # required, and field_strength == field_strength_target (the image's own field).
    ds = MRIxFieldsPairedDataset(
        _unpaired_records(), pairing_policy="prior", image_loader=_fake_loader
    )
    assert len(ds) == 3
    for i in range(len(ds)):
        s = ds[i]
        assert torch.equal(s["input"], s["target"])  # identity
        assert float(s["field_strength"]) == float(s["field_strength_target"])


def test_prior_policy_pairs_when_group_has_ulf_and_higher() -> None:
    # Paired group (prospective travelling volunteer): prior emits ULF→HF pairs so
    # validation grades the real recon task (source pinned to 0.1 T).
    ds = MRIxFieldsPairedDataset(
        _records(), pairing_policy="prior", image_loader=_fake_loader
    )
    assert len(ds) == 2  # 0.1 -> {3.0, 7.0}
    for i in range(len(ds)):
        s = ds[i]
        assert abs(float(s["field_strength"]) - 0.1) < 1e-6
        assert float(s["field_strength_target"]) in (3.0, 7.0)
        assert not torch.equal(s["input"], s["target"])  # a real pair, not identity


def test_prior_policy_needs_no_target_field() -> None:
    # Unlike fixed_target/multi_source, prior must construct without target_field.
    MRIxFieldsPairedDataset(
        _unpaired_records(), pairing_policy="prior", image_loader=_fake_loader
    )


# --- multi_contrast policy (idea 2.1 relaxometry) ----------------------------


def _multicontrast_records() -> list[dict]:
    recs = []
    for subj in ("s1", "s2"):
        for field in (0.1, 1.5, 3.0, 7.0):
            for contrast in ("T1w", "T2w", "FLAIR"):
                recs.append(
                    {
                        "subject_id": subj,
                        "contrast": contrast,
                        "field_strength": field,
                        "primary_path": f"{subj}_{field}_{contrast}.nii",
                        "shape": [8, 8, 1],
                    }
                )
    return recs


def test_multi_contrast_requires_target_field() -> None:
    with pytest.raises(ValueError):
        MRIxFieldsPairedDataset(
            _multicontrast_records(), pairing_policy="multi_contrast"
        )


def test_multi_contrast_stacks_contrasts() -> None:
    ds = MRIxFieldsPairedDataset(
        _multicontrast_records(),
        pairing_policy="multi_contrast",
        target_field=7.0,
        image_loader=_fake_loader,
    )
    # 2 subjects x 3 source fields (0.1, 1.5, 3.0) below 7T = 6 tuples
    assert len(ds) == 6
    s = ds[0]
    assert s["input"].shape[0] == 3  # T1w + T2w + FLAIR stacked
    assert s["target"].shape[0] == 1
    assert "field_strength_target" in s
    assert float(s["field_strength_target"]) == 7.0
    assert float(s["field_strength"]) < 7.0


def test_multi_contrast_output_contrast_selects_target() -> None:
    ds = MRIxFieldsPairedDataset(
        _multicontrast_records(),
        pairing_policy="multi_contrast",
        target_field=7.0,
        output_contrast="T2w",
        image_loader=_fake_loader,
    )
    assert int(ds[0]["contrast_id"]) == 1  # T2w


def test_multi_contrast_skips_incomplete_and_raises_on_empty() -> None:
    # Only one contrast present -> no complete stack -> 0 tuples -> raise.
    recs = [
        {
            "subject_id": "s1",
            "contrast": "T1w",
            "field_strength": f,
            "primary_path": f"p{f}.nii",
            "shape": [8, 8, 1],
        }
        for f in (0.1, 7.0)
    ]
    # single contrast still stacks (C=1) at source 0.1 with target 7T -> 1 tuple
    ds = MRIxFieldsPairedDataset(
        recs,
        pairing_policy="multi_contrast",
        target_field=7.0,
        image_loader=_fake_loader,
    )
    assert len(ds) == 1
    # no target field present -> raise
    recs_no_target = [
        {
            "subject_id": "s1",
            "contrast": "T1w",
            "field_strength": f,
            "primary_path": f"p{f}.nii",
            "shape": [8, 8, 1],
        }
        for f in (0.1, 3.0)
    ]
    with pytest.raises(ValueError):
        MRIxFieldsPairedDataset(
            recs_no_target,
            pairing_policy="multi_contrast",
            target_field=7.0,
            image_loader=_fake_loader,
        )


def test_multi_contrast_pinned_contrasts_fix_arity() -> None:
    # Pinned set keeps the stack arity fixed even if a subject lacks a contrast.
    recs = _multicontrast_records()
    # drop FLAIR for subject s2 -> s2 skipped, but arity stays 3 (pinned)
    recs = [
        r for r in recs if not (r["subject_id"] == "s2" and r["contrast"] == "FLAIR")
    ]
    ds = MRIxFieldsPairedDataset(
        recs,
        pairing_policy="multi_contrast",
        target_field=7.0,
        contrasts=["T1w", "T2w", "FLAIR"],
        image_loader=_fake_loader,
    )
    assert all(ds[i]["input"].shape[0] == 3 for i in range(len(ds)))


def test_multi_contrast_rejects_unknown_output_contrast() -> None:
    with pytest.raises(ValueError):
        MRIxFieldsPairedDataset(
            _multicontrast_records(),
            pairing_policy="multi_contrast",
            target_field=7.0,
            output_contrast="PD",  # not a known contrast
            image_loader=_fake_loader,
        )


def test_canonical_contrasts_module_fn() -> None:
    from spectramr.data.datasets.mrixfields_dataset import canonical_contrasts

    assert canonical_contrasts(_multicontrast_records()) == ["T1w", "T2w", "FLAIR"]


# --------------------------------------------------------------------------- #
# mrixfields_pairing_feasibility — the static necessary-condition predictor
# that lets the audit reject a guaranteed-0-pairs config at pre-flight
# (the 2026-07-07 stale-manifest cluster failure) WITHOUT re-implementing
# _build_pairs. A False return is a GUARANTEE of 0 pairs; True is not a promise.
# --------------------------------------------------------------------------- #
def _multifield_group(fields: list[float], *, group: str = "s1|T1w") -> list[dict]:
    return [
        {
            "subject_id": group.split("|")[0],
            "contrast": group.split("|")[1],
            "pairing_group": group,
            "field_strength": f,
            "primary_path": f"{group}_{f}.nii",
        }
        for f in fields
    ]


def _singleton_groups(fields: list[float]) -> list[dict]:
    # Each field lands in its OWN pairing group — the stale unpaired-by-design
    # signature (ordinal pairing never applied).
    recs = []
    for i, f in enumerate(fields):
        recs.extend(_multifield_group([f], group=f"vol{i}|T1w"))
    return recs


def test_feasibility_fixed_target_singletons_infeasible_names_regeneration() -> None:
    from spectramr.data.datasets.mrixfields_dataset import mrixfields_pairing_feasibility

    # Exactly the cluster failure: fields present = [0.1,1.5,3.0,5.0,7.0] but every
    # group is a singleton, so fixed_target=7.0 forms 0 pairs.
    recs = _singleton_groups([0.1, 1.5, 3.0, 5.0, 7.0])
    ok, reason = mrixfields_pairing_feasibility(
        recs, policy="fixed_target", target_field=7.0
    )
    assert ok is False
    assert reason is not None and "build_mrixfields2026_manifest" in reason


def test_feasibility_fixed_target_paired_group_feasible() -> None:
    from spectramr.data.datasets.mrixfields_dataset import mrixfields_pairing_feasibility

    recs = _multifield_group([0.1, 3.0, 7.0])  # one group, all fields together
    ok, reason = mrixfields_pairing_feasibility(
        recs, policy="fixed_target", target_field=7.0
    )
    assert ok is True
    assert reason is None


def test_feasibility_fixed_target_absent_pinned_field_infeasible() -> None:
    from spectramr.data.datasets.mrixfields_dataset import mrixfields_pairing_feasibility

    recs = _multifield_group([0.1, 1.5, 3.0])  # no 7T column at all
    ok, reason = mrixfields_pairing_feasibility(
        recs, policy="fixed_target", target_field=7.0
    )
    assert ok is False
    assert reason is not None and "7" in reason


def test_feasibility_ulf_source_needs_ulf_and_higher_in_a_group() -> None:
    from spectramr.data.datasets.mrixfields_dataset import mrixfields_pairing_feasibility

    ok, _ = mrixfields_pairing_feasibility(
        _multifield_group([0.1, 3.0]), policy="ulf_source", target_field=None
    )
    assert ok is True
    # singletons: 0.1 and 3.0 in different groups → no ULF→HF pair
    ok2, reason2 = mrixfields_pairing_feasibility(
        _singleton_groups([0.1, 3.0]), policy="ulf_source", target_field=None
    )
    assert ok2 is False and reason2 is not None


def test_feasibility_multi_source_needs_two_sources_below_target() -> None:
    from spectramr.data.datasets.mrixfields_dataset import mrixfields_pairing_feasibility

    ok, _ = mrixfields_pairing_feasibility(
        _multifield_group([0.1, 1.5, 7.0]), policy="multi_source", target_field=7.0
    )
    assert ok is True
    ok2, reason2 = mrixfields_pairing_feasibility(
        _multifield_group([1.5, 7.0]), policy="multi_source", target_field=7.0
    )
    assert ok2 is False and reason2 is not None  # only ONE source < target


def test_feasibility_multi_contrast_requires_target_field_present() -> None:
    from spectramr.data.datasets.mrixfields_dataset import mrixfields_pairing_feasibility

    ok, _ = mrixfields_pairing_feasibility(
        _multifield_group([0.1, 3.0, 7.0]), policy="multi_contrast", target_field=7.0
    )
    assert ok is True
    ok2, _ = mrixfields_pairing_feasibility(
        _multifield_group([0.1, 3.0]), policy="multi_contrast", target_field=7.0
    )
    assert ok2 is False


def test_feasibility_all_pairs_singletons_infeasible() -> None:
    from spectramr.data.datasets.mrixfields_dataset import mrixfields_pairing_feasibility

    ok, reason = mrixfields_pairing_feasibility(
        _singleton_groups([0.1, 3.0, 7.0]), policy="all_pairs", target_field=None
    )
    assert ok is False and reason is not None


def test_feasibility_prior_always_feasible() -> None:
    from spectramr.data.datasets.mrixfields_dataset import mrixfields_pairing_feasibility

    # prior emits identity pairs for singletons, so it NEVER produces 0 pairs.
    ok, reason = mrixfields_pairing_feasibility(
        _singleton_groups([0.1, 3.0]), policy="prior", target_field=None
    )
    assert ok is True and reason is None


# --- Central-slice lazy read + cache (GPU-starvation fix) -------------------
# The volumes are 364x436x364 .nii.gz. The loader previously called
# NiftiStrategy.load() with no metadata, taking the get_fdata() branch: a
# ~462 MB float64 decode of the WHOLE volume, of which _to_chw_unit keeps a
# single central slice (0.3%). With no cache, every pair re-decoded it, which
# starved the GPU to ~0% util and OOM-killed DataLoader workers.


class _SpyNiftiStrategy:
    """Records how load() is called and honours the lazy slice_index path."""

    def __init__(self, depth: int = 9) -> None:
        self.calls: list[dict | None] = []
        self.depth = depth
        self.volume = torch.arange(4 * 4 * depth, dtype=torch.float32).reshape(
            4, 4, depth
        )

    def load(self, path: str, metadata: dict | None = None) -> dict:
        self.calls.append(metadata)
        if metadata is not None and metadata.get("slice_index") is not None:
            plane = self.volume[..., int(metadata["slice_index"])]  # (4, 4)
            return {"data": plane.reshape(1, 4, 4, 1)}
        # full-volume branch (what we must stop doing)
        return {"data": self.volume.reshape(1, 4, 4, self.depth)}


@pytest.fixture
def spy_nifti(monkeypatch: pytest.MonkeyPatch) -> _SpyNiftiStrategy:
    import sys
    import types

    from spectramr.data import io_strategies
    from spectramr.data.datasets import mrixfields_dataset as mod

    spy = _SpyNiftiStrategy(depth=9)
    monkeypatch.setattr(
        io_strategies.IOStrategyFactory, "get", staticmethod(lambda _name: spy)
    )

    # header-only shape peek must not read voxels
    fake_nib = types.ModuleType("nibabel")
    fake_nib.load = lambda p: types.SimpleNamespace(shape=(4, 4, spy.depth))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nibabel", fake_nib)

    mod._load_central_slice.cache_clear()
    return spy


def test_default_loader_reads_only_the_central_slice(spy_nifti) -> None:
    """It must use the lazy dataobj slice path, never the full get_fdata() decode."""
    _default_nifti_loader("vol.nii.gz")

    assert spy_nifti.calls, "strategy was never invoked"
    meta = spy_nifti.calls[0]
    assert meta is not None and meta.get("slice_index") == 4, (
        f"loader must request the central slice (depth 9 -> index 4); got {meta!r}. "
        "Passing no metadata takes the get_fdata() full-volume branch."
    )


def test_default_loader_caches_repeated_paths(spy_nifti) -> None:
    """The same volume is referenced by many pairs; decode it once, not N times."""
    for _ in range(5):
        _default_nifti_loader("vol.nii.gz")
    _default_nifti_loader("other.nii.gz")

    assert len(spy_nifti.calls) == 2, (
        f"expected 1 decode per unique path (2 total), got {len(spy_nifti.calls)} — "
        "uncached loads are what starve the GPU."
    )


def test_cached_slice_is_not_mutable_by_callers(spy_nifti) -> None:
    """Callers get a clone; in-place transforms must not poison the cache."""
    first = _default_nifti_loader("vol.nii.gz")
    first.add_(999.0)
    second = _default_nifti_loader("vol.nii.gz")

    assert not torch.allclose(first, second), "cache handed out a mutable shared tensor"


def test_central_slice_value_matches_full_volume_behaviour(spy_nifti) -> None:
    """Behaviour-preserving: same pixels the old full-volume path produced."""
    got = MRIxFieldsPairedDataset._to_chw_unit(_default_nifti_loader("vol.nii.gz"))

    expected = spy_nifti.volume[..., 4].unsqueeze(0)  # central slice, [1, H, W]
    lo, hi = expected.amin(), expected.amax()
    expected = ((expected - lo) / (hi - lo)).clamp(0, 1)
    assert got.shape == (1, 4, 4)
    assert torch.allclose(got, expected)


# --------------------------------------------------------------------------- #
# mrixfields_slice_mode — central (default) / all_slices / volume
# --------------------------------------------------------------------------- #
def _two_field_group() -> list[dict]:
    """One pairing group with two fields -> fixed_target(7.0) yields exactly ONE pair,
    so all_slices len == the target volume's foreground-slice count (no pair-count factor).
    """
    base = {"subject_id": "s1", "contrast": "T1w", "pairing_group": "s1|T1w"}
    return [
        {**base, "primary_path": "f01.nii", "field_strength": 0.1},
        {**base, "primary_path": "f70.nii", "field_strength": 7.0},
    ]


def _multislice_loader(fg_indices, air_indices):
    """A deterministic ``path -> [1,8,8,D]`` loader with known foreground/air slices.

    Foreground slices carry a non-constant, clearly-lit pattern (so per-slice renorm
    spans [0,1] and the served content is distinguishable); air slices are pure zeros.
    Returns the SAME volume for every path so source/target share the slice (the
    co-registration invariant all_slices relies on)."""
    depth = len(fg_indices) + len(air_indices)
    vol = torch.zeros(1, 8, 8, depth)
    for i in fg_indices:
        g = torch.Generator().manual_seed(100 + i)
        vol[0, :, :, i] = 0.1 + 0.9 * torch.rand(8, 8, generator=g)

    def loader(_path: str) -> torch.Tensor:
        return vol.clone()

    return loader, vol


def test_slice_mode_defaults_to_all_slices() -> None:
    # all_slices is the default because serving one central slice discards ~363/364 of
    # every volume. It is affordable because of the EPOCH ORDER (VolumeBlockedSliceSampler
    # amortises one volume decode over every slice of every container sharing it), not
    # because slices are individually cheap — see docs/mrixfields_gpu_starvation_fix.rst.
    ds = MRIxFieldsPairedDataset(
        _records(), pairing_policy="all_pairs", image_loader=_fake_loader
    )
    assert ds._slice_mode == "all_slices"


def test_all_slices_serves_every_foreground_slice() -> None:
    # A 5-slice all-foreground volume yields 5 samples per pair where central yields 1.
    # _records() has 3 fields -> 6 ordered all_pairs pairs.
    ds_all = MRIxFieldsPairedDataset(
        _records(), pairing_policy="all_pairs", image_loader=_fake_loader_4d
    )
    ds_central = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        slice_mode="central",
        image_loader=_fake_loader_4d,
    )
    assert len(ds_central) == 6  # one central slice per pair
    assert (
        len(ds_all) == 6 * 5
    )  # every foreground slice of the depth-5 volume (default)
    assert ds_all[0]["input"].shape == (1, 8, 8)  # still a 2-D slice payload


def test_container_volume_paths_reports_what_each_container_reads() -> None:
    """The mapping VolumeBlockedSliceSampler blocks on.

    Only the dataset knows the per-policy container shape — a pair reads {src, tgt}, a
    multi_source tuple reads every source plus the target — so a wrong mapping here would
    silently make the sampler block on the wrong volume set and buy no reuse.
    """
    ds = MRIxFieldsPairedDataset(
        _records(), pairing_policy="all_pairs", image_loader=_fake_loader_4d
    )
    volumes = ds.container_volume_paths()
    assert len(volumes) == 6  # one entry per container
    assert all(len(v) == 2 for v in volumes)  # a pair reads exactly source + target
    assert {v for entry in volumes for v in entry} == {"f01.nii", "f30.nii", "f70.nii"}


def test_container_volume_paths_is_empty_for_central_mode() -> None:
    # No slice expansion -> nothing to order -> the sampler does not apply.
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        slice_mode="central",
        image_loader=_fake_loader_4d,
    )
    assert ds.container_volume_paths() == []


def test_max_resident_volumes_is_read_and_validated() -> None:
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        max_resident_volumes=9,
        image_loader=_fake_loader_4d,
    )
    assert ds._max_resident_volumes == 9

    with pytest.raises(ValueError, match="max_resident_volumes must be >= 2"):
        MRIxFieldsPairedDataset(
            _records(),
            pairing_policy="all_pairs",
            max_resident_volumes=1,
            image_loader=_fake_loader_4d,
        )


def test_volume_cache_is_not_pickled_to_workers() -> None:
    """A cache of ~226 MB volumes must not be serialised into every spawn worker.

    Also pins that the dataset still WORKS after the round-trip: __setstate__ has to
    rebuild the loader, which was bound to the dropped cache.
    """
    import pickle

    from spectramr.data.datasets import mrixfields_dataset as mod

    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        max_resident_volumes=4,
        image_loader=_fake_loader_4d,
    )
    # Attach a cache after construction: the injected loader keeps the foreground scan
    # off disk, and the cache is what this test is about.
    ds._volume_cache = mod._VolumeLRU(4, _fake_loader_4d)
    ds._load = ds._volume_cache
    ds._load("f01.nii")
    assert ds._volume_cache.misses == 1

    revived = pickle.loads(pickle.dumps(ds))
    assert revived._volume_cache is not None  # rebuilt, not carried over
    assert revived._volume_cache.misses == 0  # empty in the worker
    assert revived._max_resident_volumes == 4


def test_volume_lru_evicts_least_recently_used() -> None:
    from spectramr.data.datasets import mrixfields_dataset as mod

    loaded: list[str] = []

    def _loader(path: str):
        loaded.append(path)
        return torch.zeros(1, 2, 2, 1)

    lru = mod._VolumeLRU(2, _loader)
    lru("a"), lru("b"), lru("a")  # 'a' refreshed -> 'b' is the LRU victim
    lru("c")  # evicts 'b'
    lru("a")  # still resident
    assert lru.hits == 2
    assert loaded == ["a", "b", "c"]


def test_volume_budget_exceeds_worst_case_working_set() -> None:
    """The default budget must exceed the widest per-``__getitem__`` working set.

    ``_getitem_multi_source`` loads N sources PLUS the target, so with the corpus' four
    source fields it touches five distinct volumes per item. A budget of 4 against that
    is a cyclic access of period 5 over capacity 4 — a hit rate of exactly zero. This
    pins the relationship rather than the number, so adding a fifth source field fails
    here instead of silently re-inerting the cache.
    """
    from spectramr.data.datasets.mrixfields_dataset import _VOLUME_CACHE_MAXSIZE

    widest_source_arity = 4  # {0.1, 1.5, 3, 5} -> 7T, the corpus' multi_source fan-in
    assert widest_source_arity + 1 < _VOLUME_CACHE_MAXSIZE


def test_volume_cache_cannot_be_poisoned_by_a_transform() -> None:
    """Every coercion returns a FRESH tensor, so the un-cloned volume cache is safe.

    ``_default_nifti_volume_loader`` stopped cloning its ~226 MB payload (a ~300x
    waste to extract one slice). That is only sound while no coercion hands back a
    view into the cached tensor, so assert exactly that: mutate each coercion's
    output and confirm the source volume is unchanged. A future coercion that ends
    in a view fails here rather than corrupting the LRU for every later item.
    """
    vol = torch.rand(1, 8, 8, 5)
    before = vol.clone()
    for out in (
        MRIxFieldsPairedDataset._extract_chw_unit(vol, 2),
        MRIxFieldsPairedDataset._to_chw_unit(vol),
        MRIxFieldsPairedDataset._volume_to_cdhw_unit(vol),
    ):
        out.zero_()
    assert torch.equal(vol, before)


def test_all_slices_rejects_a_source_shallower_than_its_target() -> None:
    """A shallower source is caught at construction, not as a mid-epoch IndexError.

    Slice indices come from the target and are applied to every source, so a source
    with less depth used to raise ``IndexError`` at an arbitrary item — hours into a
    cluster run, with nothing naming the offending pair.
    """
    recs = _records()
    recs[0]["shape"] = [8, 8, 2]  # source shallower than the depth-8 target below
    for r in recs[1:]:
        r["shape"] = [8, 8, 8]

    with pytest.raises(ValueError, match="shallower than its target"):
        MRIxFieldsPairedDataset(
            recs,
            pairing_policy="all_pairs",
            slice_mode="all_slices",
            image_loader=_fake_loader_4d,
        )


def test_depth_guard_skips_records_with_no_declared_shape() -> None:
    """An absent ``shape`` is a SKIP, never a rejection.

    Same polarity as the workflow axis guards: the guard cannot tell, so it says
    nothing rather than inventing a depth and rejecting a valid manifest.
    """
    recs = [{k: v for k, v in r.items() if k != "shape"} for r in _records()]
    ds = MRIxFieldsPairedDataset(
        recs,
        pairing_policy="all_pairs",
        slice_mode="all_slices",
        image_loader=_fake_loader_4d,
    )
    assert len(ds) == 6 * 5


def test_unknown_slice_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown mrixfields slice_mode"):
        MRIxFieldsPairedDataset(
            _records(),
            pairing_policy="all_pairs",
            slice_mode="bogus",
            image_loader=_fake_loader,
        )


def test_central_mode_is_byte_identical_to_pre_change_path() -> None:
    # Explicit central slice_mode must reproduce EXACTLY the pre-change payload, which
    # was _to_chw_unit(loader(path)) — so the opt-in debug shortcut is unchanged.
    #
    # mrixfields_rescale_per_image made the per-image renorm opt-in, so "the pre-change
    # payload" is now the rescale=True branch. Both settings are asserted: the original
    # guard survives verbatim under rescale_per_image=True, and the new default is
    # pinned to the un-renormalised coercion. Checking only the new default would have
    # quietly retired the refactor guard this test exists to be.
    for rescale in (True, False):
        ds_central = MRIxFieldsPairedDataset(
            _records(),
            pairing_policy="all_pairs",
            slice_mode="central",
            rescale_per_image=rescale,
            image_loader=_fake_loader_4d,
        )
        assert len(ds_central) == 6
        for i in range(len(ds_central)):
            s = ds_central[i]
            src, tgt = ds_central._pairs[i]
            assert torch.equal(
                s["input"],
                MRIxFieldsPairedDataset._to_chw_unit(
                    _fake_loader_4d(src["primary_path"]), rescale=rescale
                ),
            )
            assert torch.equal(
                s["target"],
                MRIxFieldsPairedDataset._to_chw_unit(
                    _fake_loader_4d(tgt["primary_path"]), rescale=rescale
                ),
            )


def test_all_slices_expands_to_foreground_slices_only() -> None:
    # K=3 foreground + M=4 air slices -> exactly K samples (air dropped); each [1,H,W];
    # the served slice is the correct foreground index of the (co-registered) volume.
    fg, air = [1, 3, 5], [0, 2, 4, 6]
    loader, vol = _multislice_loader(fg, air)
    ds = MRIxFieldsPairedDataset(
        _two_field_group(),
        pairing_policy="fixed_target",
        target_field=7.0,
        slice_mode="all_slices",
        image_loader=loader,
    )
    assert len(ds) == len(fg)  # exactly K, air dropped
    for j, sidx in enumerate(fg):
        s = ds[j]
        assert s["input"].shape == (1, 8, 8)
        assert s["target"].shape == (1, 8, 8)
        # rescale=False mirrors the dataset default (mrixfields_rescale_per_image).
        expected = MRIxFieldsPairedDataset._extract_chw_unit(
            vol, sidx, rescale=False
        )
        # right slice served, for BOTH source and target (same co-registered index)
        assert torch.allclose(s["input"], expected)
        assert torch.allclose(s["target"], expected)


def test_all_slices_foreground_filter_has_teeth() -> None:
    # The filter must DROP a pure-air slice and a barely-lit slice below the fraction
    # threshold, and KEEP slices at/above it — assert the exact boundary. 8x8 = 64
    # voxels; _FOREGROUND_MIN_FRAC = 0.02 -> need >= 1.28 voxels, i.e. 1 voxel (1.56%)
    # is dropped, 2 voxels (3.125%) is kept.
    depth = 4
    vol = torch.zeros(1, 8, 8, depth)
    # slice 0: pure air (0 lit) -> DROP
    vol[0, 0, 0, 1] = 1.0  # slice 1: 1 lit voxel = 1.56% < 2% -> DROP
    vol[0, 0, 0, 2] = 1.0
    vol[0, 0, 1, 2] = 1.0  # slice 2: 2 lit voxels = 3.125% >= 2% -> KEEP
    vol[0, :, :, 3] = 1.0  # slice 3: fully lit -> KEEP

    def loader(_p: str) -> torch.Tensor:
        return vol.clone()

    ds = MRIxFieldsPairedDataset(
        _two_field_group(),
        pairing_policy="fixed_target",
        target_field=7.0,
        slice_mode="all_slices",
        image_loader=loader,
    )
    assert ds._foreground_slices("f70.nii") == [2, 3]
    assert len(ds) == 2  # only slices 2 and 3 survive


def test_all_slices_all_air_raises() -> None:
    # A target volume that is pure air everywhere -> 0 foreground slices -> fail fast
    # (no silently-empty loader, #9/#15).
    def loader(_p: str) -> torch.Tensor:
        return torch.zeros(1, 8, 8, 4)

    with pytest.raises(ValueError, match="0 foreground slices"):
        MRIxFieldsPairedDataset(
            _two_field_group(),
            pairing_policy="fixed_target",
            target_field=7.0,
            slice_mode="all_slices",
            image_loader=loader,
        )


def test_all_slices_wraps_multi_source_tuples() -> None:
    # all_slices must apply to the tuple policies too (index map wraps _tuples). One
    # multi_source tuple x K foreground slices -> K samples, each with a sliced sources stack.
    fg, air = [2, 4], [0, 1, 3]
    loader, vol = _multislice_loader(fg, air)
    ds = MRIxFieldsPairedDataset(
        _records(),  # 3-field group -> 1 tuple, sources = {0.1, 3.0}
        pairing_policy="multi_source",
        target_field=7.0,
        slice_mode="all_slices",
        image_loader=loader,
    )
    assert len(ds) == len(fg)  # 1 tuple x K foreground slices
    s = ds[0]
    assert s["sources"].shape == (2, 1, 8, 8)  # N=2 sources, sliced to [1,H,W]
    assert s["input"].shape == (1, 8, 8)
    assert s["target"].shape == (1, 8, 8)
    assert torch.allclose(
        s["target"],
        MRIxFieldsPairedDataset._extract_chw_unit(vol, fg[0], rescale=False),
    )


def test_volume_mode_emits_whole_normalized_volume() -> None:
    # volume slice_mode: emit the WHOLE [C,H,W,D], normalized over the whole volume.
    # The whole-volume min-max is the rescale_per_image=True path; the default keeps
    # the corpus scale (test_rescale_flag_reaches_every_coercion covers the default).
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        slice_mode="volume",
        rescale_per_image=True,
        image_loader=_fake_loader_4d,
    )
    assert len(ds) == 6  # one sample per pair (not per slice)
    s = ds[0]
    assert s["input"].shape == (1, 8, 8, 5)  # [C, H, W, D]
    assert s["target"].shape == (1, 8, 8, 5)
    assert float(s["input"].min()) == pytest.approx(0.0)  # global min -> 0
    assert float(s["input"].max()) == pytest.approx(1.0)  # global max -> 1


def test_volume_mode_through_transform() -> None:
    # volume [C,H,W,D] is already 4-D (tio-shaped): _apply_transform must NOT unsqueeze
    # it into a 5-D tensor tio would reject.
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="all_pairs",
        slice_mode="volume",
        image_loader=_fake_loader_4d,
        transform=_tio_compose(),
    )
    s = ds[0]
    assert s["input"].shape == (1, 8, 8, 5)
    assert s["target"].shape == (1, 8, 8, 5)


def test_volume_mode_rejects_tuple_policies() -> None:
    # Honest refusal (no facade): volume + a tuple policy would emit a 5-D sources stack
    # no strategy consumes -> raise at construction.
    for policy in ("multi_source", "multi_contrast"):
        with pytest.raises(ValueError, match="not supported"):
            MRIxFieldsPairedDataset(
                _records(),
                pairing_policy=policy,
                target_field=7.0,
                slice_mode="volume",
                image_loader=_fake_loader,
            )


def test_all_slices_default_volume_loader_end_to_end(monkeypatch) -> None:
    # Exercise the DEFAULT (non-injected) whole-volume loader path: foreground scan +
    # per-slice serving through _default_nifti_volume_loader / _load_full_volume, proving
    # the production path is real (not a facade that only works with injected loaders).
    from spectramr.data import io_strategies

    depth = 4
    vol = torch.zeros(1, 4, 4, depth)
    vol[0, :, :, 1] = 1.0  # foreground
    vol[0, :, :, 3] = 1.0  # foreground; slices 0, 2 are air

    class _Strat:
        def load(self, path, metadata=None):
            # full-volume path takes NO slice metadata
            assert metadata is None
            return {"data": vol.clone()}

    monkeypatch.setattr(
        io_strategies.IOStrategyFactory, "get", staticmethod(lambda _n: _Strat())
    )
    ds = MRIxFieldsPairedDataset(
        _two_field_group(),
        pairing_policy="fixed_target",
        target_field=7.0,
        slice_mode="all_slices",  # NO image_loader -> default whole-volume loader
    )
    assert len(ds) == 2  # foreground slices 1 and 3
    s = ds[0]
    assert s["input"].shape == (1, 4, 4)
    assert s["target"].shape == (1, 4, 4)
    # Nothing to clear or leak: the volume cache is per-instance now, so it cannot
    # outlive this dataset into another test (a module-level lru_cache used to).
    assert ds._volume_cache is not None


# ---------------------------------------------------------------------------
# mrixfields_rescale_per_image: the per-image renorm is OPT-IN
# ---------------------------------------------------------------------------


def _scaled_volume_loader(depth: int = 5):
    """Loader whose volumes differ ONLY in global scale, mimicking two field strengths.

    ``f01.nii`` (the 0.1 T stand-in) is dim, ``f70.nii`` (7 T) is bright, over the
    SAME structure. That is exactly the signal field translation must learn, so it is
    the signal a per-image renorm destroys.
    """
    structure = torch.linspace(0.0, 1.0, 8 * 8).reshape(8, 8)
    scales = {"f01.nii": 0.2, "f30.nii": 0.5, "f70.nii": 0.9}

    def _load(path: str) -> torch.Tensor:
        vol = torch.zeros(1, 8, 8, depth)
        for d in range(depth):
            vol[0, :, :, d] = structure * scales[path]
        return vol

    return _load


def test_rescale_per_image_defaults_off_and_keeps_the_cross_field_scale() -> None:
    """Default (off) preserves the source/target intensity RATIO — the learnable signal.

    The corpus is already [0,1] globally (module docstring), so the renorm does not
    establish a scale, it replaces a meaningful global one with a per-image one. With
    it off, a 0.2-scaled source against a 0.9-scaled target keeps that ratio; the
    field conditioning then has something to explain.
    """
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="ulf_source",
        slice_mode="all_slices",
        image_loader=_scaled_volume_loader(),
    )
    assert ds._rescale_per_image is False  # the default, asserted explicitly
    s = ds[0]
    src_max = float(s["input"].max())
    tgt_max = float(s["target"].max())
    # Source is the 0.1 T record; target is 3.0 or 7.0 T -> strictly brighter.
    assert src_max < tgt_max, "the cross-field intensity relationship must survive"
    assert src_max == pytest.approx(0.2, abs=1e-5)


def test_rescale_per_image_on_destroys_the_cross_field_scale() -> None:
    """The opt-in path is the OLD behaviour, and this pins why it is no longer default.

    Both images renormalise to max 1.0, so the source arrives pre-scaled to the
    target's range and the intensity mapping the model is meant to learn is gone.
    """
    ds = MRIxFieldsPairedDataset(
        _records(),
        pairing_policy="ulf_source",
        slice_mode="all_slices",
        rescale_per_image=True,
        image_loader=_scaled_volume_loader(),
    )
    s = ds[0]
    assert float(s["input"].max()) == pytest.approx(1.0, abs=1e-5)
    assert float(s["target"].max()) == pytest.approx(1.0, abs=1e-5)


def test_rescale_flag_reaches_every_coercion() -> None:
    """All three coercions honour the flag; none silently keeps renormalising (#15)."""
    vol = torch.rand(1, 8, 8, 5) * 0.4  # global max ~0.4, never 1.0
    for got in (
        MRIxFieldsPairedDataset._extract_chw_unit(vol, 2, rescale=False),
        MRIxFieldsPairedDataset._to_chw_unit(vol, rescale=False),
        MRIxFieldsPairedDataset._volume_to_cdhw_unit(vol, rescale=False),
    ):
        assert float(got.max()) < 0.95, "rescale=False must not stretch to full range"
    for got in (
        MRIxFieldsPairedDataset._extract_chw_unit(vol, 2, rescale=True),
        MRIxFieldsPairedDataset._to_chw_unit(vol, rescale=True),
        MRIxFieldsPairedDataset._volume_to_cdhw_unit(vol, rescale=True),
    ):
        assert float(got.max()) == pytest.approx(1.0, abs=1e-5)


def test_rescale_false_still_returns_a_fresh_tensor() -> None:
    """Extends the cache-poisoning invariant to the NEW rescale=False path.

    ``rescale=False`` skips the arithmetic that used to guarantee a fresh tensor, so
    the un-cloned volume cache is now safe only because ``clamp`` never returns a
    view. Assert that directly rather than trusting it.
    """
    vol = torch.rand(1, 8, 8, 5)
    before = vol.clone()
    for out in (
        MRIxFieldsPairedDataset._extract_chw_unit(vol, 2, rescale=False),
        MRIxFieldsPairedDataset._to_chw_unit(vol, rescale=False),
        MRIxFieldsPairedDataset._volume_to_cdhw_unit(vol, rescale=False),
    ):
        out.zero_()
    assert torch.equal(vol, before)


class TestVolumeCacheByteSanityBound:
    """#198 (audit D3). The budget stays a VOLUME COUNT; bytes are a guard.

    Re-denominating the budget in bytes would be the obvious fix and the wrong
    one: the count is a working-set floor. ``_getitem_multi_source`` touches N
    sources plus the target, so a budget under the widest container's volume
    count makes every sample evict the volume the next one needs — a cyclic
    access of period 5 over capacity 4 is a hit rate of exactly zero. A byte
    budget could silently pick a count under that floor.

    So the byte bound sits on top, and can only run after the first decode:
    nothing knows a volume's size at construction time.
    """

    @staticmethod
    def _lru(maxsize, nbytes, max_bytes):
        import torch

        from spectramr.data.datasets.mrixfields_dataset import _VolumeLRU

        n = max(1, nbytes // 4)  # float32
        return _VolumeLRU(
            maxsize, lambda _p: torch.zeros(n, dtype=torch.float32), max_bytes
        )

    def test_a_budget_that_cannot_fit_raises_on_the_first_decode(self) -> None:
        lru = self._lru(maxsize=6, nbytes=2**30, max_bytes=2**30)  # 6 GB vs 1 GB
        with pytest.raises(ValueError, match="per DataLoader worker"):
            lru("a.nii.gz")

    def test_the_message_carries_the_arithmetic_and_the_floor(self) -> None:
        """An oom_kill has no attribution; this error is where the run explains
        itself, so it must name the count, the size, and why lowering the count
        has a limit."""
        lru = self._lru(maxsize=6, nbytes=2**30, max_bytes=2**30)
        with pytest.raises(ValueError) as exc:
            lru("a.nii.gz")
        message = str(exc.value)
        assert "max_resident_volumes" in message
        assert "num_workers" in message
        assert "sources + the target" in message  # the floor, so nobody goes under it

    def test_a_budget_that_fits_is_untouched(self) -> None:
        lru = self._lru(maxsize=6, nbytes=2**20, max_bytes=2**30)  # 6 MB vs 1 GB
        assert lru("a.nii.gz") is not None
        assert lru.misses == 1

    def test_the_check_runs_once_not_per_load(self) -> None:
        """It is on the hot path; recomputing per miss buys nothing because the
        volume size does not change within a run."""
        lru = self._lru(maxsize=2, nbytes=2**20, max_bytes=2**30)
        for i in range(5):
            lru(f"{i}.nii.gz")
        assert lru._checked_footprint is True

    def test_it_can_be_disabled(self) -> None:
        """``max_bytes=None`` opts out — a node that genuinely has the RAM
        should not be blocked by a heuristic."""
        lru = self._lru(maxsize=6, nbytes=2**30, max_bytes=None)
        assert lru("a.nii.gz") is not None

    def test_the_default_bound_leaves_headroom_over_the_documented_corpus(
        self,
    ) -> None:
        """6 x ~226 MB = ~1.4 GB is the figure the schema quotes. The bound must
        sit comfortably above it or it fires on a working setup."""
        from spectramr.data.datasets.mrixfields_dataset import (
            _VOLUME_CACHE_MAX_BYTES,
            _VOLUME_CACHE_MAXSIZE,
        )

        documented = _VOLUME_CACHE_MAXSIZE * 226 * 2**20
        assert 2 * documented < _VOLUME_CACHE_MAX_BYTES


# --------------------------------------------------------------------------------------
# multi_field: the SAME anatomy at every declared field, stacked as [M,H,W], with the
# held-out field(s) emitted separately and never fitted.
# --------------------------------------------------------------------------------------

_ALL_FIELDS = (0.1, 1.5, 3.0, 5.0, 7.0)


def _records_multi_field(subjects=("s1",), contrasts=("T1w",), fields=_ALL_FIELDS) -> list[dict]:
    out: list[dict] = []
    for sub in subjects:
        for con in contrasts:
            for f in fields:
                out.append(
                    {
                        "subject_id": sub,
                        "contrast": con,
                        "pairing_group": f"{sub}|{con}",
                        "shape": [8, 8, 1],
                        "primary_path": f"{sub}_{con}_{f}.nii",
                        "field_strength": f,
                    }
                )
    return out


def _mf(records=None, **kw):
    kw.setdefault("fields", [1.5, 3.0, 5.0, 7.0])
    kw.setdefault("image_loader", _fake_loader)
    return MRIxFieldsPairedDataset(
        records if records is not None else _records_multi_field(),
        pairing_policy="multi_field",
        **kw,
    )


def test_multi_field_stacks_declared_fields_in_declared_order() -> None:
    """Channel m is fields[m] -- the ORDER is the contract, not the sorted set.

    Declared deliberately out of ascending order: a builder that sorts internally
    would still produce a [M,H,W] stack of the right arity and the right images,
    and every shape assertion would pass while channel m no longer meant
    fields[m]. The model's `fields` buffer is registered from config in that same
    declared order (disp_bloch_ae.py:120), so a silent sort mis-assigns every
    field in the dispersion fit without raising.
    """
    declared = [7.0, 1.5, 5.0, 3.0]
    item = _mf(fields=declared)[0]
    assert item["input"].shape == (4, 8, 8)
    assert item["field_strengths"].tolist() == pytest.approx(declared)
    # Each channel is the image of its declared field, not merely some permutation.
    for m, f in enumerate(declared):
        expected = _fake_loader(f"s1_T1w_{f}.nii").permute(2, 0, 1)[0]
        torch.testing.assert_close(item["input"][m], expected)


def test_multi_field_emits_no_target_key() -> None:
    """No `target`, deliberately.

    DispersionBlochAEStrategy reads its reconstruction target as
    ``batch.get("target", x)`` (dispersion_bloch_ae_strategy.py:70), so an absent
    key makes the autoencoder reconstruct its own input -- the intended objective.
    Emitting a target would silently REPLACE that objective: a single-field image
    against an M-channel prediction (a shape error), or the held-out image (no
    error at all, and the extrapolation test would be fitting the field it claims
    to predict).
    """
    item = _mf(heldout_fields=[0.1])[0]
    assert "target" not in item
    assert "field_strength_target" not in item


def test_multi_field_heldout_is_emitted_and_never_in_the_stack() -> None:
    """Test 2 (fit 1.5/3/5/7 T, predict 0.1 T) becomes expressible.

    This configuration is what multi_source cannot express at all: its contract is
    "sources strictly below the target", so target_field=0.1 yields zero sources
    and raises.
    """
    item = _mf(fields=[1.5, 3.0, 5.0, 7.0], heldout_fields=[0.1])[0]
    assert item["input"].shape == (4, 8, 8)
    assert item["heldout"].shape == (1, 8, 8)
    assert item["heldout_field_strengths"].tolist() == pytest.approx([0.1])
    assert not any(abs(f - 0.1) < 1e-6 for f in item["field_strengths"].tolist())
    ulf = _fake_loader("s1_T1w_0.1.nii").permute(2, 0, 1)[0]
    torch.testing.assert_close(item["heldout"][0], ulf)


def test_multi_field_without_heldout_omits_the_keys() -> None:
    item = _mf()[0]
    assert "heldout" not in item
    assert "heldout_field_strengths" not in item


def test_multi_field_emits_scalar_field_strength_too() -> None:
    """~15 strategies and the generic unpack path index batch["field_strength"]."""
    item = _mf(fields=[1.5, 3.0, 5.0, 7.0])[0]
    assert item["field_strength"].ndim == 0
    assert float(item["field_strength"]) == 1.5


@pytest.mark.parametrize("with_transform", [True, False])
def test_multi_field_stack_survives_the_tio_round_trip(with_transform: bool) -> None:
    """Both paths, because _apply_transform RETURNS EARLY when transform is None.

    A test that only covered the configured-transform case would leave the
    no-transform path -- the one every arm without a `data.processing.transforms`
    block takes -- unexercised.
    """
    ds = _mf(heldout_fields=[0.1], transform=_tio_compose() if with_transform else None)
    item = ds[0]
    assert item["input"].shape == (4, 8, 8)
    assert item["heldout"].shape == (1, 8, 8)
    assert item["field_strengths"].tolist() == pytest.approx([1.5, 3.0, 5.0, 7.0])
    assert item["subject_id"] == "s1"


def test_multi_field_heldout_rides_the_same_subject_as_the_stack() -> None:
    """`heldout` must be in _IMAGE_KEYS or a spatial transform desynchronises it.

    tio samples one warp per Subject, so keys wrapped together are cropped/flipped
    identically and keys left out ride through untouched. A held-out image that
    skipped the wrap would be scored against a render of a DIFFERENTLY cropped
    stack -- misaligned, with no error raised. Cropping is the shape that shows it:
    it changes the spatial size, so an unwrapped key keeps the original one.
    """
    import torchio as tio

    ds = _mf(heldout_fields=[0.1], transform=tio.Compose([tio.Crop((1, 1, 2, 2, 0, 0))]))
    item = ds[0]
    assert item["input"].shape[-2:] == item["heldout"].shape[-2:]
    assert item["input"].shape[-2:] != (8, 8)  # the crop really fired


def test_multi_field_containers_cover_every_volume_the_item_reads() -> None:
    """The sampler orders an epoch by these paths; a short set thrashes the cache.

    `container_volume_paths` reads container[0] plus container[1], so the held-out
    volume has to be inside container[0] -- and the resident-volume budget floor is
    len(fields) + len(heldout_fields), not len(fields).
    """
    ds = _mf(fields=[1.5, 3.0, 5.0, 7.0], heldout_fields=[0.1], slice_mode="all_slices")
    paths = ds.container_volume_paths()
    assert len(paths) == 1
    assert paths[0] == frozenset(f"s1_T1w_{f}.nii" for f in _ALL_FIELDS)


def test_multi_field_skips_incomplete_groups_and_warns(caplog) -> None:
    """A corpus quietly losing groups must not look identical to a complete one."""
    recs = _records_multi_field(subjects=("s1", "s2"))
    recs = [r for r in recs if not (r["subject_id"] == "s2" and r["field_strength"] == 5.0)]
    with caplog.at_level("WARNING"):
        ds = _mf(recs, fields=[1.5, 3.0, 5.0, 7.0])
    assert len(ds) == 1  # s2 dropped
    assert ds._skipped_groups == 1
    assert any("skipped 1 incomplete group" in r.getMessage() for r in caplog.records)


def test_multi_field_requires_fields() -> None:
    with pytest.raises(ValueError, match="requires a non-empty"):
        MRIxFieldsPairedDataset(
            _records_multi_field(), pairing_policy="multi_field", image_loader=_fake_loader
        )


def test_multi_field_rejects_overlap_between_stack_and_heldout() -> None:
    """An overlap would fit the field the extrapolation test claims to predict."""
    with pytest.raises(ValueError, match="overlap"):
        _mf(fields=[1.5, 3.0, 7.0], heldout_fields=[7.0])


def test_multi_field_rejects_duplicate_fields() -> None:
    with pytest.raises(ValueError, match="duplicate field strengths"):
        _mf(fields=[3.0, 3.0, 7.0])


def test_multi_field_names_the_absent_field_not_zero_tuples() -> None:
    """A declared field the corpus lacks names ITSELF, not a broken manifest.

    Without the up-front check every group skips and the "0 tuples" raise fires,
    which is loud but attributes the failure to the manifest rather than to the
    config line that asked for a field the corpus never had.
    """
    with pytest.raises(ValueError, match=r"declares field\(s\) \[11.7\]"):
        _mf(fields=[1.5, 3.0, 11.7])


def test_fields_knobs_rejected_under_other_policies() -> None:
    """Never accept a knob the policy does not read (non-negotiable 8)."""
    with pytest.raises(ValueError, match="only read by"):
        MRIxFieldsPairedDataset(
            _records_multi_field(),
            pairing_policy="all_pairs",
            fields=[1.5, 3.0],
            image_loader=_fake_loader,
        )


def test_multi_field_refuses_volume_slice_mode() -> None:
    """cat(dim=0) over M volumes is [M,H,W,D] -- the 5-D shape no strategy consumes."""
    with pytest.raises(ValueError, match="slice_mode='volume' is not supported"):
        _mf(slice_mode="volume")


def test_every_non_pair_policy_has_a_loader_that_exists() -> None:
    """``__getitem__`` routes the three multi-* policies through a table of method names."""
    from spectramr.data.datasets.mrixfields_dataset import MRIxFieldsPairedDataset

    table = MRIxFieldsPairedDataset._GETITEM_BY_POLICY
    assert set(table) == {"multi_source", "multi_contrast", "multi_field"}
    for method in table.values():
        assert callable(getattr(MRIxFieldsPairedDataset, method))
