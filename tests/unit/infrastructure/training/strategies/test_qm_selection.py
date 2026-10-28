"""Which volumes a quality match reads, and how each one's geometry is measured."""

from __future__ import annotations

import json

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from spectramr.infrastructure.physics.acquisition_params import (  # noqa: E402
    MissingAcquisitionParameter,
)
from spectramr.infrastructure.training.strategies.qm_selection import (  # noqa: E402
    acquisition_of_path,
    hq_source_paths,
    ref_spacing_mm,
    select_paths,
)


def _nifti(path, zooms=(1.0, 1.0, 1.0)):
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 2), np.float32), np.diag([*zooms, 1.0])), str(path))
    return str(path)


def _manifest(tmp_path, records):
    doc = {"data_root": str(tmp_path), "records": records}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(doc))
    return str(p)


def _rec(name, field, contrast):
    return {"relative_path": name, "field_strength": field, "contrast": contrast}


def test_select_paths_filters_by_field_and_contrast(tmp_path):
    m = _manifest(
        tmp_path,
        [
            _rec("a.nii.gz", 3.0, "T1W"),
            _rec("b.nii.gz", 0.1, "T1W"),
            _rec("c.nii.gz", 0.1, "T2W"),
        ],
    )
    got = select_paths(m, field_t=0.1, contrast="T1W")
    assert [p.rsplit("/", 1)[-1] for p in got] == ["b.nii.gz"]


def test_select_paths_raises_and_lists_fields_when_nothing_matches(tmp_path):
    m = _manifest(tmp_path, [_rec("a.nii.gz", 3.0, "T1W")])
    with pytest.raises(ValueError, match="Fields present"):
        select_paths(m, field_t=0.1)


def test_unpaired_hq_side_is_the_declared_field_not_the_whole_manifest(tmp_path):
    # Planted: a mixed-field manifest must not hand 0.1 T volumes to the HQ side.
    m = _manifest(
        tmp_path,
        [
            _rec("hq.nii.gz", 3.0, "T1W"),
            _rec("lq.nii.gz", 0.1, "T1W"),
            _rec("x.nii.gz", 3.0, "T2W"),
        ],
    )
    got = hq_source_paths(
        pair_layout=None,
        field_pairs=None,
        hq_manifest=m,
        source_field_t=3.0,
        contrast="T1W",
    )
    assert [p.rsplit("/", 1)[-1] for p in got] == ["hq.nii.gz"]


def test_hq_side_of_an_explicit_restoration_pair_is_the_target(tmp_path):
    # The restoration naming inverts: the HQ source of a degradation is the TARGET.
    pairs = [("/hq/3t.nii.gz", "/lq/64mt.nii.gz")]
    got = hq_source_paths(
        pair_layout="explicit",
        field_pairs=pairs,
        hq_manifest=None,
        source_field_t=None,
        contrast=None,
    )
    assert got == ["/hq/3t.nii.gz"]


def test_field_keyed_pairs_take_the_source_side(tmp_path):
    pairs = [("/src/3t.nii.gz", "/dst/01t.nii.gz")]
    got = hq_source_paths(
        pair_layout="field_keyed",
        field_pairs=pairs,
        hq_manifest=None,
        source_field_t=None,
        contrast=None,
    )
    assert got == ["/src/3t.nii.gz"]


def test_no_manifest_and_no_pairs_is_undeclared_not_empty():
    assert (
        hq_source_paths(
            pair_layout=None,
            field_pairs=None,
            hq_manifest=None,
            source_field_t=None,
            contrast=None,
        )
        is None
    )


def test_nifti_spacing_comes_from_the_affine(tmp_path):
    from spectramr.data.reference_volume import load_reference_volume

    path = _nifti(tmp_path / "v.nii.gz", zooms=(0.5, 0.5, 2.5))
    assert ref_spacing_mm(path, load_reference_volume(path)) == pytest.approx((2.5, 0.5, 0.5))


def test_spacing_raises_when_a_volume_has_neither_affine_nor_header():
    with pytest.raises(ValueError, match="no voxel spacing"):
        ref_spacing_mm("/x.h5", {"data": None, "header": None, "spacing_mm": None})


def test_nifti_acquisition_is_read_from_the_sidecar(tmp_path):
    path = _nifti(tmp_path / "v.nii.gz")
    (tmp_path / "v.json").write_text(
        json.dumps({"MagneticFieldStrength": 0.064, "RepetitionTime": 1.2})
    )
    acq = acquisition_of_path(path)
    assert acq.field_strength_t == pytest.approx(0.064)
    assert acq.tr_ms == pytest.approx(1200.0)


def test_nifti_without_a_sidecar_raises_rather_than_defaulting(tmp_path):
    path = _nifti(tmp_path / "v.nii.gz")
    with pytest.raises(FileNotFoundError, match="acquisition_prior: false"):
        acquisition_of_path(path)


def test_nifti_sidecar_without_field_strength_raises(tmp_path):
    path = _nifti(tmp_path / "v.nii.gz")
    (tmp_path / "v.json").write_text(json.dumps({"RepetitionTime": 1.2}))
    with pytest.raises(MissingAcquisitionParameter):
        acquisition_of_path(path)


def test_split_pairs_is_seeded_disjoint_and_leaves_both_sides_nonempty():
    from spectramr.infrastructure.training.strategies.qm_selection import split_pairs

    pairs = [(f"/hq/{i}.nii.gz", f"/lq/{i}.nii.gz") for i in range(8)]
    fit_a, held_a = split_pairs(pairs, 0.25, seed=3)
    fit_b, held_b = split_pairs(pairs, 0.25, seed=3)
    assert (fit_a, held_a) == (fit_b, held_b)  # reproducible
    assert len(held_a) == 2 and len(fit_a) == 6
    assert not set(fit_a) & set(held_a)


def test_split_pairs_raises_when_one_side_would_be_empty():
    # Planted: a fraction that leaves nothing to score (or nothing to fit) must refuse.
    from spectramr.infrastructure.training.strategies.qm_selection import split_pairs

    pairs = [("/hq/a.nii.gz", "/lq/a.nii.gz"), ("/hq/b.nii.gz", "/lq/b.nii.gz")]
    with pytest.raises(ValueError, match="both must be at least one"):
        split_pairs(pairs, 0.1, seed=0)  # rounds to zero held out
    with pytest.raises(ValueError, match="both must be at least one"):
        split_pairs(pairs, 0.9, seed=0)  # rounds to every pair held out


def test_split_pairs_rejects_a_fraction_outside_the_open_interval():
    from spectramr.infrastructure.training.strategies.qm_selection import split_pairs

    with pytest.raises(ValueError, match="must lie in"):
        split_pairs([("/a", "/b")], 1.0, seed=0)
