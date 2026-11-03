"""M4RawSource: centred slices, and the refusal to grade against a noisy reference."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from spectramr.data.sources.m4raw_source import M4RawSource  # noqa: E402


class TestSingletonGroupsAreRefused:
    """``discover_repetition_groups`` falls back to SINGLETON groups when it finds
    no multi-rep ones. A 1-rep "pseudo-GT" is just the noisy acquisition itself --
    no sqrt(N) averaging, so the "clean" reference carries the same noise as the
    input, and every full-reference metric is graded against noise."""

    def test_a_singleton_only_tree_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "spectramr.data.sources.m4raw_source.discover_repetition_groups",
            lambda *_a, **_k: [[Path("a01.h5")], [Path("b01.h5")]],
        )
        with pytest.raises(FileNotFoundError, match="graded against noise"):
            M4RawSource(tmp_path)

    def test_singletons_are_dropped_and_counted(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "spectramr.data.sources.m4raw_source.discover_repetition_groups",
            lambda *_a, **_k: [
                [Path("2022_T101.h5"), Path("2022_T102.h5")],
                [Path("solo_T101.h5")],
            ],
        )
        src = M4RawSource(tmp_path)
        assert src.dropped_singletons == 1
        assert len(src._groups) == 1


class TestSliceSelection:
    @pytest.fixture
    def src(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "spectramr.data.sources.m4raw_source.discover_repetition_groups",
            lambda *_a, **_k: [[Path("2022_T101.h5"), Path("2022_T102.h5")]],
        )
        return M4RawSource(tmp_path, slices_per_volume=5)

    def test_slices_are_centred_and_absolute(self, src) -> None:
        """A negative OFFSET like -2 would index from the END of the volume, quietly
        sampling the outermost slices -- which are mostly air, where every metric
        agrees, diluting every leaderboard cell with a region that has nothing in it."""
        assert src._slice_indices(20) == [7, 8, 9, 10, 11]
        assert all(i >= 0 for i in src._slice_indices(20))

    def test_a_short_volume_is_not_over_sampled(self, src) -> None:
        assert src._slice_indices(3) == [0, 1, 2]

    def test_the_group_id_ignores_the_repetition_number(self, src) -> None:
        """content_id is the ranker's covariate C -- an id that changed with which
        rep happened to sort first would look like a different subject."""
        a = M4RawSource._group_id([Path("2022091411_T101.h5"), Path("2022091411_T102.h5")])
        b = M4RawSource._group_id([Path("2022091411_T102.h5"), Path("2022091411_T101.h5")])
        assert a == b == "2022091411_T1"

    def test_every_contrast_of_a_scan_is_one_subject(self, src) -> None:
        assert M4RawSource._subject_id("2022091411_T1") == "2022091411"
        assert M4RawSource._subject_id("2022091411_FLAIR") == "2022091411"


class TestNoPathologyRegions:
    def test_m4raw_never_emits_lesion_regions(self, tmp_path, monkeypatch) -> None:
        """M4Raw carries no annotations. It is the CONTROL cohort for the region
        axis -- if the leaderboard does not move between full and brain here, the
        region machinery has a problem that has nothing to do with lesions."""
        monkeypatch.setattr(
            "spectramr.data.sources.m4raw_source.discover_repetition_groups",
            lambda *_a, **_k: [[Path("2022_T101.h5"), Path("2022_T102.h5")]],
        )
        src = M4RawSource(tmp_path)
        masks, _ = src._regions("2022_T1", 0, 16, 16)
        assert set(masks) == {"full"}
        assert not any(k.startswith(("path:", "control:")) for k in masks)


# ──────────────────────────────────────────────────────────────────────
# __iter__ -- the only code path that touches real data, and the one path no test
# exercised. The source unpacked synthesize_pseudo_gt's 4-tuple in the WRONG ORDER
# (`clean_c, kspace, smaps, _` for a function returning
# `(coil_images, smaps, x_gt_mag, p99)`), so every field was off by one and M4Raw --
# the control cohort -- could not yield a single slice. The 33 existing tests all
# passed: they monkeypatch the path list and never open an HDF5.
# ──────────────────────────────────────────────────────────────────────


def _write_m4raw_h5(path, *, n_slices: int = 3, n_coils: int = 4, hw: int = 32) -> None:
    """A minimal M4Raw-shaped file: kspace (S, C, H, W) complex."""
    import h5py
    import numpy as np

    rng = np.random.default_rng(0)
    k = (
        rng.standard_normal((n_slices, n_coils, hw, hw))
        + 1j * rng.standard_normal((n_slices, n_coils, hw, hw))
    ).astype(np.complex64)
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=k)


def test_iter_yields_a_well_formed_slice(tmp_path) -> None:
    """The reference must be [1, H, W] magnitude -- not the (1, C, H, W) coil stack."""
    from spectramr.data.sources.m4raw_source import M4RawSource

    for rep in ("01", "02"):
        _write_m4raw_h5(tmp_path / f"2022101301_FLAIR{rep}.h5")

    src = M4RawSource(tmp_path, slices_per_volume=1, max_subjects=1, max_contrasts=1)
    sl = next(iter(src))

    assert sl.clean.shape == (1, 32, 32), (
        f"clean is {tuple(sl.clean.shape)}; the magnitude reference must be [1, H, W]. "
        "A (1, C, H, W) coil stack here means the pseudo-GT tuple was unpacked in the "
        "wrong order."
    )
    assert not sl.clean.is_complex()
    assert sl.kspace is not None and sl.kspace.is_complex(), (
        "kspace must be the MEASUREMENT (complex k-space). coil_images is image-domain; "
        "handing it over unchanged would give M4Raw and fastMRI different domains under "
        "the same field name."
    )
    assert sl.provenance["normalisation"]["mode"] == "p99"


class TestContrastsUseTheOneParser:
    def test_t1_and_t2_are_distinct_and_unparseable_names_raise(self, tmp_path, monkeypatch) -> None:
        groups = [
            [Path("2022_T101.h5"), Path("2022_T102.h5")],
            [Path("2022_T201.h5"), Path("2022_T202.h5")],
        ]
        monkeypatch.setattr(
            "spectramr.data.sources.m4raw_source.discover_repetition_groups",
            lambda *_a, **_k: groups,
        )
        assert M4RawSource(tmp_path).contrasts == ("T1", "T2")
        groups[:] = [[Path("mystery01.h5"), Path("mystery02.h5")]]
        with pytest.raises(ValueError, match="not an M4Raw"):
            _ = M4RawSource(tmp_path).contrasts
