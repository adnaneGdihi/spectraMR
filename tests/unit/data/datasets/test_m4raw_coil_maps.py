"""``m4raw_coil_maps``: the one recipe, key and file format for precomputed coil maps.

Pinned: the key is the repetition group whatever repetition asks; a file that
disagrees with its consumer raises on every field the consumer declares; the
``estimate`` fallback reproduces the file, so a run before and after the maps
exist sees the same geometry; a miss under ``raise`` stops the load.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

from spectramr.data.datasets.m4raw_coil_maps import (
    MAP_FORMAT_VERSION,
    CoilMapMismatchError,
    calibration_tag,
    estimate_group_maps,
    group_map_path,
    group_repetition_files,
    load_coil_maps,
    map_key,
    map_path,
    save_coil_maps,
)
from spectramr.data.datasets.m4raw_coil_map_source import CoilMapSource
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.scan_coil_calibration import DEFAULT_CALIBRATION

_S, _C, _H, _W = 2, 4, 48, 48


def _reps(n: int = 3, *, phases=(0.0, 1.9, -2.4)) -> list[torch.Tensor]:
    """``n`` repetitions of one smooth 4-coil scan at distinct global phases."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, _H), np.linspace(-1, 1, _W), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = np.stack([np.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])
    anatomy = ((xx**2 + yy**2) < 0.5) * 20.0
    out = []
    for i in range(n):
        rng = np.random.default_rng(i)
        noise = rng.normal(size=(_S, _C, _H, _W)) + 1j * rng.normal(size=(_S, _C, _H, _W))
        image = sens[None] * anatomy * np.exp(1j * phases[i]) + noise
        k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1))), axes=(-2, -1))
        k = k / np.sqrt(_H * _W)
        k[..., -8:] = 0
        out.append(torch.from_numpy(k.astype(np.complex64)))
    return out


def _calibration(**override) -> dict:
    return {**DEFAULT_CALIBRATION, **override}


@pytest.fixture(scope="module")
def estimated() -> tuple[torch.Tensor, dict]:
    return estimate_group_maps(_reps(), _calibration())


def _write(tmp_path, maps, meta, name="2022061001_T1.pt"):
    return save_coil_maps(tmp_path / name, maps, meta)


class TestTheKey:
    @pytest.mark.parametrize(
        "file_id", ["2022061001_T101", "2022061001_T103", "2022061001_T102.h5"]
    )
    def test_every_repetition_of_a_group_finds_one_file(self, tmp_path, file_id) -> None:
        assert map_key(file_id) == "2022061001_T1"
        assert map_path(tmp_path, file_id) == tmp_path / "2022061001_T1.pt"

    def test_a_key_is_not_grouped_twice(self, tmp_path) -> None:
        """The planted script bug: map_path(key) strips the contrast's digits."""
        assert group_map_path(tmp_path, "2022061001_T1") == tmp_path / "2022061001_T1.pt"
        assert map_path(tmp_path, "2022061001_T1") != group_map_path(tmp_path, "2022061001_T1")

    def test_files_are_grouped_by_the_same_rule(self, tmp_path) -> None:
        paths = [tmp_path / f"{p}_{c}0{r}.h5" for p in ("1", "2") for c in ("T1",) for r in (2, 1)]
        groups = group_repetition_files(paths)
        assert list(groups) == ["1_T1", "2_T1"]
        assert [f.name for f in groups["1_T1"]] == ["1_T101.h5", "1_T102.h5"]


class TestTheRecipe:
    def test_maps_describe_every_slice_and_coil(self, estimated) -> None:
        maps, meta = estimated
        assert maps.shape == (_S, _C, _H, _W) and maps.dtype == torch.complex64
        assert meta["num_repetitions_averaged"] == 3
        assert meta["pe_support_width"] == _W - 8
        assert meta["format_version"] == MAP_FORMAT_VERSION

    def test_the_mean_is_phase_aligned_before_calibration(self) -> None:
        """Opposed global phases cancel a plain mean; the recipe must not see that."""
        reps = _reps(2, phases=(0.0, np.pi))
        maps, _ = estimate_group_maps(reps, _calibration())
        support = maps.abs().pow(2).sum(1) > 0.5
        assert support.float().mean() > 0.2

    def test_repetitions_of_different_shapes_are_refused(self) -> None:
        reps = _reps(2)
        with pytest.raises(ValueError, match="disagree on shape"):
            estimate_group_maps([reps[0], reps[1][:1]], _calibration())


class TestTheFileIsValidated:
    def test_round_trip(self, tmp_path, estimated) -> None:
        maps, meta = estimated
        path = _write(tmp_path, maps, meta)
        loaded, loaded_meta = load_coil_maps(
            path, expect_shape=maps.shape, calibration=_calibration()
        )
        torch.testing.assert_close(loaded, maps)
        assert loaded_meta == meta

    @pytest.mark.parametrize(
        "override",
        [{"acs_size": 32}, {"kernel_size": 5}, {"eigen_threshold": 0.9}],
        ids=["acs", "kernel", "threshold"],
    )
    def test_a_calibration_mismatch_raises(self, tmp_path, estimated, override) -> None:
        maps, meta = estimated
        path = _write(tmp_path, maps, meta)
        with pytest.raises(CoilMapMismatchError, match=next(iter(override))):
            load_coil_maps(path, expect_shape=maps.shape, calibration=_calibration(**override))

    @pytest.mark.parametrize(
        "shape",
        [(_S + 1, _C, _H, _W), (_S, _C + 4, _H, _W), (_S, _C, _H, _W + 16)],
        ids=["slices", "coils", "matrix"],
    )
    def test_a_shape_mismatch_raises(self, tmp_path, estimated, shape) -> None:
        maps, meta = estimated
        path = _write(tmp_path, maps, meta)
        with pytest.raises(CoilMapMismatchError, match="do not match"):
            load_coil_maps(path, expect_shape=shape, calibration=_calibration())

    def test_another_format_version_raises(self, tmp_path, estimated) -> None:
        maps, meta = estimated
        path = _write(tmp_path, maps, {**meta, "format_version": 0})
        with pytest.raises(CoilMapMismatchError, match="format_version"):
            load_coil_maps(path, expect_shape=maps.shape, calibration=_calibration())

    def test_one_slice_of_a_volume_file(self, tmp_path, estimated) -> None:
        maps, meta = estimated
        path = _write(tmp_path, maps, meta)
        one, _ = load_coil_maps(
            path, expect_shape=(1, _C, _H, _W), calibration=_calibration(), slice_index=1
        )
        torch.testing.assert_close(one, maps[1:2])
        with pytest.raises(CoilMapMismatchError, match="outside"):
            load_coil_maps(
                path, expect_shape=(1, _C, _H, _W), calibration=_calibration(), slice_index=_S
            )

    def test_one_slice_owns_only_its_own_storage(self, tmp_path, estimated) -> None:
        """A slice record's maps cross the loader's shared memory; a view of the
        file's tensor would carry every slice with it (S x the bytes, #2673)."""
        maps, meta = estimated
        path = _write(tmp_path, maps, meta)
        one, _ = load_coil_maps(
            path, expect_shape=(1, _C, _H, _W), calibration=_calibration(), slice_index=1
        )
        slice_bytes = _C * _H * _W * one.element_size()
        assert one.untyped_storage().nbytes() == slice_bytes
        # The planted shape: the view the loader used to return.
        planted = torch.load(path, map_location="cpu", weights_only=True)["maps"][1:2]
        assert planted.untyped_storage().nbytes() == _S * slice_bytes


class TestTheSource:
    def test_a_file_is_read_not_estimated(self, tmp_path, estimated, monkeypatch) -> None:
        import spectramr.data.datasets.m4raw_coil_maps as store

        maps, meta = estimated
        _write(tmp_path, maps, meta)
        monkeypatch.setattr(store, "estimate_group_maps", lambda *a, **k: pytest.fail("estimated"))
        got, source = CoilMapSource(tmp_path, "raise", _calibration()).maps_for(
            "2022061001_T102", _reps()
        )
        torch.testing.assert_close(got, maps)
        assert source == f"file:{tmp_path / '2022061001_T1.pt'}"

    def test_a_miss_raises_by_default(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="precompute_m4raw_coil_maps"):
            CoilMapSource(tmp_path, "raise", _calibration()).maps_for("2022061001_T101", _reps())

    def test_estimate_reproduces_the_file_and_reports_once(
        self, tmp_path, estimated, caplog
    ) -> None:
        source = CoilMapSource(tmp_path, "estimate", _calibration())
        with caplog.at_level(logging.WARNING):
            first, tag = source.maps_for("2022061001_T101", _reps())
            source.maps_for("2022061001_T101", _reps())
        assert tag == "estimated"
        torch.testing.assert_close(first, estimated[0])
        assert sum("coil maps absent" in r.message for r in caplog.records) == 1

    def test_a_mismatched_file_raises_even_under_estimate(self, tmp_path, estimated) -> None:
        maps, meta = estimated
        _write(tmp_path, maps, {**meta, "acs_size": 16})
        with pytest.raises(CoilMapMismatchError):
            CoilMapSource(tmp_path, "estimate", _calibration()).maps_for("2022061001_T101", _reps())

    def test_an_unknown_policy_is_refused(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="coil_maps_missing"):
            CoilMapSource(tmp_path, "recompute", _calibration())

    def test_the_calibration_tag_is_order_stable(self) -> None:
        shuffled = dict(reversed(list(_calibration().items())))
        assert calibration_tag(shuffled) == "acs_size=24;kernel_size=6;eigen_threshold=0.95"


class TestTheRecipeIsRecorded:
    def test_input_repetition_is_stamped_and_ignores_the_other_repetitions(self) -> None:
        reps = _reps()
        maps, meta = estimate_group_maps(reps, _calibration(), "input_repetition")
        assert meta["recipe"] == "input_repetition" and meta["num_repetitions_averaged"] == 1
        swapped, _ = estimate_group_maps(
            [reps[0], reps[2], reps[1]], _calibration(), "input_repetition"
        )
        torch.testing.assert_close(swapped, maps)

    def test_a_recipe_mismatch_raises(self, tmp_path, estimated) -> None:
        """Planted: an all-repetitions file served to an input_repetition consumer."""
        maps, meta = estimated
        path = _write(tmp_path, maps, meta)
        assert meta["recipe"] == "all_reps_mean"
        with pytest.raises(CoilMapMismatchError, match="recipe 'all_reps_mean'"):
            load_coil_maps(
                path, expect_shape=maps.shape, calibration=_calibration(), recipe="input_repetition"
            )

    def test_a_legacy_file_without_a_recipe_still_loads_as_the_mean(self, tmp_path, estimated):
        maps, meta = estimated
        legacy = {k: v for k, v in meta.items() if k != "recipe"}
        path = _write(tmp_path, maps, legacy)
        loaded, _ = load_coil_maps(path, expect_shape=maps.shape, calibration=_calibration())
        torch.testing.assert_close(loaded, maps)
        with pytest.raises(CoilMapMismatchError, match="recipe 'all_reps_mean'"):
            load_coil_maps(
                path, expect_shape=maps.shape, calibration=_calibration(), recipe="input_repetition"
            )

    def test_the_meta_check_reads_the_file_without_its_maps(self, tmp_path, estimated) -> None:
        from spectramr.data.datasets.m4raw_coil_maps import check_map_meta, read_map_meta

        maps, meta = estimated
        path = _write(tmp_path, maps, meta)
        stored = read_map_meta(path)
        check_map_meta(path, stored, calibration=_calibration(), recipe="all_reps_mean")
        with pytest.raises(CoilMapMismatchError, match="acs_size"):
            check_map_meta(
                path, stored, calibration=_calibration(acs_size=20), recipe="all_reps_mean"
            )


def test_the_input_repetition_store_equals_espirit_on_the_masked_input() -> None:
    """The probe: with the ACS inside the mask, the store is the per-load estimate.

    ESPIRiT reads only its central ``acs_size`` block, so maps estimated once from
    the whole input repetition must equal the ``espirit_sensitivity`` transform
    run per load on the masked input, when the mask's ACS band covers that block.
    Non-square, so a height/width swap in either centring would show.
    """
    import torchio as tio

    from spectramr.core.metrics.coil_manifold import as_interleaved
    from spectramr.data.transforms.complementary_rep_masking import draw_rep_masks
    from spectramr.data.transforms.registry import build_transform

    h, w, acs = 40, 250, 20  # round(250 * 0.08) == 20, the corpus ACS band
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = np.stack([np.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])
    anatomy = ((xx**2 + yy**2) < 0.5) * 20.0
    reps = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=(1, _C, h, w)) + 1j * rng.normal(size=(1, _C, h, w))
        image = torch.from_numpy((sens[None] * anatomy + noise).astype(np.complex64))
        reps.append(fft2c(image))
    stored, _ = estimate_group_maps(reps, _calibration(acs_size=acs), "input_repetition")

    m1, _ = draw_rep_masks(
        torch.ones(w, dtype=torch.bool),
        acceleration=4.0,
        center_fraction=0.08,
        mode="same",
        keys=("probe", 0, 0),
    )
    assert int(m1.sum()) == round(w / 4) and int(m1.sum()) > acs
    assert m1[w // 2 - acs // 2 : w // 2 + acs // 2].all(), "the ACS block lies inside the mask"
    masked = reps[0][0] * m1.to(torch.complex64)  # (C, H, W), lines along dim -1
    subject = tio.Subject(input=tio.ScalarImage(tensor=as_interleaved(masked[None])[0, ..., None]))
    per_load = build_transform("espirit_sensitivity", acs_size=acs, keys=["input"])(subject)
    per_load_maps = per_load["sensitivity_complex"].data.permute(3, 0, 1, 2)
    torch.testing.assert_close(per_load_maps, stored, rtol=1e-4, atol=1e-5)
    # Planted contrast: the all-repetitions store is a different estimate, so the
    # comparison above can fail.
    pooled, _ = estimate_group_maps(reps, _calibration(acs_size=acs), "all_reps_mean")
    assert not torch.allclose(per_load_maps, pooled, rtol=1e-4, atol=1e-5)
