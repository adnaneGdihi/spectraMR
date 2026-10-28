"""Sim2RankSlice: the invariants that keep a region attached to the right pixels."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.data.sources.sim2rank_source import (  # noqa: E402
    DEFAULT_SYNTHETIC_VOLUME_HWD,
    Sim2RankSlice,
    Sim2RankVolume,
    Sim2RankVolumeSource,
    synthetic_phantom_volume,
)


def _slice(**kw) -> Sim2RankSlice:
    defaults = {
        "content_id": "f#4",
        "clean": torch.rand(1, 32, 32),
        "contrast": "AXT2",
        "file_id": "f",
        "slice_index": 4,
    }
    return Sim2RankSlice(**{**defaults, **kw})


class TestSliceWiseInvariant:
    def test_a_volume_is_rejected(self) -> None:
        """The digital-twin simulator is slice-wise. A [S, H, W] tensor here would
        be silently degraded as a BATCH of independent slices -- each getting its
        own motion trajectory, its own noise draw -- which is not what a volume is."""
        with pytest.raises(ValueError, match=r"must be \[1, H, W\]"):
            _slice(clean=torch.rand(16, 32, 32))

    def test_a_bare_2d_slice_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be \[1, H, W\]"):
            _slice(clean=torch.rand(32, 32))


class TestRegionGeometryIsChecked:
    def test_a_mask_from_another_geometry_is_rejected(self) -> None:
        """Regions are computed on the clean reference. A geometry mismatch means
        the mask belongs to a different slice, and resampling it would move the
        region boundary."""
        with pytest.raises(ValueError, match="another geometry"):
            _slice(region_masks={"full": torch.ones(64, 64, dtype=torch.bool)})

    def test_a_float_mask_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="bool mask"):
            _slice(region_masks={"full": torch.ones(32, 32)})

    def test_matching_masks_are_accepted(self) -> None:
        s = _slice(
            region_masks={
                "full": torch.ones(32, 32, dtype=torch.bool),
                "path:lesion_any": torch.zeros(32, 32, dtype=torch.bool).index_fill_(
                    0, torch.tensor([1, 2]), True
                ),
            }
        )
        assert set(s.region_masks) == {"full", "path:lesion_any"}


class TestProtocol:
    def test_a_minimal_source_satisfies_the_protocol(self) -> None:
        class Fake:
            name = "fake"
            contrasts = ("AXT2",)

            def __iter__(self):
                yield _slice()

            def __len__(self) -> int:
                return 1

        assert isinstance(Fake(), Sim2RankVolumeSource)
        assert next(iter(Fake())).contrast == "AXT2"


class TestSim2RankVolume:
    def _volume(self, clean):
        return Sim2RankVolume(content_id="v", clean=clean, contrast="T1")

    def test_layout_is_one_by_d_h_w(self) -> None:
        vol = self._volume(torch.rand(1, 4, 12, 10))
        assert vol.as_batch().shape == (1, 1, 4, 12, 10)
        assert vol.size_hwd == {"H": 12, "W": 10, "D": 4}

    @pytest.mark.parametrize("shape", [(4, 12, 10), (2, 4, 12, 10), (1, 1, 12, 10)])
    def test_bad_shapes_are_rejected(self, shape) -> None:
        with pytest.raises(ValueError):
            self._volume(torch.rand(*shape))

    def test_complex_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="real magnitude"):
            self._volume(torch.rand(1, 4, 12, 10, dtype=torch.complex64))

    def test_synthetic_phantom_defaults_and_depth_structure(self) -> None:
        assert DEFAULT_SYNTHETIC_VOLUME_HWD == (128, 128, 64)
        vol = synthetic_phantom_volume(height=16, width=12, depth=5)
        assert vol.clean.shape == (1, 5, 16, 12)
        assert bool((vol.clean[0].flatten(1).amax(dim=1) > 0).all())
        assert not torch.equal(vol.clean[:, 0], vol.clean[:, 2])
