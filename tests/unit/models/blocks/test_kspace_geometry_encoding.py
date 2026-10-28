"""The k-space geometry planes: values at known positions, and the fft2c centring."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.kspace_geometry_encoding import (
    KSPACE_GEOMETRY_CHANNELS,
    KSPACE_GEOMETRY_PLANES,
    as_interleaved_complex,
    kspace_geometry_channels,
    mask_plane,
    measurement_support,
    model_consumes_acquired_plane,
    nearest_known_column_distance,
    resolve_geometry_planes,
    support_forward_kwargs,
)
from spectramr.models.generators.kspace_cold_diffusion_generator import KSpaceCrop

pytestmark = pytest.mark.unit

H, W = 16, 32


def _columns(*cols: int, batch: int = 1) -> torch.Tensor:
    """A ``[B, 1, H, W]`` mask acquiring whole phase-encode columns."""
    mask = torch.zeros(batch, 1, H, W)
    mask[..., list(cols)] = 1.0
    return mask


def _plane(name: str, planes: torch.Tensor) -> torch.Tensor:
    return planes[:, KSPACE_GEOMETRY_PLANES.index(name)]


def test_the_plane_count_matches_the_declared_names() -> None:
    planes = kspace_geometry_channels(_columns(W // 2), None, (H, W), batch=1)
    assert planes.shape == (1, KSPACE_GEOMETRY_CHANNELS, H, W)


def test_support_and_acquired_planes_are_the_masks() -> None:
    support, acquired = _columns(3, 16), _columns(*range(2, 30))
    planes = kspace_geometry_channels(support, acquired, (H, W), batch=1)
    torch.testing.assert_close(_plane("support", planes), support[:, 0])
    torch.testing.assert_close(_plane("acquired", planes), acquired[:, 0])


def test_an_undeclared_acquisition_is_fully_acquirable() -> None:
    planes = kspace_geometry_channels(_columns(16), None, (H, W), batch=2)
    assert torch.equal(_plane("acquired", planes), torch.ones(2, H, W))


def test_coordinates_are_zero_at_the_fft2c_centre() -> None:
    planes = kspace_geometry_channels(_columns(16), None, (H, W), batch=1)
    ky, kx, radius = (_plane(n, planes)[0] for n in ("ky", "kx", "radius"))
    assert float(ky[:, W // 2].abs().max()) == 0.0
    assert float(kx[H // 2].abs().max()) == 0.0
    assert float(ky[0, 0]) == -1.0 and float(kx[0, 0]) == -1.0
    assert float(radius[H // 2, W // 2]) == 0.0
    assert float(radius.max()) == pytest.approx(1.0)
    # ky varies only along W (phase encode), kx only along H (readout).
    assert torch.equal(ky, ky[:1].expand_as(ky)) and torch.equal(kx, kx[:, :1].expand_as(kx))


def test_the_dc_column_is_at_distance_zero_when_acquired() -> None:
    planes = kspace_geometry_channels(_columns(W // 2), None, (H, W), batch=1)
    assert float(_plane("pe_gap", planes)[0, :, W // 2].abs().max()) == 0.0


def test_distance_to_the_nearest_known_column_on_a_hand_built_mask() -> None:
    known = torch.zeros(1, 10, dtype=torch.bool)
    known[0, [2, 7]] = True
    expected = torch.tensor([[2, 1, 0, 1, 2, 2, 1, 0, 1, 2]], dtype=torch.float32) / 10
    torch.testing.assert_close(nearest_known_column_distance(known), expected)


def test_a_sample_with_no_known_column_is_at_distance_one() -> None:
    known = torch.zeros(2, 8, dtype=torch.bool)
    known[1, 0] = True
    dist = nearest_known_column_distance(known)
    assert torch.equal(dist[0], torch.ones(8))
    torch.testing.assert_close(dist[1], torch.arange(8, dtype=torch.float32) / 8)


def test_a_multichannel_mask_counts_a_bin_known_in_any_channel() -> None:
    mask = torch.zeros(1, 3, H, W)
    mask[:, 2, :, 5] = 1.0
    assert torch.equal(mask_plane(mask, 1, (H, W))[0, 0, :, 5], torch.ones(H))


def test_a_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="geometry mask"):
        kspace_geometry_channels(torch.ones(1, 1, H, W + 2), None, (H, W), batch=1)


def test_interleaving_puts_each_plane_on_a_real_part() -> None:
    planes = torch.randn(2, 3, 4, 4)
    pairs = as_interleaved_complex(planes)
    assert pairs.shape == (2, 6, 4, 4)
    assert torch.equal(pairs[:, 0::2], planes)
    assert not pairs[:, 1::2].any()


@pytest.mark.parametrize("size", [16, 32, 64])
def test_the_centre_survives_every_unet_crop(size: int) -> None:
    """``KSpaceCrop`` keeps DC at ``N // 2`` on each level, so a plane cropped with its
    arithmetic is still zero there and is an affine rescale of the level's own plane."""
    planes = kspace_geometry_channels(torch.ones(1, 1, size, size), None, (size, size), batch=1)
    crop = KSpaceCrop(scale_factor=2)
    level = planes
    while level.shape[-1] >= 4:
        level = crop(level) * crop.scale_factor  # undo the amplitude rescale
        n = level.shape[-1]
        own = kspace_geometry_channels(torch.ones(1, 1, n, n), None, (n, n), batch=1)
        ky, own_ky = _plane("ky", level)[0], _plane("ky", own)[0]
        assert float(ky[:, n // 2].abs().max()) == 0.0
        torch.testing.assert_close(ky, own_ky * (ky[0, 0] / own_ky[0, 0]))


def test_the_consumer_predicate_reads_the_built_width() -> None:
    from types import SimpleNamespace

    assert model_consumes_acquired_plane(SimpleNamespace(kspace_geometry_channels=6))
    assert not model_consumes_acquired_plane(SimpleNamespace(kspace_geometry_channels=0))
    wrapped = SimpleNamespace(module=SimpleNamespace(kspace_geometry_channels=6))
    assert model_consumes_acquired_plane(wrapped)
    assert not model_consumes_acquired_plane(object())


class TestMeasurementSupport:
    def test_a_bin_is_observed_when_any_channel_is_non_zero(self):
        y = torch.zeros(2, 4, H, W)
        y[0, 3, :, 5] = 1.0
        y[1, :, :, 7] = -2.0
        support = measurement_support(y)
        assert support.shape == (2, 1, H, W) and support.dtype == torch.float32
        torch.testing.assert_close(support[0], _columns(5)[0])
        torch.testing.assert_close(support[1], _columns(7)[0])

    def test_complex_measurements_use_the_magnitude(self):
        y = torch.zeros(1, 2, H, W, dtype=torch.complex64)
        y[..., 9] = 1j
        torch.testing.assert_close(measurement_support(y), _columns(9))

    def test_the_mask_reaches_only_a_geometry_plane_model(self):
        y = torch.zeros(1, 2, H, W)
        y[..., 3] = 1.0

        class _Geo:
            kspace_geometry_channels = KSPACE_GEOMETRY_CHANNELS

        wrapped = type("W", (), {"module": _Geo()})()
        assert support_forward_kwargs(object(), y) == {}
        for model in (_Geo(), wrapped):
            torch.testing.assert_close(support_forward_kwargs(model, y)["mask"], _columns(3))


def test_a_plane_selection_returns_only_those_planes_in_order() -> None:
    full = kspace_geometry_channels(_columns(3, 16), None, (H, W), batch=2)
    chosen = ("pe_gap", "support")
    part = kspace_geometry_channels(_columns(3, 16), None, (H, W), batch=2, planes=chosen)
    assert part.shape == (2, 2, H, W)
    torch.testing.assert_close(part[:, 0], _plane("pe_gap", full))
    torch.testing.assert_close(part[:, 1], _plane("support", full))


def test_an_empty_selection_builds_no_planes() -> None:
    assert resolve_geometry_planes([]) == ()
    assert kspace_geometry_channels(_columns(3), None, (H, W), batch=1, planes=()).shape == (
        1,
        0,
        H,
        W,
    )


def test_no_declaration_selects_every_plane() -> None:
    assert resolve_geometry_planes(None) == KSPACE_GEOMETRY_PLANES


@pytest.mark.parametrize("bad", ["ky", ["ky", "ky"], ["kz"], 3])
def test_a_bad_selection_raises(bad: object) -> None:
    with pytest.raises(ValueError, match="kspace_geometry_planes"):
        resolve_geometry_planes(bad)
