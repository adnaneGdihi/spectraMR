"""Tests for the quality-descriptor SSOT."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from spectramr.core.metrics.registry import get_metric
from spectramr.infrastructure.physics.quality_descriptors import (
    QualityDescriptor,
    UnmeasurableAttributeError,
    measure,
    measure_attributes,
    read_spacing_mm,
    resample_to_spacing,
    validate_attributes,
)

HEADER = """<?xml version="1.0"?>
<ismrmrdHeader>
  <encoding>
    <reconSpace>
      <matrixSize><x>64</x><y>64</y><z>1</z></matrixSize>
      <fieldOfView_mm><x>256.0</x><y>256.0</y><z>5.0</z></fieldOfView_mm>
    </reconSpace>
  </encoding>
</ismrmrdHeader>"""


def test_validate_attributes_rejects_unregistered_name():
    with pytest.raises(UnmeasurableAttributeError, match="not a registered metric"):
        validate_attributes(["definitely_not_a_metric"])


def test_validate_attributes_rejects_full_reference_metric():
    # psnr needs a reference; a target descriptor is measured from ONE volume with no
    # ground truth, so a full-reference metric can never be matched.
    with pytest.raises(UnmeasurableAttributeError, match="requires a reference"):
        validate_attributes(["psnr"])


def test_validate_attributes_accepts_no_reference_metric():
    validate_attributes(["tenengrad_variance"])  # must not raise


def test_measure_reads_spacing_from_header_not_array_size():
    # 64x64 array, FOV 256 mm -> 4.0 mm in-plane. An implementation dividing FOV by
    # the size of a CROPPED array would inflate the voxel.
    vol = torch.rand(1, 64, 64)
    d = measure(vol, HEADER, attributes=["tenengrad_variance"])
    assert d.spacing_mm[1] == pytest.approx(4.0)
    assert d.spacing_mm[2] == pytest.approx(4.0)
    assert d.spacing_mm[0] == pytest.approx(5.0)


def test_measure_populates_every_requested_attribute():
    vol = torch.rand(1, 64, 64)
    d = measure(vol, HEADER, attributes=["tenengrad_variance", "laplacian_variance"])
    assert set(d.attributes) == {"tenengrad_variance", "laplacian_variance"}
    assert all(isinstance(v, float) for v in d.attributes.values())


def _phantom(n_slices: int = 1, size: int = 64) -> torch.Tensor:
    """A STRUCTURED phantom: nested blocks with real edges, plus a little noise.

    Deliberately not ``torch.rand``. A uniform-random volume is not merely a weak
    phantom for an edge-based statistic, it is an actively misleading one: with no
    dominant edge, added noise RAISES Tenengrad variance and the metric appears
    non-monotone in quality. Every sharpness assertion here must run on structure.
    """
    torch.manual_seed(0)
    v = torch.zeros(n_slices, size, size)
    v[:, size // 5 : 4 * size // 5, size // 5 : 4 * size // 5] = 1.0
    v[:, 2 * size // 5 : 3 * size // 5, 2 * size // 5 : 3 * size // 5] = 0.55
    return v + 0.02 * torch.randn_like(v)


def test_measure_is_sensitive_to_blur():
    # A sharpness attribute MUST fall when the image is blurred. This catches a
    # descriptor wired to the wrong tensor: a shape-only assertion passes on a constant.
    sharp = _phantom()
    blurred = torch.nn.functional.avg_pool2d(
        sharp.unsqueeze(0), kernel_size=5, stride=1, padding=2
    ).squeeze(0)
    d_sharp = measure(sharp, HEADER, attributes=["tenengrad_variance"])
    d_blur = measure(blurred, HEADER, attributes=["tenengrad_variance"])
    assert d_blur.attributes["tenengrad_variance"] < d_sharp.attributes["tenengrad_variance"]


def test_sharpness_falls_under_noise_too_on_structured_data():
    """Both degradation directions must LOWER the attribute, or the fit is ill-posed.

    If noise raised Tenengrad while blur lowered it, a fit could satisfy a sharpness
    target by adding noise -- trading one degradation for another and reporting a
    perfect residual. On structure both fall, so the target is approached from one
    side. On uniform-random 'anatomy' this does NOT hold, which is exactly why that
    substrate is banned above.
    """
    from spectramr.infrastructure.physics.degradation_chain import (
        ChainLink,
        DegradationChain,
    )

    clean = _phantom()
    noisy = (
        DegradationChain(links=(ChainLink(axis="complex_gaussian", theta=0.9),))
        .apply(clean.unsqueeze(1), seed=1)
        .squeeze(1)
        .abs()
    )
    d_clean = measure(clean, HEADER, attributes=["tenengrad_variance"])
    d_noisy = measure(noisy, HEADER, attributes=["tenengrad_variance"])
    assert d_noisy.attributes["tenengrad_variance"] < d_clean.attributes["tenengrad_variance"]


def test_vector_orders_by_requested_keys():
    d = QualityDescriptor(spacing_mm=(5.0, 4.0, 4.0), attributes={"a": 1.0, "b": 2.0})
    assert d.vector(["b", "a"]) == (2.0, 1.0)


def test_descriptor_is_frozen():
    d = QualityDescriptor(spacing_mm=(5.0, 4.0, 4.0), attributes={"a": 1.0})
    with pytest.raises(FrozenInstanceError):
        d.spacing_mm = (1.0, 1.0, 1.0)  # type: ignore[misc]


def test_measure_accepts_a_2d_slice():
    d = measure(torch.rand(64, 64), HEADER, attributes=["tenengrad_variance"])
    assert d.spacing_mm[1] == pytest.approx(4.0)


def test_measure_attributes_agrees_with_measure():
    """The inner-loop path must not drift from the full path.

    measure_attributes skips geometry so the fitter avoids hundreds of redundant XML
    parses; if the two ever disagreed, the fit would optimise a different quantity
    than the one finally reported.
    """
    torch.manual_seed(0)
    vol = torch.rand(2, 64, 64)
    attrs = ["tenengrad_variance", "laplacian_variance"]
    assert (
        measure_attributes(vol, attributes=attrs)
        == measure(vol, HEADER, attributes=attrs).attributes
    )


def test_measure_attributes_needs_no_header():
    # Degradation does not change voxel spacing, so the inner loop must not require
    # a header at all.
    vals = measure_attributes(torch.rand(1, 32, 32), attributes=["tenengrad_variance"])
    assert set(vals) == {"tenengrad_variance"}


def test_measure_attributes_validates_too():
    with pytest.raises(UnmeasurableAttributeError, match="requires a reference"):
        measure_attributes(torch.rand(1, 32, 32), attributes=["psnr"])


# ── geometry: the IMPOSED half of a quality match ─────────────────────


def test_read_spacing_mm_uses_the_header_pair():
    got = read_spacing_mm(HEADER, torch.rand(1, 64, 64))
    assert got == pytest.approx((5.0, 4.0, 4.0))


def test_read_spacing_mm_raises_on_a_missing_header():
    # An assumed 1 mm default would resample every volume onto the wrong grid.
    from spectramr.data.nifti_export import GeometryUnavailableError

    with pytest.raises(GeometryUnavailableError):
        read_spacing_mm(None, torch.rand(1, 64, 64))


def test_resample_to_coarser_spacing_returns_to_the_source_grid():
    # A paired restoration set needs input and target on the SAME grid.
    vol = torch.rand(2, 64, 64)
    out = resample_to_spacing(vol, (5.0, 4.0, 4.0), (5.0, 8.0, 8.0))
    assert out.shape == vol.shape


def test_resample_to_coarser_spacing_actually_loses_resolution():
    """The round trip must destroy high-frequency content, not merely reshape it.

    A resample that returned the array unchanged would pass a shape assertion while
    imposing no resolution at all -- the geometric equivalent of an inert mechanism.
    """
    torch.manual_seed(0)
    vol = torch.rand(1, 64, 64)
    out = resample_to_spacing(vol, (5.0, 1.0, 1.0), (5.0, 4.0, 4.0))
    sharp = get_metric("tenengrad_variance")(vol.unsqueeze(1))
    blurred = get_metric("tenengrad_variance")(out.unsqueeze(1))
    assert blurred < sharp


def test_resample_is_a_noop_when_spacings_match():
    vol = torch.rand(2, 32, 32)
    out = resample_to_spacing(vol, (5.0, 1.0, 1.0), (5.0, 1.0, 1.0))
    assert torch.equal(out, vol)


def test_resample_without_returning_gives_the_coarse_grid():
    vol = torch.rand(1, 64, 64)
    out = resample_to_spacing(vol, (5.0, 1.0, 1.0), (5.0, 2.0, 2.0), return_to_source_grid=False)
    assert out.shape == (1, 32, 32)


@pytest.mark.parametrize("bad", [(5.0, 0.0, 1.0), (5.0, 1.0, -1.0)])
def test_resample_rejects_nonpositive_spacing(bad):
    with pytest.raises(ValueError, match="positive"):
        resample_to_spacing(torch.rand(1, 16, 16), (5.0, 1.0, 1.0), bad)


# ── through-plane: averaging down is a measurement, up is invention ───


def test_thicker_target_slice_averages_through_plane():
    """A 2x thicker slice IS the average of two thin ones. That is partial volume."""
    vol = torch.rand(8, 16, 16)
    out = resample_to_spacing(vol, (2.0, 1.0, 1.0), (4.0, 1.0, 1.0), return_to_source_grid=False)
    assert out.shape == (4, 16, 16)


def test_through_plane_averaging_destroys_slice_to_slice_detail():
    """The loss must be real, not a reshape.

    Alternating bright/dark slices average to a uniform stack: exactly what a slab
    twice as thick would measure. A test asserting only the shape would pass on a
    no-op.
    """
    vol = torch.zeros(8, 8, 8)
    vol[::2] = 1.0  # alternating slices
    out = resample_to_spacing(vol, (2.0, 1.0, 1.0), (4.0, 1.0, 1.0), return_to_source_grid=False)
    # Each output slice averages one bright and one dark slice -> ~0.5 everywhere.
    assert out.std().item() < 1e-5
    assert out.mean().item() == pytest.approx(0.5, abs=1e-5)


def test_thicker_slice_round_trip_preserves_the_pair_grid():
    vol = torch.rand(8, 16, 16)
    out = resample_to_spacing(vol, (2.0, 1.0, 1.0), (4.0, 1.0, 1.0))
    assert out.shape == vol.shape


@pytest.mark.parametrize(
    ("src", "dst", "axis"),
    [
        ((5.0, 1.0, 1.0), (2.0, 1.0, 1.0), "slice"),
        ((5.0, 2.0, 1.0), (5.0, 1.0, 1.0), "row"),
        ((5.0, 1.0, 2.0), (5.0, 1.0, 1.0), "col"),
    ],
)
def test_finer_target_raises_on_every_axis(src, dst, axis):
    """Interpolating UP invents resolution that was never acquired.

    A synthetic 'low-quality' volume carrying invented detail is worse than useless:
    the restorer would learn to reproduce an artefact of the simulation.
    """
    with pytest.raises(ValueError, match="FINER than the source"):
        resample_to_spacing(torch.rand(4, 16, 16), src, dst)
    # and the message must name which axis, so the fix is obvious
    with pytest.raises(ValueError, match=axis):
        resample_to_spacing(torch.rand(4, 16, 16), src, dst)


def test_all_three_axes_can_coarsen_together():
    vol = torch.rand(8, 32, 32)
    out = resample_to_spacing(vol, (2.0, 1.0, 1.0), (4.0, 2.0, 2.0), return_to_source_grid=False)
    assert out.shape == (4, 16, 16)


def test_intensity_normalisation_makes_attributes_unit_invariant():
    # Planted: without a shared unit the same anatomy at 1e-2 and 1e2 scores differently.
    from spectramr.infrastructure.physics.quality_descriptors import normalise_intensity

    x = _phantom(n_slices=2, size=48)
    a = measure_attributes(normalise_intensity(x, 99.0), attributes=["tenengrad_variance"])
    for c in (1e-2, 100.0):
        b = measure_attributes(normalise_intensity(c * x, 99.0), attributes=["tenengrad_variance"])
        assert b["tenengrad_variance"] == pytest.approx(a["tenengrad_variance"], rel=1e-5)


def test_robust_scale_raises_on_a_volume_with_no_signal():
    from spectramr.infrastructure.physics.quality_descriptors import robust_scale

    with pytest.raises(ValueError, match="no signal"):
        robust_scale(torch.zeros(2, 8, 8), 99.0)


def test_robust_scale_ignores_a_single_hot_voxel():
    from spectramr.infrastructure.physics.quality_descriptors import robust_scale

    x = torch.ones(1, 100, 100)
    x[0, 0, 0] = 1e6
    assert robust_scale(x, 99.0) == pytest.approx(1.0)
