"""Physics of the analytic 4D-flow phantom, checked against closed forms."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.signal_models.flow_phantom import (
    FlowPhantomSpec,
    draw_flow_phantom,
    phantom_slice,
)

SPEC = FlowPhantomSpec(
    grid=(24, 40, 36),
    voxel_size_mm=(1.0, 0.8, 0.8),
    frames=5,
    coils=3,
    vessels=2,
    peak_velocity=1.2,
    background_order=1,
    background_amplitude=0.3,
)


def _phantom(seed: int = 3):
    return draw_flow_phantom(SPEC, torch.Generator().manual_seed(seed))


def test_slice_fields_have_the_documented_layout() -> None:
    fields = phantom_slice(SPEC, _phantom(), 5)
    assert fields["velocity"].shape == (3, 5, 40, 36)
    assert fields["background"].shape == (3, 40, 36)
    assert fields["coil_maps"].shape == (3, 40, 36) and fields["coil_maps"].is_complex()
    assert fields["lumen"].any() and not (fields["lumen"] & fields["static"]).any()


def test_coil_maps_have_unit_root_sum_of_squares() -> None:
    maps = phantom_slice(SPEC, _phantom(), 7)["coil_maps"]
    assert torch.allclose(maps.abs().pow(2).sum(dim=0), torch.ones(40, 36), atol=1e-5)


def test_velocity_is_zero_outside_the_lumen_and_along_each_tube_axis() -> None:
    phantom = _phantom()
    fields = phantom_slice(SPEC, phantom, 11)
    v = fields["velocity"]
    assert v[:, :, ~fields["lumen"]].abs().max() == 0
    speed = v.norm(dim=0)
    assert speed.max() <= SPEC.peak_velocity + 1e-6


def test_flux_through_every_readout_plane_is_the_analytic_tube_flux() -> None:
    """Q = v_peak w(t) pi R^2 / 2 through any section of a straight Poiseuille tube."""
    phantom = _phantom()
    area = SPEC.voxel_size_mm[1] * SPEC.voxel_size_mm[2]
    analytic = (phantom.peaks * phantom.radii**2).sum() * math.pi / 2 * phantom.waveform
    for x in (4, 12, 19):
        flux = phantom_slice(SPEC, phantom, x)["velocity"][0].sum(dim=(-2, -1)) * area
        assert torch.allclose(flux, analytic, rtol=0.05), x


def test_volume_is_divergence_free_inside_the_lumen() -> None:
    from spectramr.core.metrics.context import MetricContext
    from spectramr.core.metrics.flow_metrics import NormalisedDivergence

    phantom = _phantom()
    slices = [phantom_slice(SPEC, phantom, x) for x in range(SPEC.grid[0])]
    velocity = torch.stack([s["velocity"] for s in slices], dim=2)[None].double()
    lumen = torch.stack([s["lumen"] for s in slices]).float()
    # Erode the lumen AND the volume border: torch.gradient is one-sided on the border.
    outside = torch.nn.functional.pad((1.0 - lumen)[None, None], (1,) * 6, value=1.0)
    interior = 1.0 - torch.nn.functional.max_pool3d(outside, 3, 1, 0)[0, 0]
    ctx = MetricContext(voxel_size_mm=torch.tensor(SPEC.voxel_size_mm), foreground_mask=interior)
    # The residual is the float32 rounding of the phantom's velocity values.
    assert NormalisedDivergence()(velocity, context=ctx).item() == pytest.approx(0.0, abs=1e-6)
    compressed = velocity.clone()
    compressed[:, 0] *= torch.linspace(0.5, 1.5, SPEC.grid[0]).view(-1, 1, 1).double()
    assert NormalisedDivergence()(compressed, context=ctx).item() > 0.05


def test_draws_are_deterministic_in_the_generator() -> None:
    a, b = _phantom(9), _phantom(9)
    assert torch.equal(a.directions, b.directions) and torch.equal(a.background, b.background)
    assert not torch.equal(a.directions, _phantom(10).directions)
