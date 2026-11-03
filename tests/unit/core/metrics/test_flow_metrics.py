"""Unit tests for core/metrics/flow_metrics.py, against analytic velocity fields.

Velocity is ``[B, n_dir, *extra, *spatial]`` in m/s, component ``i`` along spatial
axis ``i``; spacing and planes arrive through ``MetricContext``.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.computer import ValidationMetricsComputer
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.flow_metrics import (
    ConsistencyOfMass,
    Divergence,
    NetFlowError,
    NormalisedDivergence,
    VelocityNoiseRatio,
    VelocityRMSEROI,
)
from spectramr.core.metrics.outcome import MetricNotApplicableError, NotApplicableReason
from spectramr.core.metrics.types import MetricSpec, ValidationMetricsConfig

SHAPE = (10, 12, 12)
SPACING = torch.tensor([2.0, 1.5, 1.5])
RADIUS = 4.0


def _lumen() -> torch.Tensor:
    """Cylinder along spatial axis 0, ``[*SHAPE]`` boolean."""
    _, yy, xx = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float32) for n in SHAPE), indexing="ij"
    )
    r2 = (yy - (SHAPE[1] - 1) / 2) ** 2 + (xx - (SHAPE[2] - 1) / 2) ** 2
    return r2 <= RADIUS**2


def _poiseuille(frames: int = 3) -> torch.Tensor:
    """Axial parabolic flow scaled per frame, ``[1, 3, T, *SHAPE]``; divergence-free."""
    _, yy, xx = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float32) for n in SHAPE), indexing="ij"
    )
    r2 = (yy - (SHAPE[1] - 1) / 2) ** 2 + (xx - (SHAPE[2] - 1) / 2) ** 2
    profile = (1.0 - r2 / RADIUS**2).clamp_min(0.0)
    v = torch.zeros(1, 3, frames, *SHAPE)
    v[0, 0] = profile * torch.linspace(0.2, 1.0, frames).view(-1, 1, 1, 1)
    return v


def _plane(index: int, scale: float = 1.0) -> torch.Tensor:
    """Area-weighted normal along axis 0 over the lumen of slice ``index``, mm^2."""
    plane = torch.zeros(3, *SHAPE)
    plane[0, index] = _lumen()[index].float() * (SPACING[1] * SPACING[2]) * scale
    return plane


def _ctx(**fields) -> MetricContext:
    base = {"voxel_size_mm": SPACING, "foreground_mask": _lumen().float()}
    return MetricContext(**{**base, **fields})


def test_poiseuille_flow_is_divergence_free() -> None:
    v = _poiseuille()
    assert Divergence()(v, context=_ctx()).item() == pytest.approx(0.0, abs=1e-6)
    assert NormalisedDivergence()(v, context=_ctx()).item() == pytest.approx(0.0, abs=1e-6)


def test_linear_compression_has_the_analytic_divergence_in_per_second() -> None:
    """v_i = c * x_i (x in metres) has div v = 3c everywhere, at any anisotropic spacing."""
    c = 4.0
    axes = torch.meshgrid(
        *(
            torch.arange(n, dtype=torch.float32) * h * 1e-3
            for n, h in zip(SHAPE, SPACING, strict=True)
        ),
        indexing="ij",
    )
    v = (c * torch.stack(axes))[None]
    ones = MetricContext(voxel_size_mm=SPACING, foreground_mask=torch.ones(SHAPE))
    assert Divergence()(v, context=ones).item() == pytest.approx(3 * c, rel=1e-4)
    assert NormalisedDivergence()(v, context=ones).item() == pytest.approx(1.0, rel=1e-4)


def test_normalised_divergence_of_a_still_field_is_not_applicable() -> None:
    with pytest.raises(MetricNotApplicableError) as err:
        NormalisedDivergence()(torch.zeros(1, 3, *SHAPE), context=_ctx())
    assert err.value.reason is NotApplicableReason.DEGENERATE_VELOCITY_FIELD


def test_divergence_refuses_through_plane_data() -> None:
    """Two-point PC encodes one direction; a 2D divergence needs two."""
    ctx = MetricContext(voxel_size_mm=SPACING[1:], foreground_mask=torch.ones(SHAPE[1:]))
    with pytest.raises(MetricNotApplicableError) as err:
        Divergence()(torch.zeros(1, 1, 5, *SHAPE[1:]), context=ctx)
    assert err.value.reason is NotApplicableReason.VELOCITY_DIRECTIONS_BELOW_SPATIAL_RANK


def test_divergence_without_an_roi_is_not_applicable() -> None:
    with pytest.raises(MetricNotApplicableError) as err:
        Divergence()(_poiseuille(), context=MetricContext(voxel_size_mm=SPACING))
    assert err.value.reason is NotApplicableReason.MISSING_MEASUREMENT_CONTEXT


def test_mass_is_conserved_along_a_straight_tube() -> None:
    v = _poiseuille()
    ctx = _ctx(inlet_mask=_plane(1), outlet_mask=_plane(SHAPE[0] - 2))
    assert ConsistencyOfMass()(v, context=ctx).item() == pytest.approx(0.0, abs=1e-4)
    leaky = _ctx(inlet_mask=_plane(1), outlet_mask=_plane(SHAPE[0] - 2, scale=0.5))
    assert ConsistencyOfMass()(v, context=leaky).item() == pytest.approx(50.0, rel=1e-4)


def test_net_flow_error_is_in_millilitres_per_second() -> None:
    """1 m/s through 1 mm^2 is 1 mL/s, so a uniform field gives the plane's area."""
    v_true = torch.zeros(1, 3, 2, *SHAPE)
    v_true[:, 0] = 1.0
    plane = _plane(3)
    expected = plane[0].sum().item()
    got = NetFlowError()(2 * v_true, v_true, context=_ctx(inlet_mask=plane)).item()
    assert got == pytest.approx(expected, rel=1e-5)


def test_vnr_is_speed_over_static_noise() -> None:
    torch.manual_seed(0)
    sigma = 0.05
    v = _poiseuille(frames=1)[:, :, 0]
    static = ~_lumen()
    v = v + sigma * torch.randn_like(v) * static
    ctx = _ctx(static_mask=static.float())
    expected = v.norm(dim=1)[0][_lumen()].mean() / sigma
    assert VelocityNoiseRatio()(v, context=ctx).item() == pytest.approx(expected.item(), rel=0.1)


def test_roi_rmse_ignores_errors_outside_the_vessel() -> None:
    v = _poiseuille()
    pred = v + 0.3 * (~_lumen()).float()
    assert VelocityRMSEROI()(pred, v, context=_ctx()).item() == pytest.approx(0.0, abs=1e-7)


def test_the_validation_computer_routes_the_context() -> None:
    """Through the real computer: context reaches the metrics, not the target slot."""
    names = ("divergence", "mass_conservation", "velocity_rmse_roi", "net_flow_error")
    computer = ValidationMetricsComputer(
        config=ValidationMetricsConfig(metrics=[MetricSpec.from_name(n) for n in names])
    )
    v = _poiseuille()
    ctx = _ctx(inlet_mask=_plane(1), outlet_mask=_plane(SHAPE[0] - 2))
    out = computer.compute(v, v, context=ctx)
    for name in names:
        assert out[name] == pytest.approx(0.0, abs=1e-4), name
