"""Coverage tests for the quantitative-qMap certificate metrics (WS-2 core).

Covers two registered metrics:

* :class:`spectramr.core.metrics.quantitative.geodesic_qmap_error.GeodesicQMapErrorMetric`
  — the Fisher-geodesic analogue of MAE on ``(M0, T1, T2)`` maps. Asserts
  identity-zero, strict monotonicity under a scaled displacement about a
  fixed midpoint, and a shape-mismatch ``ValueError``.
* :class:`spectramr.core.metrics.quantitative.cohomology_certificate.GaloisH1Certificate`
  — the Galois-cohomological ``H^1`` defensibility certificate for phase
  unwrapping. Asserts zero on a globally consistent (smooth, unwrapped)
  phase, a strictly positive value when a winding residue (vortex) is
  present, and that the ``target`` argument is ignored.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.quantitative.cohomology_certificate import (
    GaloisH1Certificate,
)
from spectramr.core.metrics.quantitative.geodesic_qmap_error import (
    GeodesicQMapErrorMetric,
)
from spectramr.infrastructure.physics.galois_unwrap import wrap_to_principal

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _qmap(b: int = 1, h: int = 4, w: int = 4) -> torch.Tensor:
    """A physically plausible ``[B, 3, H, W]`` (M0, T1, T2) parameter map."""
    torch.manual_seed(0)
    out = torch.empty(b, 3, h, w)
    out[:, 0].uniform_(0.5, 1.2)  # M0 (arbitrary units)
    out[:, 1].uniform_(800.0, 1200.0)  # T1 (ms)
    out[:, 2].uniform_(50.0, 200.0)  # T2 (ms), kept < T1
    return out


def _midpoint_map(h: int = 4, w: int = 4) -> torch.Tensor:
    """A constant (M0, T1, T2) field used as a fixed geodesic midpoint."""
    return torch.stack(
        [
            torch.full((h, w), 1.0),
            torch.full((h, w), 900.0),
            torch.full((h, w), 70.0),
        ]
    ).unsqueeze(0)


def _vortex_phase(h: int = 16, w: int = 16) -> torch.Tensor:
    """A wrapped phase field carrying one +2pi winding singularity.

    ``atan2`` over a centred grid has a single branch point at the origin;
    its plaquette residue is the topological witness of a non-trivial
    ``H^1`` obstruction. Shape is ``[1, D=1, H, W]`` as expected by the
    cubic-grid certificate.
    """
    y = torch.arange(h).float() - h / 2 + 0.5
    x = torch.arange(w).float() - w / 2 + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase = torch.atan2(yy, xx).reshape(1, 1, h, w)
    return wrap_to_principal(phase)


def _smooth_phase(h: int = 8, w: int = 8) -> torch.Tensor:
    """A globally consistent (sub-pi gradient, no wraps) phase field."""
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, h),
        torch.linspace(0.0, 1.0, w),
        indexing="ij",
    )
    # Max magnitude ~0.8 rad over the field; every adjacent step << pi, so
    # the integer winding cocycle is identically zero.
    phase = 0.5 * xx + 0.3 * yy
    return phase.reshape(1, 1, h, w)


# --------------------------------------------------------------------------- #
# GeodesicQMapErrorMetric
# --------------------------------------------------------------------------- #


class TestGeodesicQMapError:
    def test_contract_properties(self) -> None:
        m = GeodesicQMapErrorMetric()
        assert m.name == "geodesic_qmap_error"
        assert m.higher_is_better is False

    def test_identity_is_zero(self) -> None:
        """Prediction == target ⇒ zero displacement ⇒ zero geodesic distance."""
        m = GeodesicQMapErrorMetric()
        theta = _qmap()
        d = m(theta, theta.clone())
        assert isinstance(d, float)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_monotonic_in_displacement(self) -> None:
        """Distance grows strictly with displacement at a fixed midpoint.

        Building pairs ``a = c - s*u`` / ``b = c + s*u`` keeps the metric
        evaluation point ``mid = (a+b)/2 = c`` constant, so the geodesic
        distance ``sqrt((b-a)^T g(c) (b-a))`` scales linearly in ``s`` and
        must be strictly increasing.
        """
        m = GeodesicQMapErrorMetric()
        c = _midpoint_map()
        u = torch.tensor([0.1, 50.0, 20.0]).reshape(1, 3, 1, 1)
        vals = [m(c - s * u, c + s * u) for s in (0.25, 0.5, 1.0, 2.0)]
        assert all(later > earlier for earlier, later in zip(vals, vals[1:]))
        # Linearity in s (fixed metric): doubling s doubles the distance.
        assert vals[2] == pytest.approx(2.0 * vals[1], rel=1e-3)

    def test_nonzero_when_maps_differ(self) -> None:
        m = GeodesicQMapErrorMetric()
        a = _qmap()
        b = a.clone()
        b[:, 1] += 100.0  # perturb T1
        assert m(a, b) > 0.0

    def test_shape_mismatch_raises(self) -> None:
        m = GeodesicQMapErrorMetric()
        with pytest.raises(ValueError, match="shape mismatch"):
            m(torch.rand(1, 3, 4, 4), torch.rand(1, 3, 4, 5))


# --------------------------------------------------------------------------- #
# GaloisH1Certificate (cohomology certificate wrapper)
# --------------------------------------------------------------------------- #


class TestGaloisH1Certificate:
    def test_contract_properties(self) -> None:
        m = GaloisH1Certificate()
        assert m.name == "galois_h1_certificate"
        assert m.higher_is_better is False

    def test_zero_on_consistent_phase(self) -> None:
        """A smooth, never-wrapping phase has a vanishing H^1 obstruction."""
        m = GaloisH1Certificate()
        cert = m(_smooth_phase())
        assert isinstance(cert, float)
        assert cert == pytest.approx(0.0, abs=1e-6)

    def test_positive_on_residue(self) -> None:
        """A single winding singularity yields a strictly positive certificate."""
        m = GaloisH1Certificate()
        cert = m(_vortex_phase())
        assert cert > 0.0
        # One +2pi vortex ⇒ exactly one unit plaquette residue ⇒ l2 norm 1.
        assert cert == pytest.approx(1.0, abs=1e-5)

    def test_target_ignored(self) -> None:
        """The certificate is a property of the prediction alone."""
        m = GaloisH1Certificate()
        phi = _vortex_phase()
        base = m(phi)
        assert m(phi, None) == pytest.approx(base, abs=1e-9)
        assert m(phi, torch.zeros_like(phi)) == pytest.approx(base, abs=1e-9)
        assert m(phi, torch.rand_like(phi)) == pytest.approx(base, abs=1e-9)

    def test_invariant_to_global_2pi_shift(self) -> None:
        """A global deck-group shift (±2pi) leaves the certificate unchanged."""
        m = GaloisH1Certificate()
        phi = _vortex_phase()
        shifted = wrap_to_principal(phi + 2.0 * math.pi)
        assert m(shifted) == pytest.approx(m(phi), abs=1e-5)


class TestGeodesicQMapManifoldInjection:
    def test_registry_passes_an_injected_manifold_through(self):
        from spectramr.core.metrics.registry import MetricsRegistry
        from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold

        manifold = BlochRelaxationManifold(cache_resolution=4)
        metric = MetricsRegistry.get("geodesic_qmap_error", manifold=manifold)
        assert metric.manifold is manifold
        assert metric(_qmap(h=8, w=8), _qmap(h=8, w=8)) == pytest.approx(0.0, abs=1e-6)

    def test_uncached_metric_refuses_a_map_it_would_loop_over(self):
        big = torch.ones(1, 3, 65, 65)
        big[:, 1] = 900.0
        big[:, 2] = 70.0
        with pytest.raises(ValueError, match="cache_resolution"):
            GeodesicQMapErrorMetric()(big, big)

    def test_foreground_mask_restricts_the_mean(self):
        from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold

        metric = GeodesicQMapErrorMetric(BlochRelaxationManifold(cache_resolution=4))
        target = _qmap(h=4, w=4)
        pred = target.clone()
        pred[:, 1, 0, 0] += 200.0  # one wrong voxel
        fg = torch.ones(1, 4, 4, dtype=torch.bool)
        fg[:, 0, 0] = False
        assert metric(pred, target, foreground=fg) == pytest.approx(0.0, abs=1e-6)
        assert metric(pred, target) > 0.0
        with pytest.raises(ValueError, match="does not match"):
            metric(pred, target, foreground=fg[:, :2])
