"""Routing-by-agreement must use per-sample prediction vectors (#2402)."""

from __future__ import annotations

import torch

from spectramr.models.generators.capsule_network import _RoutingCapsules


def _routing() -> _RoutingCapsules:
    torch.manual_seed(0)
    return _RoutingCapsules(in_caps=4, in_dim=8, out_caps=3, out_dim=5)


def test_output_of_one_sample_does_not_depend_on_the_rest_of_the_batch():
    r = _routing()
    u = torch.randn(10, 4, 8)
    with torch.no_grad():
        alone = r(u[:1])
        batched = r(u)
    torch.testing.assert_close(alone[0], batched[0])


def test_routing_runs_in_float64_and_squashes_below_unit_norm():
    r = _routing().double()
    u = torch.randn(6, 4, 8, dtype=torch.float64)
    v = r(u)
    assert v.dtype == torch.float64
    assert v.shape == (6, 3, 5)
    # A squashed vector has norm strictly below one.
    assert (v.norm(dim=-1) < 1).all()


def test_gradients_are_finite_at_a_zero_capsule():
    r = _routing()
    u = torch.randn(32, 4, 8)
    u[0] = 0.0
    r(u).sum().backward()
    assert torch.isfinite(r.W.grad).all()
