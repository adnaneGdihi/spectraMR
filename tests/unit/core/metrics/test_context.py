"""Tests for the MetricContext seam (spectramr.core.metrics.context)."""

from __future__ import annotations

import pytest
import torch

from spectramr.core.metrics.context import MetricContext, resolve_context


def test_defaults_all_none() -> None:
    ctx = MetricContext()
    assert ctx.y_kspace is None
    assert ctx.mask is None
    assert ctx.coil_maps is None
    assert ctx.nr_params is None


def test_frozen() -> None:
    from dataclasses import FrozenInstanceError

    ctx = MetricContext()
    with pytest.raises(FrozenInstanceError):
        ctx.y_kspace = torch.zeros(1)  # type: ignore[misc]


def test_has() -> None:
    k = torch.zeros(1, 1, 4, 4)
    ctx = MetricContext(y_kspace=k, mask=k)
    assert ctx.has("y_kspace", "mask")
    assert not ctx.has("y_kspace", "coil_maps")
    assert not ctx.has("coil_maps")


def test_require_raises_on_missing() -> None:
    ctx = MetricContext(y_kspace=torch.zeros(1))
    ctx.require("ndcr", "y_kspace")  # present → no raise
    with pytest.raises(ValueError, match="requires MetricContext fields"):
        ctx.require("ndcr", "y_kspace", "coil_maps")


def test_from_kwargs_maps_sensitivity_maps_alias() -> None:
    smaps = torch.zeros(1, 4, 8, 8)
    ctx = MetricContext.from_kwargs({"sensitivity_maps": smaps, "acceleration": 4.0})
    assert ctx.coil_maps is smaps
    assert ctx.acceleration == 4.0


def test_from_kwargs_ignores_unknown() -> None:
    ctx = MetricContext.from_kwargs({"not_a_field": 1, "mask": torch.zeros(2)})
    assert ctx.mask is not None
    # unknown key did not crash and is not stored anywhere
    assert not hasattr(ctx, "not_a_field")


def test_resolve_context_passthrough() -> None:
    ctx = MetricContext(acceleration=2.0)
    assert resolve_context(ctx, {}) is ctx


def test_resolve_context_builds_from_kwargs() -> None:
    out = resolve_context(None, {"acceleration": 3.0})
    assert isinstance(out, MetricContext)
    assert out.acceleration == 3.0


def test_ensemble_std_field_defaults_none_and_is_carried() -> None:
    """The per-pixel spread of a reverse-sample ensemble rides the same seam."""
    assert MetricContext().ensemble_std is None
    std = torch.rand(1, 1, 4, 4)
    assert MetricContext(ensemble_std=std).has("ensemble_std")
    assert MetricContext.from_kwargs({"ensemble_std": std}).ensemble_std is std


def test_pred_alt_field_defaults_none_and_is_carried() -> None:
    """The second-repetition prediction rides the seam rep_variance reads."""
    assert MetricContext().pred_alt is None
    alt = torch.rand(1, 8, 4, 4)
    assert MetricContext(pred_alt=alt).has("pred_alt")
    assert MetricContext.from_kwargs({"pred_alt": alt}).pred_alt is alt


def test_signal_domain_field_defaults_none_and_is_carried() -> None:
    """The caller's statement of what the tensors are; None leaves the metric's default."""
    assert MetricContext().signal_domain is None
    assert MetricContext(signal_domain="kspace").signal_domain == "kspace"
    assert MetricContext.from_kwargs({"signal_domain": "image"}).signal_domain == "image"


def test_pred_kspace_field_defaults_none_and_is_carried() -> None:
    """The prediction's coil k-space, which ndcr compares per coil (#2316)."""
    assert MetricContext().pred_kspace is None
    k = torch.zeros(1, 4, 8, 8, dtype=torch.complex64)
    assert MetricContext(pred_kspace=k).has("pred_kspace")
    assert MetricContext.from_kwargs({"pred_kspace": k}).pred_kspace is k
