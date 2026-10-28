"""Tests for :mod:`spectramr.models.losses.composed_loss`.

The contract under test is the ``want_metrics`` gate on ``WeightedLoss.compute``
(production-plan row D07#8). Three properties have to hold together, and each is
a separate failure mode:

1. ``ComposedLoss.forward`` must not run any component's ``compute_metrics``.
   That method performs ~7 ``.item()`` device-to-host syncs per component, and
   the training loop calls ``forward`` every step (non-negotiable 9).
2. ``ComposedLoss.compute_metrics`` must still return component metrics. The
   gate defaults to off, so the metric-consuming call sites have to opt back in
   explicitly -- forgetting one silently drops the feature rather than failing.
3. A component whose ``forward`` returns ``(loss, metrics)`` must still yield a
   scalar. Gating the branch made the plain path reachable for the five losses
   that take a ``compute_metrics=True`` constructor flag; without an unwrap,
   ``tuple * float`` raises and ``tuple * int`` silently returns a longer tuple.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.composed_loss import (
    ComposedLoss,
    ConditionalComposedLoss,
    WeightedLoss,
)
from spectramr.models.losses.standard_losses import L1Loss


@pytest.fixture
def pair() -> tuple[torch.Tensor, torch.Tensor]:
    """A deterministic (pred, target) pair on CPU."""
    g = torch.Generator().manual_seed(0)
    return torch.rand(2, 1, 8, 8, generator=g), torch.rand(2, 1, 8, 8, generator=g)


class _MetricsSpy(L1Loss):
    """An L1 loss that records how often its metrics path is entered."""

    def __init__(self) -> None:
        super().__init__()
        self.metrics_calls = 0

    def compute_metrics(self, *args, **kwargs) -> dict[str, float]:
        self.metrics_calls += 1
        return super().compute_metrics(*args, **kwargs)


class _TupleLoss(torch.nn.Module):
    """Mimics a loss built with ``compute_metrics=True``: ``forward`` -> tuple.

    It deliberately also exposes ``forward_with_metrics`` -- that combination is
    what the five real mixin losses have, and it is the combination that makes
    the gated plain branch reachable with a tuple in hand.
    """

    def forward(self, pred, target, **kwargs):
        return torch.nn.functional.l1_loss(pred, target), {"probe": 1.0}

    def forward_with_metrics(self, pred, target, **kwargs):
        loss, metrics = self.forward(pred, target, **kwargs)
        return loss, metrics


# --------------------------------------------------------------------------
# 1. the forward path is sync-free
# --------------------------------------------------------------------------


def test_forward_does_not_enter_the_metrics_path(pair):
    """``forward`` must leave ``compute_metrics`` untouched.

    Planted-violation guard (non-negotiable 15): reverting the gate -- i.e.
    taking ``forward_with_metrics`` whenever the component has it -- makes this
    assert read 1 instead of 0.
    """
    pred, target = pair
    spy = _MetricsSpy()
    composed = ComposedLoss([WeightedLoss("l1", spy, 1.0)])

    composed(pred, target)

    assert spy.metrics_calls == 0


def test_forward_does_no_device_to_host_syncs(pair):
    """No ``.item()`` anywhere under ``ComposedLoss.forward``.

    This is the property non-negotiable 9 actually cares about; the call-count
    test above is the mechanism. Asserting both means a future refactor that
    reintroduces a sync by another route still turns something red.
    """
    pred, target = pair
    composed = ComposedLoss([WeightedLoss("l1", L1Loss(), 1.0), WeightedLoss("l1b", L1Loss(), 0.5)])

    calls = 0
    real_item = torch.Tensor.item

    def counting_item(self):
        nonlocal calls
        calls += 1
        return real_item(self)

    torch.Tensor.item = counting_item
    try:
        composed(pred, target)
    finally:
        torch.Tensor.item = real_item

    assert calls == 0


def test_forward_returns_the_weighted_sum(pair):
    """The gate must not change the number ``forward`` produces."""
    pred, target = pair
    composed = ComposedLoss([WeightedLoss("a", L1Loss(), 2.0), WeightedLoss("b", L1Loss(), 0.5)])

    expected = torch.nn.functional.l1_loss(pred, target) * 2.5

    torch.testing.assert_close(composed(pred, target), expected)


# --------------------------------------------------------------------------
# 2. the metrics feature is conserved
# --------------------------------------------------------------------------


def test_compute_metrics_still_collects_component_metrics(pair):
    """``compute_metrics`` opts back in, so component metrics survive the gate.

    Dropping ``want_metrics=True`` at the call site is the regression this
    catches: ``forward_with_metrics`` would never run, every component's dict
    would be ``None``, and the returned mapping would silently shrink to the
    two unprefixed keys.
    """
    pred, target = pair
    spy = _MetricsSpy()
    composed = ComposedLoss([WeightedLoss("l1", spy, 1.0)])

    metrics = composed.compute_metrics(pred, target, composed(pred, target))

    assert spy.metrics_calls == 1
    assert "l1_loss" in metrics
    assert [k for k in metrics if k.startswith("l1_") and k != "l1_loss"], (
        "component metrics were dropped: only the aggregate key survived"
    )


def test_conditional_compute_metrics_also_opts_back_in(pair):
    """``ConditionalComposedLoss`` duplicates both loops and needs the same fix.

    Neither the plan row nor the dossier mentions this subclass; it has its own
    ``forward`` and its own ``compute_metrics``, so a fix applied only to the
    base class conserves the feature in one place and drops it in the other.
    """
    pred, target = pair
    spy = _MetricsSpy()
    composed = ConditionalComposedLoss([WeightedLoss("l1", spy, 1.0)])

    composed(pred, target)
    assert spy.metrics_calls == 0, "ConditionalComposedLoss.forward still syncs"

    metrics = composed.compute_metrics(pred, target, composed(pred, target))
    assert spy.metrics_calls == 1
    assert [k for k in metrics if k.startswith("l1_") and k != "l1_loss"]


# --------------------------------------------------------------------------
# 3. the branch the gate made reachable
# --------------------------------------------------------------------------


@pytest.mark.parametrize("weight", [2.0, 2], ids=["float-weight", "int-weight"])
def test_tuple_returning_component_is_unwrapped(pair, weight):
    """A ``(loss, metrics)`` ``forward`` must still weight to a scalar.

    Both weight types are exercised on purpose: without the unwrap the float
    case raises ``TypeError`` (loud) while the int case returns a repeated
    tuple (silent). Only the second would reach a training run unnoticed.
    """
    pred, target = pair
    composed = ComposedLoss([WeightedLoss("tup", _TupleLoss(), weight)])

    total = composed(pred, target)

    assert isinstance(total, torch.Tensor)
    assert total.ndim == 0
    torch.testing.assert_close(total, torch.nn.functional.l1_loss(pred, target) * float(weight))


def test_want_metrics_is_keyword_only(pair):
    """It must not be reachable positionally.

    ``compute`` forwards ``**kwargs`` verbatim to the wrapped loss, so a
    positional or plain-keyword parameter here could be filled by an argument
    meant for that loss.
    """
    pred, target = pair
    wl = WeightedLoss("l1", L1Loss(), 1.0)

    with pytest.raises(TypeError):
        wl.compute(pred, target, True)  # type: ignore[misc]


def test_compute_returns_no_metrics_by_default(pair):
    """The default is off, and off means ``None`` -- not an empty dict."""
    pred, target = pair
    _, metrics = WeightedLoss("l1", L1Loss(), 1.0).compute(pred, target)

    assert metrics is None


class _MagnitudeL1(torch.nn.Module):
    """A real-valued loss of complex inputs, as every complex-MRI loss is."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (pred - target).abs().mean()


@pytest.mark.parametrize("cls", [ComposedLoss, ConditionalComposedLoss])
def test_a_complex_prediction_yields_a_real_total_that_backpropagates(cls):
    """The seed took ``pred.dtype``, so complex data made the total complex and
    ``backward()`` raised "grad can be implicitly created only for real scalar outputs"."""
    pred = torch.randn(2, 1, 4, 4, dtype=torch.complex64, requires_grad=True)
    target = torch.randn(2, 1, 4, 4, dtype=torch.complex64)

    total = cls([WeightedLoss("l1", _MagnitudeL1(), 2.0)])(pred, target)

    assert total.dtype == torch.float32
    assert float(total) == pytest.approx(2.0 * float(_MagnitudeL1()(pred, target)))
    total.backward()
    assert pred.grad is not None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_a_real_prediction_keeps_its_dtype(dtype):
    pred = torch.rand(2, 1, 4, 4, dtype=dtype)
    total = ComposedLoss([WeightedLoss("l1", _MagnitudeL1(), 1.0)])(pred, pred.clone())
    assert total.dtype == dtype
