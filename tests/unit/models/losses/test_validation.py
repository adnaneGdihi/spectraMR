"""Tests for the loss-input validation decorator.

F34 (2026-05-22) — TorchIO wraps 1-D/2-D data with trailing singleton axes
(``[B, C, L, 1, 1]`` for 1-D PDE benchmarks). Shape-rigid operators squeeze
those from their input, so the model output is ``[B, C, L]`` while the raw
target keeps the trailing unit axes. ``validate_loss_inputs`` then raised
``[L1Loss] Shape mismatch`` and crashed cs_mno_burgers / cs_mno_darcy at
train step 1. The decorator now strips trailing size-1 axes (numel-preserving)
before the shape check, while still failing loudly on a genuine extent
mismatch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # noqa: E402

import torch.nn as nn  # noqa: E402

from spectramr.models.losses.validation import (  # noqa: E402
    _strip_trailing_singletons,
    validate_loss_inputs,
)


class _L1(nn.Module):
    @validate_loss_inputs
    def forward(self, pred, target):
        return torch.nn.functional.l1_loss(pred, target)


def test_strip_trailing_singletons_removes_only_trailing_unit_axes() -> None:
    t = torch.randn(16, 1, 128, 1, 1)
    out = _strip_trailing_singletons(t)
    assert out.shape == (16, 1, 128)
    # No trailing singletons → unchanged.
    t2 = torch.randn(4, 2, 32, 32)
    assert _strip_trailing_singletons(t2).shape == (4, 2, 32, 32)
    # Never strips below [B, C, *].
    t3 = torch.randn(4, 1, 1)
    assert _strip_trailing_singletons(t3).shape == (4, 1, 1)


def test_pred_3d_vs_target_5d_trailing_singletons_does_not_raise() -> None:
    """The cs_mno_burgers/darcy crash: pred [16,1,128] vs target [16,1,128,1,1]."""
    loss = _L1()
    pred = torch.randn(16, 1, 128)
    target = pred.detach().clone().reshape(16, 1, 128, 1, 1)
    out = loss(pred, target)
    assert torch.isfinite(out)
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_genuine_extent_mismatch_still_raises() -> None:
    loss = _L1()
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss(torch.randn(16, 1, 128), torch.randn(16, 1, 64))


def test_nan_diagnostics_are_debug_gated(monkeypatch, caplog) -> None:
    """WS-6: ``torch.isnan(t).any()`` in a Python ``if`` is a host-device sync
    plus a full tensor pass, per loss call, on the training hot path. The
    decorator must not evaluate it unless DEBUG logging is enabled on the
    validation logger.
    """
    import logging

    calls = {"count": 0}
    real_isnan = torch.isnan

    def counting_isnan(t):
        calls["count"] += 1
        return real_isnan(t)

    monkeypatch.setattr(torch, "isnan", counting_isnan)

    loss = _L1()
    pred = torch.randn(2, 1, 4, 4)
    pred[0, 0, 0, 0] = float("nan")
    target = torch.randn(2, 1, 4, 4)

    logger_name = "spectramr.models.losses.validation"
    with caplog.at_level(logging.INFO, logger=logger_name):
        loss(pred, target)
    assert calls["count"] == 0, "NaN checks must not run on the default hot path"

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        loss(pred, target)
    assert calls["count"] > 0, "DEBUG level must re-enable the NaN diagnostics"
    assert any("NaN detected in prediction" in rec.getMessage() for rec in caplog.records)


def test_the_wrapper_keeps_the_forward_identity() -> None:
    """``kwargs_accepted_by`` reads the signature through ``__wrapped__``."""
    import inspect

    fwd = _L1.forward
    assert fwd.__name__ == "forward"
    assert list(inspect.signature(fwd).parameters) == ["self", "pred", "target"]
