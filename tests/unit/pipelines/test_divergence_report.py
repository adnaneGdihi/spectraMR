"""The divergence report tells a NaN forward apart from a poisoned weight update."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.pipelines.divergence_report import (  # noqa: E402
    describe_nonfinite_state,
    log_divergence_state,
)

pytestmark = pytest.mark.unit


def _model() -> torch.nn.Module:
    m = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    return m


def test_finite_state_blames_the_forward() -> None:
    assert "the forward produced the NaN" in describe_nonfinite_state(_model())


def test_planted_nan_weight_is_named() -> None:
    m = _model()
    with torch.no_grad():
        m[0].weight[0, 0] = float("nan")
    out = describe_nonfinite_state(m)
    assert "already non-finite" in out and "0.weight" in out


def test_planted_inf_gradient_with_finite_weights() -> None:
    m = _model()
    m[0].bias.grad[1] = float("inf")
    out = describe_nonfinite_state(m)
    assert "gradients are not" in out and "0.bias" in out


def test_planted_nan_buffer_is_counted() -> None:
    m = _model()
    m[1].running_mean[0] = float("nan")
    assert "buffers=1" in describe_nonfinite_state(m)


def test_log_never_raises_on_a_broken_model(caplog: pytest.LogCaptureFixture) -> None:
    log_divergence_state(object(), 3)  # type: ignore[arg-type]
    assert "state scan failed" in caplog.text
