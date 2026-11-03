"""Tests for ``spectramr.core.metrics.temporal_difference``."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.computer import create_validation_metrics_computer
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.temporal_difference import TemporalDifferenceNRMSE
from spectramr.core.metrics.temporal_metrics import TemporalFidelity


def _moving_series(frames: int = 5, size: int = 8) -> torch.Tensor:
    series = torch.zeros(2, frames, size, size)
    for t in range(frames):
        series[:, t, 2:5, 1 + t : 4 + t] = 1.0
    return series + 0.1


def test_metrics_compute_list_reaches_the_metric() -> None:
    computer = create_validation_metrics_computer(config=["temporal_difference_nrmse"])
    target = _moving_series()
    static = target[:, :1].repeat(1, target.shape[1], 1, 1)
    scores = computer.compute(static, target)
    assert scores["temporal_difference_nrmse"] == pytest.approx(1.0)
    assert computer.compute(target, target)["temporal_difference_nrmse"] == pytest.approx(0.0)


def test_registered_lower_is_better() -> None:
    assert MetricsRegistry.is_registered("inter_frame_nrmse")
    assert TemporalDifferenceNRMSE().higher_is_better is False


def test_damped_motion_is_caught_where_fidelity_is_blind() -> None:
    target = _moving_series()
    mean = target.mean(dim=1, keepdim=True)
    damped = mean + 0.5 * (target - mean)
    assert TemporalFidelity()(damped, target) == pytest.approx(1.0, abs=1e-5)
    assert TemporalDifferenceNRMSE()(damped, target) == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize(
    ("pred", "target"),
    [
        (torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4)),
        (torch.rand(1, 3, 4, 4), torch.rand(1, 4, 4, 4)),
        (torch.ones(1, 3, 4, 4), torch.ones(1, 3, 4, 4)),
        (
            torch.randn(1, 3, 4, 4, dtype=torch.complex64),
            torch.randn(1, 3, 4, 4, dtype=torch.complex64),
        ),
    ],
    ids=["single_frame", "shape_mismatch", "static_reference", "complex"],
)
def test_degenerate_input_skips_with_nan(pred: torch.Tensor, target: torch.Tensor) -> None:
    assert math.isnan(TemporalDifferenceNRMSE()(pred, target))
