"""Tests for ``spectramr.models.losses.temporal_consistency_loss``."""

from __future__ import annotations

import pytest
import torch

from spectramr.config.schemas.loss import LossComponentConfig, LossConfigSchema
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.builders.loss_builder import LossBuilder
from spectramr.models.losses.registry import LossRegistry
from spectramr.models.losses.temporal_consistency_loss import TemporalConsistencyLoss


def _moving_series(frames: int = 4, size: int = 8) -> torch.Tensor:
    """A bright square translating one pixel per frame: [1, T, H, W]."""
    series = torch.zeros(1, frames, size, size)
    for t in range(frames):
        series[0, t, 2:5, 1 + t : 4 + t] = 1.0
    return series


class _ConfigShim:
    def __init__(self, losses: LossConfigSchema) -> None:
        self.losses = losses
        self.objectives = None
        self.training = None
        self.training_mode = "reconstruction"
        self.deep_supervision_weight = 0.0


def _build(**kwargs: object) -> LossBuilder:
    entry = LossComponentConfig(
        name="temporal_consistency", weight=1.0, enabled=True, kwargs=kwargs
    )
    losses = LossConfigSchema(output_domain="image", image_losses=[entry])
    builder = LossBuilder(_ConfigShim(losses), device="cpu")  # type: ignore[arg-type]
    builder._build_list_based_losses()
    return builder


def test_yaml_entry_resolves_through_the_builder_and_fires() -> None:
    builder = _build(norm="l2")
    loss_fn = builder._losses["temporal_consistency"]
    target = _moving_series()
    # The repeat()-fabricated axis of the static route: every frame identical.
    static = target[:, :1].repeat(1, target.shape[1], 1, 1)
    assert float(loss_fn(static, target)) > 0.0
    assert float(loss_fn(target.clone(), target)) == 0.0


def test_unknown_kwarg_raises_at_build() -> None:
    with pytest.raises(ConfigurationError, match="cannot consume"):
        _build(order=2)


def test_registry_alias_resolves() -> None:
    assert LossRegistry.get_loss_class("inter_frame_consistency") is TemporalConsistencyLoss


def test_constant_offset_is_ignored_but_damped_motion_is_not() -> None:
    loss = TemporalConsistencyLoss()
    target = _moving_series()
    assert float(loss(target + 0.3, target)) == pytest.approx(0.0, abs=1e-6)
    assert float(loss(0.5 * target, target)) > 0.0


@pytest.mark.parametrize(("norm", "reduction"), [("l3", "mean"), ("l1", "none")])
def test_unsupported_options_raise(norm: str, reduction: str) -> None:
    with pytest.raises(ValueError):
        TemporalConsistencyLoss(norm=norm, reduction=reduction)


def test_single_frame_raises_rather_than_returning_zero() -> None:
    loss = TemporalConsistencyLoss()
    x = torch.rand(2, 1, 8, 8)
    with pytest.raises(ValueError, match="single frame"):
        loss(x, x)


def test_complex_and_shape_mismatch_raise() -> None:
    loss = TemporalConsistencyLoss()
    z = torch.randn(1, 3, 4, 4, dtype=torch.complex64)
    with pytest.raises(ValueError, match="complex"):
        loss(z, z)
    with pytest.raises(ValueError, match="shape mismatch"):
        loss(torch.rand(1, 3, 4, 4), torch.rand(1, 4, 4, 4))


def test_gradient_flows_to_prediction() -> None:
    target = _moving_series()
    pred = torch.zeros_like(target, requires_grad=True)
    TemporalConsistencyLoss()(pred, target).backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0.0
