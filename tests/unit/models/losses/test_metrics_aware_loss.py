"""``add_metrics_to_loss`` splatted the source module's ``__dict__`` values into the
constructor positionally, so it raised ``TypeError`` for every ``nn.Module`` (#1419).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.metrics_aware_loss import (
    MetricsAwareLossMixin,
    add_metrics_to_loss,
)


def test_wraps_a_plain_module_and_preserves_its_state() -> None:
    base = nn.MSELoss(reduction="sum")
    wrapped = add_metrics_to_loss(base)

    assert isinstance(wrapped, MetricsAwareLossMixin)
    assert isinstance(wrapped, nn.MSELoss)
    assert wrapped.reduction == "sum"
    pred, target = torch.randn(2, 3), torch.randn(2, 3)
    torch.testing.assert_close(wrapped(pred, target), base(pred, target))


def test_forward_with_metrics_reports_the_same_loss() -> None:
    wrapped = add_metrics_to_loss(nn.L1Loss())
    pred, target = torch.randn(4, 5), torch.randn(4, 5)
    loss, metrics = wrapped.forward_with_metrics(pred, target)
    torch.testing.assert_close(loss, nn.functional.l1_loss(pred, target))
    assert metrics["loss"] == loss.item()
    assert {"pred_mean", "pred_std", "target_mean", "target_std"} <= set(metrics)


def test_a_module_with_parameters_keeps_them() -> None:
    class Scaled(nn.Module):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(scale))

        def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return self.scale * (pred - target).abs().mean()

    base = Scaled(2.5)
    wrapped = add_metrics_to_loss(base)
    assert next(iter(wrapped.parameters())) is base.scale


def test_an_already_metrics_aware_loss_is_returned_unchanged() -> None:
    from spectramr.models.losses.standard_losses import L1Loss

    loss = L1Loss()
    assert add_metrics_to_loss(loss) is loss
