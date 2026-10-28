"""``UnifiedGANLossComputer`` applies a dynamic ``losses_dict`` weight exactly once.

Both ``compute`` and ``compute_generator_loss`` stored each dynamic term as
``loss_val * weight`` and then handed it to ``_stack_components``, which multiplies
by the same weight again: a declared ``mse`` at 0.5 trained at 0.25. Invisible
at 1.0, which is why no test caught it. The sibling fix already landed in the VAE
and diffusion computers (see ``test_unified_vae.py``).

The per-loss ``try`` still owns what it owned before: a weight that cannot be
resolved is reported and skipped there, not raised out of the stack (#289).
"""

from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from spectramr.config.settings import TrainingSettings
from spectramr.models.losses.computers import UnifiedGANLossComputer


class _Constant(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self._value = value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return pred.sum() * 0.0 + self._value


def _computer() -> UnifiedGANLossComputer:
    settings = TrainingSettings.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {"dataset_type": "synthetic"},
            "optimization": {},
            "logging": {},
            "losses": {"output_domain": "image", "image_losses": [{"name": "mse", "weight": 0.5}]},
        }
    )
    computer = UnifiedGANLossComputer(config=settings)
    computer.reconstruction_loss_fn = _Constant(5.0)
    return computer


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    return nn.Conv2d(1, 1, 3, padding=1)(torch.rand(2, 1, 8, 8)), torch.rand(2, 1, 8, 8)


@pytest.mark.parametrize("method", ["compute", "compute_generator_loss"])
def test_a_dynamic_term_is_weighted_once(method: str) -> None:
    computer = _computer()
    pred, target = _pair()

    out = getattr(computer, method)(
        pred=pred, target=target, iteration=5000, losses_dict={"mse": _Constant(2.0)}
    )

    assert float(out.components["mse"]) == pytest.approx(2.0), "stored raw"
    # 1.0 * 5.0 (reconstruction) + 0.5 * 2.0 -- the old code gave 5.0 + 0.25 * 2.0.
    assert float(out.total) == pytest.approx(6.0)


@pytest.mark.parametrize("method", ["compute", "compute_generator_loss"])
def test_an_undeclared_dynamic_term_is_still_skipped_in_the_loop(method: str, caplog) -> None:
    computer = _computer()
    pred, target = _pair()

    with caplog.at_level(logging.WARNING):
        out = getattr(computer, method)(
            pred=pred,
            target=target,
            iteration=5000,
            losses_dict={"not_a_declared_loss": _Constant(2.0)},
        )

    assert "not_a_declared_loss" not in out.components
    assert float(out.total) == pytest.approx(5.0)
    assert any("not_a_declared_loss" in r.getMessage() for r in caplog.records)
