"""``_compute_gradient_norm``: one clip path, global on every backend.

The training loop clips through ``strategy._clip_and_log_gradients``, so this hook
owns the clip. Under FSDP2 the plain utility over DTensor gradients is already the
global norm; that case runs on two real ranks in
``tests/unit/infrastructure/distributed/test_state_io_fsdp.py``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy


class _Strat(BaseTrainingStrategy):
    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - stub
        return {}


def _strategy() -> _Strat:
    return object.__new__(_Strat)


def _model() -> nn.Linear:
    model = nn.Linear(2, 2)
    model.weight.grad = torch.full_like(model.weight, 10.0)
    model.bias.grad = torch.full_like(model.bias, 10.0)
    return model


def test_clip_uses_the_standard_utility() -> None:
    model = _model()
    _strategy()._compute_gradient_norm(model, True, 1.0)
    total = torch.cat([model.weight.grad.flatten(), model.bias.grad.flatten()])
    assert total.norm().item() == pytest.approx(1.0, rel=1e-4)


def test_log_only_path_measures_without_clipping() -> None:
    model = _model()
    norm = _strategy()._compute_gradient_norm(model, False, 0.5)
    assert norm.item() == pytest.approx(10.0 * 6**0.5, rel=1e-5)
    assert torch.equal(model.weight.grad, torch.full_like(model.weight, 10.0))


def test_a_module_exposing_its_own_clip_is_not_special_cased() -> None:
    """The FSDP1 root's ``clip_grad_norm_`` branch is gone: one owner for the clip."""

    class _OwnClip(nn.Linear):
        def clip_grad_norm_(self, max_norm):  # pragma: no cover - must not be called
            raise AssertionError("the hook must not route through a module's own clip")

    model = _OwnClip(2, 2)
    model.weight.grad = torch.ones_like(model.weight)
    model.bias.grad = torch.ones_like(model.bias)
    _strategy()._compute_gradient_norm(model, True, 1.0)
