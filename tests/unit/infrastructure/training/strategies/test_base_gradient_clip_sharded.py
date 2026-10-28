"""``_compute_gradient_norm`` must clip through an FSDP root's own method.

The training loop always passes ``strategy._clip_and_log_gradients`` as the clip
hook, and ``AMPPolicy`` prefers that hook to ``policy.clip_gradients``. So the
FSDP-aware override in ``FSDPStepPolicy`` is bypassed unless the hook itself
reduces the norm across shards; the plain utility clips each rank's shard by its
own local norm.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy


class _Strat(BaseTrainingStrategy):
    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - stub
        return {}


class _FakeFSDP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)
        self.calls: list[float] = []

    def clip_grad_norm_(self, max_norm):
        self.calls.append(max_norm)
        return torch.tensor(3.0)


def _strategy() -> _Strat:
    return object.__new__(_Strat)


def test_sharded_model_clips_through_its_own_method() -> None:
    model = _FakeFSDP()
    norm = _strategy()._compute_gradient_norm(model, True, 0.5)
    assert model.calls == [0.5]
    assert norm.item() == 3.0


def test_log_only_path_uses_an_infinite_threshold_on_the_sharded_method() -> None:
    model = _FakeFSDP()
    _strategy()._compute_gradient_norm(model, False, 0.5)
    assert model.calls == [float("inf")]


def test_plain_module_still_uses_the_standard_utility() -> None:
    model = nn.Linear(2, 2)
    model.weight.grad = torch.full_like(model.weight, 10.0)
    model.bias.grad = torch.full_like(model.bias, 10.0)
    _strategy()._compute_gradient_norm(model, True, 1.0)
    total = torch.cat([model.weight.grad.flatten(), model.bias.grad.flatten()])
    assert total.norm().item() == pytest.approx(1.0, rel=1e-4)
