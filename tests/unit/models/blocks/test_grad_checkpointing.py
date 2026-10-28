"""The checkpointing mixins: the switch reaches the stacks, and an empty model refuses."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.blocks import delta_mamba
from spectramr.models.blocks.grad_checkpointing import (
    GradCheckpointingMixin,
    NestedGradCheckpointingMixin,
)


class _Leaf(GradCheckpointingMixin, nn.Module):
    pass


class _Owner(NestedGradCheckpointingMixin, nn.Module):
    def __init__(self, n_leaves: int) -> None:
        super().__init__()
        self.leaves = nn.ModuleList(_Leaf() for _ in range(n_leaves))


def test_nested_switch_reaches_every_child() -> None:
    owner = _Owner(2)
    owner.set_grad_checkpointing(True)
    assert all(leaf.grad_checkpointing for leaf in owner.leaves)
    owner.set_grad_checkpointing(False)
    assert not any(leaf.grad_checkpointing for leaf in owner.leaves)


def test_nested_switch_without_children_raises() -> None:
    with pytest.raises(NotImplementedError, match="_Owner has no submodule"):
        _Owner(0).set_grad_checkpointing(True)


def _grads(stack: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    stack.zero_grad(set_to_none=True)
    stack(x).square().sum().backward()
    return [p.grad.clone() for p in stack.parameters() if p.grad is not None]


def test_delta_mamba_stack_checkpoints_each_layer_and_keeps_the_gradients(monkeypatch) -> None:
    torch.manual_seed(0)
    stack = delta_mamba.DeltaMambaStack(8, num_layers=2, d_state=4, backend="reference").train()
    x = torch.randn(1, 16, 8)
    plain = _grads(stack, x)

    calls = []
    real = delta_mamba.checkpoint

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(delta_mamba, "checkpoint", counting)
    stack.set_grad_checkpointing(True)
    checkpointed = _grads(stack, x)

    assert len(calls) == 2
    for a, b in zip(plain, checkpointed, strict=True):
        torch.testing.assert_close(a, b)


def test_delta_mamba_stack_does_not_checkpoint_under_eval(monkeypatch) -> None:
    stack = delta_mamba.DeltaMambaStack(8, num_layers=2, d_state=4, backend="reference").eval()
    monkeypatch.setattr(delta_mamba, "checkpoint", lambda *a, **k: pytest.fail("checkpointed"))
    stack.set_grad_checkpointing(True)
    with torch.no_grad():
        stack(torch.randn(1, 16, 8))
