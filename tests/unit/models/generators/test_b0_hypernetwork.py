"""Tests for the B0-conditioned hypernetwork generator (differentiable forward)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.b0_hypernetwork import B0ConditionedGenerator, B0HyperNetwork


def _gen(**kw) -> B0ConditionedGenerator:
    return B0ConditionedGenerator(1, 1, ngf=8, n_blocks=1, hidden_dim=32, **kw)


def test_gradient_reaches_hypernetwork_from_loss() -> None:
    gen = _gen()
    x = torch.randn(2, 1, 16, 16)
    gen(x, field_strength=torch.tensor([0.064, 3.0])).square().mean().backward()
    grads = [p.grad for p in gen.hyper_network.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert gen.hyper_network.heads[0].weight.grad.abs().sum() > 0


def test_two_fields_give_different_outputs() -> None:
    torch.manual_seed(0)
    gen = _gen()
    x = torch.randn(1, 1, 16, 16).expand(2, 1, 16, 16).contiguous()
    y = gen(x, field_strength=torch.tensor([0.064, 3.0]))
    assert (y[0] - y[1]).abs().max() > 1e-4


def test_missing_field_raises() -> None:
    with pytest.raises(ValueError, match="requires field_strength"):
        _gen()(torch.randn(1, 1, 16, 16))


def test_field_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="entries for a batch"):
        _gen()(torch.randn(2, 1, 16, 16), field_strength=torch.tensor([0.064, 3.0, 1.5]))


def test_out_of_range_generated_tensor_count_raises() -> None:
    with pytest.raises(ValueError, match="n_generated_tensors"):
        _gen(n_generated_tensors=0)
    with pytest.raises(ValueError, match="n_generated_tensors"):
        _gen(n_generated_tensors=10_000)


def test_template_length_mismatch_raises() -> None:
    hyper = B0HyperNetwork(target_param_count=4, hidden_dim=8, num_layers=2)
    with pytest.raises(ValueError, match="template has 3 values"):
        hyper.init_from_template(torch.zeros(3))


def test_reverse_swaps_channels_and_keeps_architecture() -> None:
    gen = B0ConditionedGenerator(2, 1, ngf=8, n_blocks=1, hidden_dim=32)
    rev = gen.reverse(1, 2)
    assert (rev.in_channels, rev.out_channels) == (1, 2)
    assert rev._cfg == gen._cfg
    y = rev(torch.randn(2, 1, 16, 16), field_strength=1.0)
    assert y.shape == (2, 2, 16, 16)
