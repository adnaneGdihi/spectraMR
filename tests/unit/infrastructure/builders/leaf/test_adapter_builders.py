"""Tests for the shared pre_model application (train and validation)."""

from __future__ import annotations

import torch

from spectramr.infrastructure.builders.leaf.adapter_builders import apply_pre_model


def test_apply_pre_model_is_identity_without_a_chain() -> None:
    x, y = torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2)
    out_x, out_y = apply_pre_model({"pre_model": []}, x, y)
    assert out_x is x and out_y is y


def test_apply_pre_model_applies_the_chain_to_input_and_target() -> None:
    chains = {"pre_model": [lambda t: t + 1.0, lambda t: t * 2.0]}
    out_x, out_y = apply_pre_model(chains, torch.zeros(2), torch.ones(2))
    assert torch.equal(out_x, torch.full((2,), 2.0))
    assert torch.equal(out_y, torch.full((2,), 4.0))


def test_apply_pre_model_passes_none_through() -> None:
    chains = {"pre_model": [lambda t: t + 1.0]}
    out_x, out_y = apply_pre_model(chains, None, torch.zeros(1))
    assert out_x is None
    assert torch.equal(out_y, torch.ones(1))
