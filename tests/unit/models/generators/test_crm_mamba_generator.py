"""Unit tests for CRM-Mamba: a ULF-coarse context that conditions the HF decoder."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.crm_mamba_generator import (
    CRMMamba,
    _broadcast_parents,
    _pool_children,
)
from spectramr.models.registry import is_strict_kwargs

BATCH, H, W = 2, 8, 8
D_MODEL, LAYERS, D_STATE = 16, 2, 4


def _model(**kw) -> CRMMamba:
    torch.manual_seed(0)
    kw.setdefault("scan_backend", "reference")
    return CRMMamba(1, 1, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE, **kw)


@pytest.fixture
def input_2d() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, 1, H, W)


def test_forward_shape(input_2d):
    assert _model()(input_2d).shape == (BATCH, 1, H, W)


def test_children_of_a_parent_share_one_state():
    seq = torch.arange(16.0).reshape(1, 16, 1)
    coarse = _pool_children(seq)  # [1, 4, 1]
    fine = _broadcast_parents(coarse)  # [1, 16, 1]
    blocks = fine.reshape(1, 4, 4, 1)
    assert torch.all(blocks == blocks[:, :, :1, :])
    torch.testing.assert_close(_pool_children(fine), coarse)


def test_delta_site_starts_as_the_unconditioned_decoder(input_2d):
    model = _model(conditioning_site="delta")
    for proj in model.delta_shift_proj:
        assert torch.count_nonzero(proj.weight) == 0 and torch.count_nonzero(proj.bias) == 0
    assert _model(conditioning_site="delta")(input_2d).shape == (BATCH, 1, H, W)


def test_ulf_context_changes_the_output_once_the_shift_is_nonzero(input_2d):
    model = _model(conditioning_site="delta")
    with torch.no_grad():
        for proj in model.delta_shift_proj:
            proj.weight.normal_(std=0.1)
    base = model(input_2d)
    other = input_2d.clone()
    other[..., :4, :4] += 3.0  # changes only the first coarse parent's block
    assert not torch.allclose(base, model(other))


def test_one_shift_per_decoder_layer():
    model = _model(conditioning_site="delta")
    assert len(model.delta_shift_proj) == LAYERS


def test_input_gate_control_has_no_delta_projection_and_runs(input_2d):
    model = _model(conditioning_site="input_gate")
    assert not hasattr(model, "delta_shift_proj")
    assert model(input_2d).shape == (BATCH, 1, H, W)


def test_gradients_reach_the_shift_projection(input_2d):
    model = _model(conditioning_site="delta")
    with torch.no_grad():
        for proj in model.delta_shift_proj:
            proj.weight.normal_(std=0.1)
    model(input_2d).sum().backward()
    assert model.delta_shift_proj[0].weight.grad.abs().sum() > 0


def test_odd_grid_raises(input_2d):
    with pytest.raises(ValueError, match="even H and W"):
        _model()(torch.randn(1, 1, 7, 8))


def test_unknown_conditioning_site_raises():
    with pytest.raises(ValueError, match="conditioning_site"):
        _model(conditioning_site="cross_attention")


def test_crm_mamba_is_registered_strict():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    assert is_strict_kwargs("crm_mamba")
