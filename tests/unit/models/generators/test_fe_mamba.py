"""Unit tests for FE-Mamba: ULF-grid pooling, Kronecker expansion to the HF grid."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.models.blocks.topology_linearizer import _rect_hilbert_2d_indices
from spectramr.models.generators.fe_mamba import FEMamba
from spectramr.models.registry import is_strict_kwargs

BATCH, D_MODEL, LAYERS, D_STATE = 2, 16, 1, 4


def _model(**kw) -> FEMamba:
    torch.manual_seed(0)
    return FEMamba(1, 1, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE, **kw)


@pytest.mark.parametrize(("s", "hw"), [(1, (8, 8)), (2, (16, 16)), (4, (16, 16))])
def test_output_is_on_the_input_grid(s, hw):
    x = torch.randn(BATCH, 1, *hw)
    assert _model(expansion_factor=s)(x).shape == x.shape


@pytest.mark.parametrize(("s", "hw"), [(2, (16, 16)), (4, (16, 16)), (2, (12, 20))])
def test_children_of_a_ulf_token_fill_its_s_by_s_block(s, hw):
    """Fine positions s^2 k .. s^2 k + s^2 - 1 all lie in ULF cell coarse[k]."""
    h, w = hw
    fine = _rect_hilbert_2d_indices(h, w)
    coarse = _rect_hilbert_2d_indices(h // s, w // s)
    rows, cols = np.divmod(fine, w)
    cell = (rows // s) * (w // s) + cols // s  # ULF cell of each HF token
    for k in range(coarse.size):
        assert np.all(cell[k * s * s : (k + 1) * s * s] == coarse[k])


def test_expansion_width_follows_the_factor():
    assert _model(expansion_factor=2).expand_proj.out_features == D_MODEL * 4
    assert _model(expansion_factor=1).expand_proj.out_features == D_MODEL


def test_factor_one_is_the_identity_grid_control():
    model = _model(expansion_factor=1)
    x = torch.randn(BATCH, 1, 8, 8)
    assert model(x).shape == (BATCH, 1, 8, 8)


def test_gradients_reach_the_expansion():
    model = _model(expansion_factor=2)
    model(torch.randn(BATCH, 1, 8, 8)).sum().backward()
    assert model.expand_proj.weight.grad is not None
    assert model.expand_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [3, 0, 6])
def test_non_power_of_two_factor_raises(bad):
    with pytest.raises(ValueError, match="power of two"):
        _model(expansion_factor=bad)


def test_grid_not_divisible_by_the_factor_raises():
    with pytest.raises(ValueError, match="divisible"):
        _model(expansion_factor=4)(torch.randn(1, 1, 6, 6))


def test_fe_mamba_is_registered_strict():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    assert is_strict_kwargs("fe_mamba")
