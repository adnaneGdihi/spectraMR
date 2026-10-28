"""Unit tests for CT-Mamba: per-stack Δ scales on DeltaMamba encoder and decoder."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.ct_mamba_generator import CTMamba
from spectramr.models.registry import is_strict_kwargs

BATCH, H, W = 2, 8, 8
D_MODEL, LAYERS, D_STATE = 16, 1, 4


def _model(**kw) -> CTMamba:
    torch.manual_seed(0)
    kw.setdefault("scan_backend", "reference")
    return CTMamba(1, 1, d_model=D_MODEL, num_layers=LAYERS, d_state=D_STATE, **kw)


@pytest.fixture
def input_2d() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(BATCH, 1, H, W)


def test_forward_shape(input_2d):
    assert _model()(input_2d).shape == (BATCH, 1, H, W)


def test_gradient_flow_reaches_both_stacks(input_2d):
    model = _model()
    model(input_2d).sum().backward()
    assert model.encoder.layers[0].in_proj.weight.grad is not None
    assert model.decoder.layers[0].in_proj.weight.grad is not None


@pytest.mark.parametrize("mode", ["hilbert_2d", "snake_2d", "zigzag_2d"])
def test_linearization_modes(mode, input_2d):
    model = _model(linearization_mode=mode)
    assert model(input_2d).shape == (BATCH, 1, H, W)


def test_non_pow2_shape_does_not_crash():
    """A full-FOV validation extent must run; the Hilbert linearizer handles any grid."""
    model = _model()
    assert model(torch.randn(1, 1, 66, 60)).shape == (1, 1, 66, 60)


def test_delta_scales_change_the_output(input_2d):
    base = _model()(input_2d)
    changed = _model(delta_scale_decoder=0.25)(input_2d)
    assert not torch.allclose(base, changed)


def test_default_scales_are_stored_per_channel():
    model = _model()
    d_inner = model.encoder.layers[0].d_inner
    assert model.log_delta_scale_encoder.shape == (d_inner,)
    torch.testing.assert_close(model.log_delta_scale_encoder.exp(), torch.ones(d_inner))
    torch.testing.assert_close(model.log_delta_scale_decoder.exp(), torch.full((d_inner,), 0.5))


def test_learnable_scale_is_a_parameter_and_fixed_scale_is_a_buffer():
    frozen = _model(learnable_delta_scale=False)
    learned = _model(learnable_delta_scale=True)
    assert "log_delta_scale_encoder" in dict(frozen.named_buffers())
    assert "log_delta_scale_encoder" in dict(learned.named_parameters())


def test_freeze_dt_bias_stops_its_gradient(input_2d):
    model = _model(freeze_dt_bias=True)
    assert not model.encoder.layers[0].dt_proj.bias.requires_grad
    assert not model.decoder.layers[0].dt_proj.bias.requires_grad
    free = _model(freeze_dt_bias=False)
    assert free.encoder.layers[0].dt_proj.bias.requires_grad


@pytest.mark.parametrize(
    ("enc", "dec"),
    [(1.0, 1.5), (0.0, 0.0), (1.0, 0.0), (-1.0, -0.5), (float("nan"), 0.5)],
)
def test_invalid_scales_raise(enc, dec):
    with pytest.raises(ValueError, match="delta_scale"):
        _model(delta_scale_encoder=enc, delta_scale_decoder=dec)


def test_equal_scales_are_allowed_for_the_unit_control():
    _model(delta_scale_encoder=1.0, delta_scale_decoder=1.0)


def test_unknown_scan_backend_raises():
    with pytest.raises(ValueError, match="backend"):
        _model(scan_backend="gru")


def test_ct_mamba_is_registered_strict():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    assert is_strict_kwargs("ct_mamba")
