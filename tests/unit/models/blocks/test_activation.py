"""Tests for activation modules.

Targets ``spectramr.models.blocks.activation``:

- ``swish_jit``, ``mish_jit`` — pure-function activations
- ``Swish``, ``Mish`` — nn.Module wrappers
- ``complex_mod_relu_jit`` — phase-preserving ReLU on magnitudes
- ``ComplexActivation`` — learnable-bias ModReLU module
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.activation import (
    ComplexActivation,
    Mish,
    Swish,
    complex_mod_relu_jit,
    mish_jit,
    swish_jit,
)


# ---------------------------------------------------------------------------
# swish / mish — pure functions
# ---------------------------------------------------------------------------


def test_swish_at_zero_is_zero() -> None:
    """``swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0``."""
    out = swish_jit(torch.zeros(4))
    assert torch.allclose(out, torch.zeros(4))


def test_swish_matches_explicit_formula() -> None:
    """``swish(x) = x * sigmoid(x)``."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = swish_jit(x)
    expected = x * torch.sigmoid(x)
    assert torch.allclose(out, expected)


def test_mish_at_zero_is_zero() -> None:
    """``mish(0) = 0 * tanh(softplus(0)) = 0``."""
    out = mish_jit(torch.zeros(4))
    assert torch.allclose(out, torch.zeros(4))


def test_mish_matches_explicit_formula() -> None:
    """``mish(x) = x * tanh(softplus(x))``."""
    x = torch.tensor([-2.0, 0.0, 2.0])
    out = mish_jit(x)
    expected = x * torch.tanh(torch.nn.functional.softplus(x))
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# Module wrappers
# ---------------------------------------------------------------------------


def test_swish_module_matches_jit() -> None:
    """``Swish.forward`` matches ``swish_jit`` exactly."""
    swish = Swish()
    x = torch.randn(2, 3, 4, 4)
    assert torch.allclose(swish(x), swish_jit(x))


def test_mish_module_matches_jit() -> None:
    """``Mish.forward`` matches ``mish_jit`` exactly."""
    mish = Mish()
    x = torch.randn(2, 3, 4, 4)
    assert torch.allclose(mish(x), mish_jit(x))


def test_swish_gradient_flows() -> None:
    """Swish is differentiable."""
    x = torch.randn(4, requires_grad=True)
    Swish()(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_mish_gradient_flows() -> None:
    """Mish is differentiable."""
    x = torch.randn(4, requires_grad=True)
    Mish()(x).sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# Complex ModReLU
# ---------------------------------------------------------------------------


def test_complex_mod_relu_zero_input_zero_output() -> None:
    """Zero magnitude with zero bias → zero output."""
    real = torch.zeros(1, 2, 4, 4)
    imag = torch.zeros(1, 2, 4, 4)
    bias = torch.zeros(1, 2, 1, 1)
    out = complex_mod_relu_jit(real, imag, bias)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_complex_mod_relu_preserves_phase() -> None:
    """For positive magnitude with positive bias, phase is preserved."""
    real = torch.tensor([[[[3.0]]], [[[0.0]]]]).unsqueeze(0)  # (1, 2, 1, 1)
    imag = torch.tensor([[[[4.0]]], [[[5.0]]]]).unsqueeze(0)  # (1, 2, 1, 1)
    bias = torch.zeros(1, 2, 1, 1)
    out = complex_mod_relu_jit(real, imag, bias)
    # Output is interleaved [R1, I1, R2, I2]; magnitude should equal |z| + b = |z|.
    out_real_0 = out[0, 0, 0, 0].item()
    out_imag_0 = out[0, 1, 0, 0].item()
    mag = (out_real_0**2 + out_imag_0**2) ** 0.5
    assert pytest.approx(mag, abs=1e-4) == 5.0  # |3+4j| = 5


def test_complex_mod_relu_negative_bias_zeroes_small_magnitudes() -> None:
    """Bias = -|z| → ReLU(|z| - |z|) = 0 → output is zero."""
    real = torch.tensor([[[[1.0]]]])  # (1, 1, 1, 1)
    imag = torch.tensor([[[[0.0]]]])
    bias = torch.tensor([[[[-2.0]]]])  # bias = -|z|·2 (forces zero)
    out = complex_mod_relu_jit(real, imag, bias)
    assert out.abs().max().item() < 1e-6


# ---------------------------------------------------------------------------
# ComplexActivation module
# ---------------------------------------------------------------------------


def test_complex_activation_module_shape() -> None:
    """``ComplexActivation`` preserves input shape ``(B, 2C, H, W)``."""
    act = ComplexActivation(channels=4, bias_init=0.1)
    x = torch.randn(2, 8, 16, 16)
    out = act(x)
    assert out.shape == x.shape


def test_complex_activation_bias_is_learnable() -> None:
    """``bias`` is a registered parameter (gradient flows)."""
    act = ComplexActivation(channels=2, bias_init=0.5)
    assert act.bias.requires_grad is True
    assert act.bias.shape == (1, 2, 1, 1)


def test_complex_activation_gradient_flows() -> None:
    """Backward through ComplexActivation reaches input + bias."""
    act = ComplexActivation(channels=2, bias_init=0.0)
    x = torch.randn(1, 4, 4, 4, requires_grad=True)
    out = act(x).sum()
    out.backward()
    assert x.grad is not None
    assert act.bias.grad is not None


def test_complex_activation_positive_bias_adds_no_kspace_floor() -> None:
    """A positive bias must not lift k-space bins into a flat floor (image-centre Dirac)."""
    act = ComplexActivation(channels=1, bias_init=0.5)
    x = torch.full((1, 2, 8, 8), 1e-3)
    assert torch.allclose(act(x), ComplexActivation(channels=1)(x))
