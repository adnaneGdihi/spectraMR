"""Tests for Heisenberg-group twisted convolution."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.heisenberg_conv import (
    HeisenbergConv2d,
    twisted_convolution_step,
)


def test_forward_shape_complex_output() -> None:
    layer = HeisenbergConv2d(in_channels=2, out_channels=2, spatial_size=(8, 8),
                             translation_radius=1, phase_radius=1)
    x = torch.randn(1, 2, 8, 8)
    out = layer(x)
    assert out.shape == (1, 2, 8, 8)
    assert out.is_complex()


def test_lattice_size_grows_with_radius() -> None:
    """The Heisenberg lattice has (2r+1)² * m² points."""
    small = HeisenbergConv2d(in_channels=2, out_channels=2, spatial_size=(8, 8),
                             translation_radius=1, phase_radius=1)
    large = HeisenbergConv2d(in_channels=2, out_channels=2, spatial_size=(8, 8),
                             translation_radius=2, phase_radius=2)
    # (2*1+1)² * 1² = 9 vs (2*2+1)² * 2² = 100.
    assert small.shifts.shape[0] == 9
    assert large.shifts.shape[0] == 100


def test_translation_equivariance_at_phase_radius_one() -> None:
    """At m=1 the phase ramps are all constant 1, so twisted conv reduces to
    a standard image-domain conv, and shift-equivariance holds modulo
    boundary roll."""
    torch.manual_seed(0)
    layer = HeisenbergConv2d(in_channels=1, out_channels=1, spatial_size=(8, 8),
                             translation_radius=1, phase_radius=1)
    x = torch.randn(1, 1, 8, 8)
    shifted_x = torch.roll(x, shifts=(2, 3), dims=(-2, -1))
    out_a = torch.roll(layer(x), shifts=(2, 3), dims=(-2, -1))
    out_b = layer(shifted_x)
    # Twisted conv with m=1 is shift-equivariant up to floating-point precision.
    assert torch.allclose(out_a.real, out_b.real, atol=1e-4)
    assert torch.allclose(out_a.imag, out_b.imag, atol=1e-4)


def test_gradient_flow() -> None:
    layer = HeisenbergConv2d(in_channels=2, out_channels=2, spatial_size=(4, 4),
                             translation_radius=1, phase_radius=1)
    x = torch.randn(1, 2, 4, 4, requires_grad=True)
    out = layer(x)
    # Sum over real + imag for a real-valued backprop target.
    (out.real.sum() + out.imag.sum()).backward()
    assert x.grad is not None
    assert layer.kernel_real.grad is not None
    assert layer.kernel_imag.grad is not None


def test_twisted_convolution_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError):
        twisted_convolution_step(
            torch.zeros(2, 2), torch.zeros(2, 2, 2), torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 4, 4)
        )
    with pytest.raises(ValueError):
        HeisenbergConv2d(in_channels=2, out_channels=2, spatial_size=(8, 8),
                         translation_radius=-1, phase_radius=1)


def test_zero_kernel_returns_zero_output() -> None:
    """A zero kernel should produce zero output regardless of input."""
    layer = HeisenbergConv2d(in_channels=2, out_channels=2, spatial_size=(8, 8),
                             translation_radius=1, phase_radius=1)
    with torch.no_grad():
        layer.kernel_real.zero_()
        layer.kernel_imag.zero_()
    x = torch.randn(1, 2, 8, 8)
    out = layer(x)
    assert torch.allclose(out.real, torch.zeros_like(out.real), atol=1e-6)
    assert torch.allclose(out.imag, torch.zeros_like(out.imag), atol=1e-6)


def test_shift_transfer_matches_per_point_item_reads() -> None:
    """One ``tolist()`` of the shifts gives the same output as reading each point with ``.item()``."""
    import torch

    from spectramr.models.blocks.heisenberg_conv import twisted_convolution_step as fn

    torch.manual_seed(0)
    psi = torch.randn(2, 3, 8, 8, dtype=torch.complex64)
    kernel = torch.randn(5, 3, 3, dtype=torch.complex64)
    shifts = torch.randint(-2, 3, (5, 2))
    ramps = torch.randn(5, 8, 8, dtype=torch.complex64)
    out = fn(psi, kernel, shifts, ramps)
    ref = torch.zeros_like(psi)
    for k in range(5):
        shifted = torch.roll(psi, (int(shifts[k, 0].item()), int(shifts[k, 1].item())), (-2, -1))
        ref = ref + torch.einsum("co,bohw->bchw", kernel[k], shifted * ramps[k])
    torch.testing.assert_close(out, ref)
