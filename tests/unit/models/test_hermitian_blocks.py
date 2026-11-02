"""Unit tests for Hermitian-equivariant blocks and models (Idea 1)."""

from __future__ import annotations

import torch

from spectramr.models.blocks.hermitian_equivariant import (
    HermitianFNOBlock,
    HermitianSpectralConv2d,
)


def test_hermitian_spectral_conv_forward_shape() -> None:
    layer = HermitianSpectralConv2d(in_channels=4, out_channels=8, modes1=6, modes2=6)
    x = torch.randn(2, 4, 32, 32)
    y = layer(x)
    assert y.shape == (2, 8, 32, 32)


def test_hermitian_fno_block_forward_shape() -> None:
    block = HermitianFNOBlock(width=16, modes1=6, modes2=6)
    x = torch.randn(2, 16, 32, 32)
    y = block(x)
    assert y.shape == (2, 16, 32, 32)


def test_parameter_count_against_complex_baseline() -> None:
    """Hermitian conv parameter count: 2 * C_in * C_out * modes1 * modes2.

    A baseline ``SpectralConv2d`` that stores a complex tensor of the
    same shape has equivalent storage (one complex value = two real
    values), so we test the absolute parameter count is exactly the
    sum of two real half-spectrum tensors of the stated shape.
    """
    C_in, C_out, m1, m2 = 8, 8, 6, 6
    layer = HermitianSpectralConv2d(C_in, C_out, m1, m2)
    expected = 2 * (C_in * C_out * m1 * m2)
    actual = sum(p.numel() for p in layer.parameters())
    assert actual == expected


def test_gradient_flows() -> None:
    layer = HermitianSpectralConv2d(in_channels=2, out_channels=2, modes1=4, modes2=4)
    x = torch.randn(1, 2, 16, 16, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None
    assert layer.w_real.grad is not None
    assert layer.w_imag.grad is not None


def test_models_register_in_registry() -> None:
    """Ensure the @register_model decorator fires at import time."""
    import spectramr.models.generators.hermitian_fno_unet  # noqa: F401
    from spectramr.models.registry import get_model_class

    cls_fno = get_model_class("hermitian_fno")
    cls_unet = get_model_class("hermitian_kspace_unet")
    assert cls_fno is not None
    assert cls_unet is not None


def _flip_error(module: torch.nn.Module, x: torch.Tensor, dims: tuple[int, ...]) -> float:
    with torch.no_grad():
        y = module(x)
        return ((module(x.flip(dims)) - y.flip(dims)).norm() / y.norm()).item()


def test_spectral_conv_is_width_flip_equivariant() -> None:
    """#2376: the old tie paired row i with frequency i - modes1, so no flip commuted."""
    torch.manual_seed(0)
    layer = HermitianSpectralConv2d(3, 4, modes1=6, modes2=5).double()
    for h, w in ((24, 20), (23, 19)):
        x = torch.randn(2, 3, h, w, dtype=torch.float64)
        assert _flip_error(layer, x, (-1,)) < 1e-12


def test_fno_block_is_width_flip_equivariant() -> None:
    torch.manual_seed(0)
    block = HermitianFNOBlock(width=8, modes1=6, modes2=6).double()
    x = torch.randn(2, 8, 32, 28, dtype=torch.float64)
    assert _flip_error(block, x, (-1,)) < 1e-12


def test_imaginary_part_of_row_zero_is_inert() -> None:
    torch.manual_seed(0)
    layer = HermitianSpectralConv2d(2, 2, modes1=4, modes2=4).double()
    x = torch.randn(1, 2, 16, 16, dtype=torch.float64)
    with torch.no_grad():
        y = layer(x)
        layer.w_imag[:, :, 0].normal_()
        assert torch.allclose(layer(x), y, atol=1e-14)
