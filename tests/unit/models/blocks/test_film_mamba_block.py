"""Unit tests for FiLMMambaBlock.

Pin shape correctness, FiLM modulation actually changes outputs,
and gradients flow through both the FiLM MLP and the Mamba state.
Forward/backward cases carry ``requires_cuda_for_mamba``: with ``mamba_ssm``
installed, MambaBlock dispatches to its CUDA-only kernel and raises
``Expected x.is_cuda() to be true`` on CPU tensors.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.blocks.film_mamba_block import FiLMMambaBlock  # noqa: E402
from tests.utils.optional_backends import requires_cuda_for_mamba  # noqa: E402


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@requires_cuda_for_mamba
def test_output_shape_matches_input() -> None:
    dev = _device()
    block = FiLMMambaBlock(d_model=16, contrast_dim=4).to(dev)
    x = torch.randn(2, 32, 16, device=dev)
    c = torch.randn(2, 4, device=dev)
    y = block(x, c)
    assert y.shape == x.shape


@requires_cuda_for_mamba
def test_contrast_embedding_actually_modulates_output() -> None:
    """Different contrast embeddings must give different outputs for the same x."""
    dev = _device()
    block = FiLMMambaBlock(d_model=16, contrast_dim=4).to(dev)
    x = torch.randn(1, 32, 16, device=dev)
    c1 = torch.zeros(1, 4, device=dev)
    c2 = torch.ones(1, 4, device=dev)
    y1 = block(x, c1)
    y2 = block(x, c2)
    assert not torch.allclose(y1, y2, atol=1e-5)


@requires_cuda_for_mamba
def test_grad_flows_to_film_mlp_and_mamba() -> None:
    dev = _device()
    block = FiLMMambaBlock(d_model=16, contrast_dim=4).to(dev)
    x = torch.randn(2, 32, 16, device=dev, requires_grad=True)
    c = torch.randn(2, 4, device=dev, requires_grad=True)
    y = block(x, c)
    y.sum().backward()
    assert x.grad is not None and torch.any(x.grad != 0)
    assert c.grad is not None and torch.any(c.grad != 0)
    # FiLM MLP weights must also receive gradient.
    film_first_layer = block.film[0]
    assert film_first_layer.weight.grad is not None
    assert torch.any(film_first_layer.weight.grad != 0)


def test_rejects_wrong_x_shape() -> None:
    block = FiLMMambaBlock(d_model=16, contrast_dim=4)
    bad_x = torch.randn(2, 32, 8)  # wrong d_model
    c = torch.randn(2, 4)
    # Error message format: "Expected x of shape (B, L, <d_model>), got <shape>"
    with pytest.raises(ValueError, match=r"\(B, L, 16\)"):
        block(bad_x, c)


def test_rejects_wrong_contrast_shape() -> None:
    block = FiLMMambaBlock(d_model=16, contrast_dim=4)
    x = torch.randn(2, 32, 16)
    bad_c = torch.randn(2, 8)  # wrong contrast_dim
    with pytest.raises(ValueError, match="contrast_dim|contrast_embedding"):
        block(x, bad_c)


def test_rejects_batch_mismatch() -> None:
    block = FiLMMambaBlock(d_model=16, contrast_dim=4)
    x = torch.randn(2, 32, 16)
    bad_c = torch.randn(3, 4)  # batch size mismatch
    with pytest.raises(ValueError, match="contrast_emb batch 3"):
        block(x, bad_c)
