"""Tests for MatrixProductStateAttention."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.mps_attention import MatrixProductStateAttention


def test_forward_shape_preserved() -> None:
    block = MatrixProductStateAttention(embed_dim=8, n_log_tokens=4, bond_dim=2)
    x = torch.randn(2, 16, 8)
    out = block(x)
    assert out.shape == x.shape


def test_token_len_validation() -> None:
    block = MatrixProductStateAttention(embed_dim=8, n_log_tokens=4)
    with pytest.raises(ValueError):
        block(torch.randn(1, 15, 8))  # 15 is not 2^4
    with pytest.raises(ValueError):
        block(torch.randn(1, 16, 5))


def test_bond_dim_validation() -> None:
    with pytest.raises(ValueError):
        MatrixProductStateAttention(embed_dim=8, n_log_tokens=4, bond_dim=0)
    with pytest.raises(ValueError):
        MatrixProductStateAttention(embed_dim=8, n_log_tokens=0)


def test_parameter_count_scales_with_bond_dim() -> None:
    """Parameter count is O(L χ² d²) — linear in L, quadratic in χ."""
    block_small = MatrixProductStateAttention(embed_dim=4, n_log_tokens=4, bond_dim=2)
    block_large = MatrixProductStateAttention(embed_dim=4, n_log_tokens=4, bond_dim=4)
    n_small = sum(p.numel() for p in block_small.cores)
    n_large = sum(p.numel() for p in block_large.cores)
    # χ goes 2 → 4 quadruples middle-core count, doubles boundary cores.
    assert n_large > 2 * n_small
    assert n_large < 5 * n_small


def test_gradient_flow_through_cores() -> None:
    block = MatrixProductStateAttention(embed_dim=4, n_log_tokens=3, bond_dim=2)
    x = torch.randn(1, 8, 4, requires_grad=True)
    out = block(x)
    out.sum().backward()
    for core in block.cores:
        assert core.grad is not None


def test_truncation_error_bound() -> None:
    """The Schmidt-tail bound returns the sum of squared truncated singular values."""
    sv = torch.tensor([1.0, 0.5, 0.25, 0.1])
    err = MatrixProductStateAttention.truncation_error_bound(sv, bond_dim=2)
    expected = 0.25 ** 2 + 0.1 ** 2
    assert err.item() == pytest.approx(expected, abs=1e-6)


def test_truncation_at_full_rank_is_zero() -> None:
    sv = torch.tensor([1.0, 0.5, 0.25])
    err = MatrixProductStateAttention.truncation_error_bound(sv, bond_dim=3)
    assert err.item() == 0.0


def test_invalid_input_shape() -> None:
    block = MatrixProductStateAttention(embed_dim=4, n_log_tokens=3, bond_dim=2)
    with pytest.raises(ValueError):
        block(torch.randn(2, 8))  # missing token axis


def _identity_projections(block: MatrixProductStateAttention) -> None:
    with torch.no_grad():
        block.in_proj.weight.copy_(torch.eye(block.embed_dim))
        block.in_proj.bias.zero_()
        block.out_proj.weight.copy_(torch.eye(block.embed_dim))
        block.out_proj.bias.zero_()


def _dense_operator(block: MatrixProductStateAttention) -> torch.Tensor:
    """A[i, j] = prod_l M^(l)[a_{l-1}, i_l, j_l, a_l], summed over bonds; bit 1 most significant."""
    n_bits, n = block.n_log_tokens, block.token_len
    a = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            vec = torch.ones(1, dtype=torch.float64)
            for level, core in enumerate(block.cores):
                shift = n_bits - 1 - level
                vec = vec @ core.detach().double()[:, (i >> shift) & 1, (j >> shift) & 1, :]
            a[i, j] = vec.sum()
    return a


def _applied_operator(block: MatrixProductStateAttention) -> torch.Tensor:
    n = block.token_len
    eye = torch.eye(n, dtype=torch.float64)
    with torch.no_grad():
        return torch.stack([block(eye[j].view(1, n, 1))[0, :, 0] for j in range(n)], 1)


@pytest.mark.parametrize("n_log_tokens,bond_dim", [(3, 1), (3, 4), (4, 2)])
def test_operator_equals_the_dense_mps_contraction(n_log_tokens: int, bond_dim: int) -> None:
    """#588: the sweep contracted bit 1 at every level, giving kron(M_2x2, I)."""
    torch.manual_seed(1)
    block = MatrixProductStateAttention(1, n_log_tokens=n_log_tokens, bond_dim=bond_dim).double()
    _identity_projections(block)
    assert torch.allclose(_applied_operator(block), _dense_operator(block), atol=1e-12)


def test_operator_is_not_a_leading_bit_kron() -> None:
    torch.manual_seed(1)
    block = MatrixProductStateAttention(1, n_log_tokens=3, bond_dim=4).double()
    _identity_projections(block)
    a = _applied_operator(block)
    m = a[::4, ::4]
    assert not torch.allclose(a, torch.kron(m, torch.eye(4, dtype=torch.float64)), atol=1e-6)
    # Tokens 0 and 1 differ only in the least-significant bit and must interact.
    assert a[0, 1].abs() > 1e-6
