r"""Area-Law Matrix-Product-State (MPS) Attention.

Reshape the token sequence of length :math:`N=2^L` into an order-:math:`L`
tensor by binary indexing and decompose each attention head as a
matrix-product-state with bond dimension :math:`\chi`:

.. math::

    A_{i_1\ldots i_L,j_1\ldots j_L} \;=\; \sum_{\alpha_1,\ldots,\alpha_{L-1}}
        M^{(1)}_{i_1 j_1,\alpha_1}\,M^{(2)}_{\alpha_1,i_2 j_2,\alpha_2}\cdots M^{(L)}_{\alpha_{L-1},i_L j_L}.

Truncation error is controlled by the tail of the Schmidt singular values
[Schollwöck 2011]:

.. math::

    \|A-A^{(\chi)}\|_F^{2} \le \sum_{\ell=1}^{L-1}\sum_{\alpha>\chi}\sigma^{(\ell)}_\alpha{}^2.

For inputs satisfying an area law (Eisert-Cramer-Plenio 2010) the Schmidt
spectrum decays exponentially and :math:`\chi=O(\log(1/\varepsilon))`
suffices. Parameter count is :math:`O(L\chi^{2}d^{2})` vs :math:`O(N^{2})`
for full attention.

The implementation performs a left-to-right contraction sweep. The
2:4-structured-sparse Tensor-Core kernel from the parent proposal is
left as future work.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["MatrixProductStateAttention"]


class MatrixProductStateAttention(nn.Module):
    r"""Attention with MPS-compressed key-value bilinear form.

    Args:
        embed_dim: Feature dimension :math:`d`.
        n_log_tokens: :math:`L` such that token-length :math:`N=2^L`.
        bond_dim: MPS bond dimension :math:`\chi`. Default 4. Larger
            :math:`\chi` -> tighter approximation, more parameters.

    Shape:
        Input: ``(B, N=2^L, d)``.
        Output: ``(B, N, d)``.
    """

    def __init__(
        self,
        embed_dim: int,
        n_log_tokens: int,
        bond_dim: int = 4,
    ) -> None:
        super().__init__()
        if n_log_tokens < 1:
            raise ValueError("n_log_tokens must be >= 1.")
        if bond_dim < 1:
            raise ValueError("bond_dim must be >= 1.")
        self.embed_dim = embed_dim
        self.n_log_tokens = n_log_tokens
        self.token_len = 2**n_log_tokens
        self.bond_dim = bond_dim
        self.in_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        # MPS cores: one per binary level. The first and last have one bond
        # contracted with the identity; middle cores have two bonds.
        self.cores: nn.ParameterList = nn.ParameterList()
        scale = 1.0 / math.sqrt(bond_dim)
        for level in range(n_log_tokens):
            left = 1 if level == 0 else bond_dim
            right = 1 if level == n_log_tokens - 1 else bond_dim
            # Physical (i_l, j_l) index range is {0, 1} (binary).
            core = nn.Parameter(torch.randn(left, 2, 2, right) * scale)
            self.cores.append(core)

    def _contract_mps_action(self, v: torch.Tensor) -> torch.Tensor:
        r"""Apply the MPS attention operator :math:`A^{(\chi)}` to ``v``.

        Args:
            v: ``(B, N, d)`` tensor.

        Returns:
            ``(B, N, d)`` tensor, the MPS-contracted attention output.
        """
        b, n, d = v.shape
        # Carry layout (B, bond, I_out, J_rest, d): token bits are most-significant
        # first, level l consumes j_l from the front of J_rest and appends i_l as the
        # least-significant bit of I_out, so every core sees its own physical index.
        carry = v.reshape(b, 1, 1, n, d)
        for core in self.cores:
            right_bond = core.shape[-1]
            n_out, n_rest = carry.shape[2], carry.shape[3] // 2
            carry_split = carry.reshape(b, carry.shape[1], n_out, 2, n_rest, d)
            new_carry = torch.einsum("lijr,blojmd->broimd", core, carry_split)
            carry = new_carry.reshape(b, right_bond, 2 * n_out, n_rest, d)
        return carry.reshape(b, n, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (B, N, d) input; got {tuple(x.shape)}.")
        b, n, d = x.shape
        if n != self.token_len:
            raise ValueError(
                f"input token-len {n} != configured 2^{self.n_log_tokens} = {self.token_len}."
            )
        if d != self.embed_dim:
            raise ValueError(f"embed_dim mismatch: {d} vs {self.embed_dim}.")
        v = self.in_proj(x)
        out = self._contract_mps_action(v)
        return self.out_proj(out)

    @staticmethod
    def truncation_error_bound(singular_values: torch.Tensor, bond_dim: int) -> torch.Tensor:
        r"""Schmidt-tail bound :math:`\sum_{\alpha>\chi}\sigma^{2}_\alpha`.

        ``singular_values`` is a 1D tensor of Schmidt values sorted descending.
        """
        if bond_dim >= singular_values.numel():
            return singular_values.new_zeros(())
        return singular_values[bond_dim:].pow(2).sum()

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, n_log_tokens={self.n_log_tokens} "
            f"(N={self.token_len}), bond_dim={self.bond_dim}, "
            f"params_per_core={self.bond_dim * 4 * self.bond_dim}"
        )
