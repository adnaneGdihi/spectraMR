"""Physics tests for the shared CG data-consistency primitive (SOTA plan T1).

``cg_data_consistency`` solves the Tikhonov normal equations

    (Aᴴ A + λ I) x = Aᴴ y + λ x₀

by conjugate gradient, initialised at ``x₀``. It is the shared k-space
data-consistency solver consumed by DDS / SSDU / EI / Ambient (extracted and
generalised from the inline ``DDSSampler._cg_solve``). The generalisation over
the inline version: it accepts arbitrary linear ``forward_op``/``adjoint_op``
callables AND uses the Hermitian inner product so it is correct on COMPLEX
k-space (the inline ``(r*r).sum()`` is wrong for complex tensors).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.tangent_cg import cg_data_consistency  # noqa: E402


def test_cg_identity_operator_matches_closed_form() -> None:
    """A = I ⇒ (1+λ)x = y + λx₀ ⇒ x = (y + λx₀)/(1+λ)."""
    torch.manual_seed(0)
    y = torch.randn(3, 1, 8, 8)
    x0 = torch.randn(3, 1, 8, 8)
    lam = 0.3
    ident = lambda t: t  # noqa: E731
    out = cg_data_consistency(x0, ident, ident, y, n_iter=20, lam=lam)
    expected = (y + lam * x0) / (1.0 + lam)
    assert torch.allclose(out, expected, atol=1e-5)


def test_cg_diagonal_operator_matches_closed_form() -> None:
    """A = diag(d) ⇒ x = (d·y + λx₀)/(d² + λ) elementwise."""
    torch.manual_seed(1)
    d = torch.rand(1, 1, 8, 8) + 0.5  # strictly positive, well-conditioned
    y = torch.randn(2, 1, 8, 8)
    x0 = torch.randn(2, 1, 8, 8)
    lam = 0.1
    fwd = lambda t: d * t  # noqa: E731
    adj = lambda t: d * t  # noqa: E731  (real diagonal is self-adjoint)
    out = cg_data_consistency(x0, fwd, adj, y, n_iter=30, lam=lam)
    expected = (d * y + lam * x0) / (d * d + lam)
    assert torch.allclose(out, expected, atol=1e-5)


def test_cg_solves_dense_least_squares_against_direct_solve() -> None:
    """For a dense operator, CG must match torch.linalg.solve on the normal equations."""
    torch.manual_seed(2)
    n = 12
    A = torch.randn(n, n)
    y = torch.randn(n, 1)
    x0 = torch.zeros(n, 1)
    lam = 0.05
    fwd = lambda t: A @ t  # noqa: E731
    adj = lambda t: A.T @ t  # noqa: E731
    # AᵀA from a random Gaussian A is ill-conditioned (cond ~1e3), so CG needs
    # more than n iterations in floating point to converge; the budget here is
    # about convergence, not correctness (the closed-form tests pin correctness).
    out = cg_data_consistency(x0, fwd, adj, y, n_iter=200, lam=lam, tol=1e-12)
    normal = A.T @ A + lam * torch.eye(n)
    rhs = A.T @ y + lam * x0
    expected = torch.linalg.solve(normal, rhs)
    assert torch.allclose(out, expected, atol=1e-4)


def test_cg_complex_kspace_operator_converges() -> None:
    """Complex operator: CG must use the Hermitian inner product and converge.

    A real-only inner product (``(r*r).sum()``) diverges or stalls on complex
    data; this guards the generalisation over the inline DDS solver.
    """
    torch.manual_seed(3)
    n = 10
    Ar = torch.randn(n, n)
    Ai = torch.randn(n, n)
    A = torch.complex(Ar, Ai)
    y = torch.complex(torch.randn(n, 1), torch.randn(n, 1))
    x0 = torch.zeros(n, 1, dtype=torch.complex64)
    lam = 0.1
    fwd = lambda t: A @ t  # noqa: E731
    adj = lambda t: A.conj().T @ t  # noqa: E731  (Hermitian adjoint)
    out = cg_data_consistency(x0, fwd, adj, y, n_iter=2 * n, lam=lam, tol=1e-10)
    normal = A.conj().T @ A + lam * torch.eye(n, dtype=torch.complex64)
    rhs = A.conj().T @ y + lam * x0
    expected = torch.linalg.solve(normal, rhs)
    assert torch.allclose(out, expected, atol=1e-3)


def test_cg_rejects_nonpositive_iterations() -> None:
    """n_iter <= 0 is a contract violation, not a silent no-op."""
    ident = lambda t: t  # noqa: E731
    with pytest.raises(ValueError):
        cg_data_consistency(
            torch.zeros(1, 1, 4, 4), ident, ident, torch.zeros(1, 1, 4, 4), n_iter=0
        )
