"""Gradients of the NUFFT with respect to the trajectory ω, against an exact NDFT."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.physics.implementations.nufft_operator import (
    TORCHKBNUFFT_AVAILABLE,
    NUFFTOperator,
)
from spectramr.infrastructure.physics.nufft_omega_grad import (
    _centred_coords,
    nufft_adjoint_omega,
    nufft_forward_omega,
)

H = W = 8
N = 16
DT = torch.float64


def _exact_ops():
    """Direct sums in the operator's convention: y_m = sum_n x_n exp(-i w_m . r_n)."""
    r = _centred_coords(H, W, torch.device("cpu"), DT).reshape(2, -1)

    def fwd(x, w):
        phase = torch.exp(-1j * (w.T @ r))
        return torch.einsum("bcn,mn->bcm", x.reshape(*x.shape[:2], -1), phase)

    def adj(y, w):
        phase = torch.exp(-1j * (w.T @ r))
        return torch.einsum("bcm,mn->bcn", y, phase.conj()).reshape(*y.shape[:2], H, W)

    return SimpleNamespace(nufft_obj=fwd, adj_nufft_obj=adj)


def _inputs(seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 2, H, W, dtype=torch.complex128, generator=gen).requires_grad_(True)
    s = torch.randn(1, 2, N, dtype=torch.complex128, generator=gen).requires_grad_(True)
    w = ((torch.rand(2, N, dtype=DT, generator=gen) * 2 - 1) * math.pi).requires_grad_(True)
    return x, s, w


def test_forward_gradient_wrt_omega_matches_the_exact_ndft_gradcheck():
    x, _, w = _inputs()
    ops = _exact_ops()
    assert torch.autograd.gradcheck(lambda a, b: nufft_forward_omega(ops, a, b, 1 / H), (x, w))


def test_adjoint_gradient_wrt_omega_matches_the_exact_ndft_gradcheck():
    _, s, w = _inputs()
    ops = _exact_ops()
    assert torch.autograd.gradcheck(
        lambda a, b: nufft_adjoint_omega(ops, a, b, 1 / H, (H, W)), (s, w)
    )


@pytest.mark.skipif(not TORCHKBNUFFT_AVAILABLE, reason="torchkbnufft not installed")
def test_torchkbnufft_omega_gradient_agrees_with_the_exact_one():
    """The real operator's ω-gradient is within interpolation error of the exact one."""
    x, _, w = _inputs(seed=1)
    x = x.detach().requires_grad_(True)
    op = NUFFTOperator(traj=torch.zeros(2, N), im_size=(H, W))
    op.nufft_obj.to(DT)
    op.adj_nufft_obj.to(DT)

    def omega_grad(ops):
        ww = w.detach().clone().requires_grad_(True)
        (nufft_forward_omega(ops, x, ww, 1 / H).abs() ** 2).sum().backward()
        return ww.grad

    exact = omega_grad(_exact_ops())
    approx = omega_grad(op)
    assert (approx - exact).norm() / exact.norm() < 1e-2
