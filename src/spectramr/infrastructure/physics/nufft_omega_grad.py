"""NUFFT forward and adjoint with a gradient in the trajectory ω (cohort5 §2.7).

torchkbnufft returns no gradient with respect to the k-space trajectory. These two
autograd functions wrap the operator's own forward and adjoint modules, keep the data
gradients as the adjoint and forward pair, and supply ∂/∂ω from the NUFFT Jacobian
identity (Wang & Fessler 2021): ∂y_m/∂ω_{m,d} = F(-i r_d x)_m, with the same unscaled
forward F on the image-centred coordinates r = n - N/2 that the operator uses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

Op = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _centred_coords(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Image-centred pixel coordinates ``(2, H, W)``, row first to match ω."""
    rows = torch.arange(h, device=device, dtype=dtype) - h // 2
    cols = torch.arange(w, device=device, dtype=dtype) - w // 2
    return torch.stack([rows[:, None].expand(h, w), cols[None, :].expand(h, w)])


def _omega_grad(
    v: torch.Tensor,
    u: torch.Tensor,
    forward: Callable[[torch.Tensor], torch.Tensor],
    coords: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """``grad_ω[d, m] = Re Σ_batch,coil conj(v_m) · s · F(-i r_d u)_m``."""
    rows = []
    for d in range(coords.shape[0]):
        sampled = scale * forward(-1j * coords[d] * u)
        rows.append((v.conj() * sampled).real.sum(dim=(0, 1)))
    return torch.stack(rows)


class _NUFFTForward(torch.autograd.Function):
    """``y = s · F(x; ω)``; the x-gradient is ``s · F^H(g)``."""

    @staticmethod
    def forward(ctx: Any, image, omega, fwd: Op, adj: Op, coords, scale: float):  # type: ignore[override]
        ctx.save_for_backward(image, omega)
        ctx.fwd, ctx.adj, ctx.coords, ctx.scale = fwd, adj, coords, scale
        return scale * fwd(image, omega)

    @staticmethod
    def backward(ctx: Any, grad):  # type: ignore[override]
        image, omega = ctx.saved_tensors
        grad_image = ctx.scale * ctx.adj(grad, omega) if ctx.needs_input_grad[0] else None
        grad_omega = None
        if ctx.needs_input_grad[1]:
            grad_omega = _omega_grad(
                grad, image, lambda z: ctx.fwd(z, omega), ctx.coords, ctx.scale
            )
        return grad_image, grad_omega, None, None, None, None


class _NUFFTAdjoint(torch.autograd.Function):
    """``x = s · F^H(c; ω)``; the c-gradient is ``s · F(g)``."""

    @staticmethod
    def forward(ctx: Any, samples, omega, fwd: Op, adj: Op, coords, scale: float):  # type: ignore[override]
        ctx.save_for_backward(samples, omega)
        ctx.fwd, ctx.adj, ctx.coords, ctx.scale = fwd, adj, coords, scale
        return scale * adj(samples, omega)

    @staticmethod
    def backward(ctx: Any, grad):  # type: ignore[override]
        samples, omega = ctx.saved_tensors
        grad_samples = ctx.scale * ctx.fwd(grad, omega) if ctx.needs_input_grad[0] else None
        grad_omega = None
        if ctx.needs_input_grad[1]:
            grad_omega = _omega_grad(
                samples, grad, lambda z: ctx.fwd(z, omega), ctx.coords, ctx.scale
            )
        return grad_samples, grad_omega, None, None, None, None


def _operator_ops(operator: Any) -> tuple[Op, Op]:
    return (
        lambda x, w: operator.nufft_obj(x, w),
        lambda y, w: operator.adj_nufft_obj(y, w),
    )


def nufft_forward_omega(
    operator: Any, image: torch.Tensor, omega: torch.Tensor, scale: float
) -> torch.Tensor:
    """Samples ``(B, C, N)`` of ``image (B, C, H, W)`` at ``omega (2, N)``, scaled by ``scale``."""
    fwd, adj = _operator_ops(operator)
    h, w = image.shape[-2:]
    coords = _centred_coords(h, w, image.device, image.real.dtype)
    return _NUFFTForward.apply(image, omega, fwd, adj, coords, scale)


def nufft_adjoint_omega(
    operator: Any,
    samples: torch.Tensor,
    omega: torch.Tensor,
    scale: float,
    im_size: tuple[int, int],
) -> torch.Tensor:
    """Image ``(B, C, H, W)`` from ``samples (B, C, N)`` at ``omega``, scaled by ``scale``."""
    fwd, adj = _operator_ops(operator)
    h, w = im_size
    coords = _centred_coords(h, w, samples.device, samples.real.dtype)
    return _NUFFTAdjoint.apply(samples, omega, fwd, adj, coords, scale)


__all__ = ["nufft_adjoint_omega", "nufft_forward_omega"]
