r"""Identify a scanner's hardware warp from repeat-session commutators.

Gradient-nonlinearity and static-field displacements in the source-free imaging volume are
gradients of harmonic potentials, so the hardware warp is :math:`\psi=\exp(v)` with :math:`v` in
the span of gradients of solid harmonics. Two sessions of one head in poses :math:`g_s, g_{s'}`
register to :math:`\phi=\psi^{-1}h\psi` with :math:`h=g_{s'}g_s^{-1}` rigid, the hand-eye equation
:math:`AX=XB` with a diffeomorphic :math:`X`, and to first order
:math:`\log\phi=\xi+\mathrm{ad}_\xi v` where, in the vector-field bracket,
:math:`\mathrm{ad}_\xi v=[v,\xi]=(v\cdot\nabla)\xi-(\xi\cdot\nabla)v` (the Lie algebra of the
diffeomorphism group carries the negative of the vector-field bracket): the non-rigid residual is a
linear observation of :math:`v` through :math:`\mathrm{ad}_\xi`. On the degree-:math:`l` block two rotation generators
about axes at angle :math:`\theta` satisfy

.. math::

    \|\mathrm{ad}_{\xi_1}v\|^2+\|\mathrm{ad}_{\xi_2}v\|^2\;\ge\;\omega^2\big(1-|P_l(\cos\theta)|\big)\|v\|^2,

Theorem 2.1 of ``TODO/ULF_paired_64mT_3T_breakthrough_candidate.md``, which is what makes the warp
identifiable from head motion alone and what :func:`identifiability_constant` certifies on a
measured pose set. Everything is torch float64; the ball inner product is a sum over a point cloud.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

Generator = tuple[torch.Tensor, torch.Tensor]


def monomial_exponents(degree: int) -> torch.Tensor:
    """Return the ``[n, 3]`` integer exponents of the monomials of total degree ``degree``."""
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    rows = [
        (a, b, degree - a - b) for a in range(degree, -1, -1) for b in range(degree - a, -1, -1)
    ]
    return torch.tensor(rows, dtype=torch.long)


def _monomial_values(exps: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Evaluate monomials ``[n_mono]`` at ``points [n_pts, 3]`` -> ``[n_mono, n_pts]``."""
    return torch.prod(points.unsqueeze(0) ** exps.unsqueeze(1).to(points.dtype), dim=-1)


def _derivative(exps: torch.Tensor, axis: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Exponents and multipliers of the partial derivative along ``axis`` of each monomial."""
    mult = exps[:, axis].clone()
    new = exps.clone()
    new[:, axis] = torch.clamp(new[:, axis] - 1, min=0)
    return new, mult


def _laplacian_matrix(degree: int) -> torch.Tensor:
    """Matrix of the Laplacian from degree-``degree`` to degree-``degree-2`` monomials."""
    src, dst = monomial_exponents(degree), monomial_exponents(degree - 2)
    lap = torch.zeros(dst.shape[0], src.shape[0], dtype=torch.float64)
    for j, e in enumerate(src):
        for axis in range(3):
            if e[axis] >= 2:
                target = e.clone()
                target[axis] -= 2
                i = int(torch.nonzero((dst == target).all(dim=1))[0])
                lap[i, j] += float(e[axis] * (e[axis] - 1))
    return lap


def harmonic_coefficients(degree: int) -> torch.Tensor:
    """Coefficients ``[2l+1, n_mono]`` of a basis of harmonic homogeneous polynomials of degree ``l``."""
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    n_mono = monomial_exponents(degree).shape[0]
    if degree == 1:
        return torch.eye(n_mono, dtype=torch.float64)
    lap = _laplacian_matrix(degree)
    _, s, vh = torch.linalg.svd(lap, full_matrices=True)
    rank = int((s > 1e-10 * s.max()).sum())
    kernel = vh[rank:]
    if kernel.shape[0] != 2 * degree + 1:
        raise RuntimeError(f"harmonic space of degree {degree} has dimension {kernel.shape[0]}")
    return kernel


def evaluate_polynomial_gradients(
    coeffs: torch.Tensor, exps: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Gradients ``[n_basis, n_pts, 3]`` of the polynomials ``coeffs @ monomials`` at ``points``."""
    cols = []
    for axis in range(3):
        d_exps, mult = _derivative(exps, axis)
        vals = _monomial_values(d_exps, points) * mult.to(points.dtype).unsqueeze(1)
        cols.append(coeffs.to(points.dtype) @ vals)
    return torch.stack(cols, dim=-1)


def evaluate_polynomial_hessians(
    coeffs: torch.Tensor, exps: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Hessians ``[n_basis, n_pts, 3, 3]`` of the polynomials ``coeffs @ monomials`` at ``points``."""
    rows = []
    for i in range(3):
        d_exps, mult_i = _derivative(exps, i)
        cols = []
        for j in range(3):
            dd_exps, mult_j = _derivative(d_exps, j)
            vals = _monomial_values(dd_exps, points) * (mult_i * mult_j).to(points.dtype).unsqueeze(
                1
            )
            cols.append(coeffs.to(points.dtype) @ vals)
        rows.append(torch.stack(cols, dim=-1))
    return torch.stack(rows, dim=-2)


def ball_points(n_per_axis: int, radius: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Cubic-grid points strictly inside the ball of the given radius, ``[n, 3]``."""
    if n_per_axis < 2 or radius <= 0:
        raise ValueError("n_per_axis must be >= 2 and radius > 0")
    axis = torch.linspace(-radius, radius, n_per_axis, dtype=dtype)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    return grid[(grid**2).sum(dim=1) < radius**2]


def legendre_p(degree: int, x: torch.Tensor | float) -> torch.Tensor:
    """Legendre polynomial :math:`P_l(x)` by the three-term recurrence."""
    x = torch.as_tensor(x, dtype=torch.float64)
    p_prev, p = torch.ones_like(x), x.clone()
    if degree == 0:
        return p_prev
    for n in range(1, degree):
        p_prev, p = p, ((2 * n + 1) * x * p - n * p_prev) / (n + 1)
    return p


def legendre_bound(degree: int, omega: float, theta: float) -> float:
    """The constant :math:`\\omega^2(1-|P_l(\\cos\\theta)|)` of Theorem 2.1 for one degree block."""
    return float(omega**2 * (1.0 - abs(float(legendre_p(degree, math.cos(theta))))))


def rigid_generator(
    axis: Sequence[float], angle: float, translation: Sequence[float] | None = None
) -> Generator:
    """Return ``(omega, t)`` for a rotation of ``angle`` about ``axis`` plus a translation."""
    u = torch.as_tensor(axis, dtype=torch.float64)
    if u.shape != (3,) or float(u.norm()) == 0.0:
        raise ValueError("axis must be a non-zero 3-vector")
    t = (
        torch.zeros(3, dtype=torch.float64)
        if translation is None
        else torch.as_tensor(translation, dtype=torch.float64)
    )
    return angle * u / u.norm(), t


class HarmonicGradientBasis:
    """Orthonormal basis of harmonic gradient fields on a point cloud, one block per degree.

    Blocks ``min_degree..max_degree`` are orthonormalized separately in the inner product
    ``sum_points v . w``; degree 1 (constant fields) is the gauge of the scanner frame and is
    excluded by default. ``coeffs`` holds the polynomial coefficients of the orthonormalized basis
    so that fields can be evaluated at any point.
    """

    def __init__(
        self,
        points: torch.Tensor,
        max_degree: int,
        min_degree: int = 2,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be [n, 3], got {tuple(points.shape)}")
        if min_degree < 1 or max_degree < min_degree:
            raise ValueError(f"need 1 <= min_degree <= max_degree, got {min_degree}, {max_degree}")
        self.points = points.to(dtype)
        self.min_degree, self.max_degree = min_degree, max_degree
        degrees, fields, hessians, self.blocks = [], [], [], {}
        for degree in range(min_degree, max_degree + 1):
            exps, coeffs = monomial_exponents(degree), harmonic_coefficients(degree)
            raw = evaluate_polynomial_gradients(coeffs, exps, self.points)
            q, r = torch.linalg.qr(raw.reshape(raw.shape[0], -1).T)
            r_inv = torch.linalg.inv(r)
            self.blocks[degree] = (exps, (r_inv.T @ coeffs))
            fields.append(q.T.reshape(raw.shape))
            hess = evaluate_polynomial_hessians(coeffs, exps, self.points)
            hessians.append(torch.einsum("ji,jnab->inab", r_inv, hess))
            degrees.extend([degree] * raw.shape[0])
        self.degrees = torch.tensor(degrees, dtype=torch.long)
        self.fields = torch.cat(fields, dim=0)
        self.hessians = torch.cat(hessians, dim=0)

    @property
    def n_basis(self) -> int:
        return int(self.fields.shape[0])

    def block_index(self, degree: int) -> torch.Tensor:
        return torch.nonzero(self.degrees == degree).flatten()

    def adjoint_action(
        self, omega: torch.Tensor, translation: torch.Tensor, index: Any = None
    ) -> torch.Tensor:
        """``ad_xi v = [v, xi]``, ``omega x grad Phi - H_Phi (omega x r + t)`` for ``v = grad Phi``,
        applied to every basis field on the points ``index`` selects (default: all)."""
        points, fields, hessians = self.points, self.fields, self.hessians
        if index is not None:
            points, fields, hessians = points[index], fields[:, index], hessians[:, index]
        xi = torch.cross(omega.expand_as(points), points, dim=1) + translation
        transported = torch.einsum("bnij,nj->bni", hessians, xi)
        return torch.cross(omega.expand_as(fields), fields, dim=-1) - transported

    def synthesize(self, coeffs: torch.Tensor, points: torch.Tensor | None = None) -> torch.Tensor:
        """Displacement field ``sum_j coeffs_j v_j`` on ``points`` (default: the basis points)."""
        if coeffs.shape != (self.n_basis,):
            raise ValueError(f"coeffs must have shape ({self.n_basis},), got {tuple(coeffs.shape)}")
        if points is None:
            return torch.einsum("b,bni->ni", coeffs.to(self.fields.dtype), self.fields)
        out = torch.zeros(points.shape[0], 3, dtype=self.points.dtype)
        for degree, (exps, poly) in self.blocks.items():
            grads = evaluate_polynomial_gradients(poly, exps, points.to(self.points.dtype))
            out = out + torch.einsum(
                "b,bni->ni", coeffs[self.block_index(degree)].to(out.dtype), grads
            )
        return out


def stacked_design(
    basis: HarmonicGradientBasis,
    generators: Sequence[Generator],
    slices: Sequence[Any] | None = None,
) -> torch.Tensor:
    """Design ``[sum_k n_k * 3, n_basis]``; pair ``k`` is observed on ``slices[k]`` (default: all)."""
    if len(generators) == 0:
        raise ValueError("at least one generator is required")
    slices = [None] * len(generators) if slices is None else list(slices)
    if len(slices) != len(generators):
        raise ValueError(f"{len(generators)} generators but {len(slices)} slices")
    cols = [
        basis.adjoint_action(omega, t, index).reshape(basis.n_basis, -1)
        for (omega, t), index in zip(generators, slices, strict=True)
    ]
    return torch.cat(cols, dim=1).T


def identifiability_constant(
    basis: HarmonicGradientBasis, generators: Sequence[Generator]
) -> dict[int, float]:
    """Per-degree smallest squared singular value of the stacked design: the certified constant.

    Because the basis is orthonormal on the point cloud, this is
    ``min_v sum_k ||ad_xi_k v||^2 / ||v||^2`` over the degree block, the left-hand side of
    Theorem 2.1 divided by ``||v||^2``; the theorem lower-bounds it by :func:`legendre_bound`.
    """
    design = stacked_design(basis, generators)
    return {
        degree: float(torch.linalg.svdvals(design[:, basis.block_index(degree)])[-1] ** 2)
        for degree in range(basis.min_degree, basis.max_degree + 1)
    }


def estimate_warp(
    basis: HarmonicGradientBasis,
    generators: Sequence[Generator],
    residuals: torch.Tensor,
    noise_std: float | None = None,
    slices: Sequence[Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Least-squares warp coefficients from residuals ``[K, n_pts, 3]`` on a shared cloud, or
    ``[n_pts, 3]`` over the whole cloud with ``slices[k]`` naming pair ``k``'s part; with
    ``noise_std`` also the Gauss-Markov covariance ``noise_std^2 (A^T A)^{-1}`` (Proposition 2.3)."""
    n = basis.points.shape[0]
    expected = (len(generators), n, 3) if slices is None else (n, 3)
    if tuple(residuals.shape) != expected:
        raise ValueError(f"residuals must have shape {expected}, got {tuple(residuals.shape)}")
    design = stacked_design(basis, generators, slices)
    coeffs = torch.linalg.lstsq(
        design, residuals.reshape(-1, 1).to(design.dtype)
    ).solution.flatten()
    if noise_std is None:
        return coeffs, None
    cov = noise_std**2 * torch.linalg.inv(design.T @ design)
    return coeffs, cov


__all__ = [
    "HarmonicGradientBasis",
    "ball_points",
    "estimate_warp",
    "evaluate_polynomial_gradients",
    "evaluate_polynomial_hessians",
    "harmonic_coefficients",
    "identifiability_constant",
    "legendre_bound",
    "legendre_p",
    "monomial_exponents",
    "rigid_generator",
    "stacked_design",
]
