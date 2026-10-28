"""Unit tests for :mod:`spectramr.infrastructure.physics.hardware_warp_commutator`.

The module identifies a scanner's hardware warp from the commutator of rigid head motions with
the warp. The tests are the planted-violation witnesses of Theorem 2.1 of
``TODO/ULF_paired_64mT_3T_breakthrough_candidate.md`` and of the estimator built on it:

* every degree block is harmonic (Hessian trace zero) and has dimension ``2l + 1``;
* ``adjoint_action`` matches a finite-difference Lie bracket;
* two rotation axes at angle ``theta`` give a per-degree constant at least
  ``omega^2 (1 - |P_l(cos theta)|)`` up to grid tolerance, one axis leaves a kernel, and
  parallel axes are degenerate;
* a planted warp is recovered from noisy commutator observations;
* the exact conjugation ``psi^{-1} h psi`` agrees with the first-order model to the stated bound;
* construction-time guards raise on malformed inputs.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.hardware_warp_commutator import (
    HarmonicGradientBasis,
    ball_points,
    estimate_warp,
    evaluate_polynomial_hessians,
    harmonic_coefficients,
    identifiability_constant,
    legendre_bound,
    legendre_p,
    monomial_exponents,
    rigid_generator,
)

pytestmark = [pytest.mark.unit, pytest.mark.physics]

RADIUS = 100.0


@pytest.fixture(scope="module")
def basis() -> HarmonicGradientBasis:
    return HarmonicGradientBasis(ball_points(15, RADIUS), max_degree=3, min_degree=1)


def _rotation_matrix(omega: torch.Tensor) -> torch.Tensor:
    angle = float(omega.norm())
    k = omega / angle
    kx = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=omega.dtype)
    return torch.eye(3, dtype=omega.dtype) + math.sin(angle) * kx + (1 - math.cos(angle)) * kx @ kx


class TestHarmonicBlocks:
    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_dimension_and_harmonicity(self, degree: int) -> None:
        coeffs = harmonic_coefficients(degree)
        assert coeffs.shape[0] == 2 * degree + 1
        pts = torch.randn(64, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        hess = evaluate_polynomial_hessians(coeffs, monomial_exponents(degree), pts)
        laplacian = hess.diagonal(dim1=-2, dim2=-1).sum(-1)
        assert torch.allclose(laplacian, torch.zeros_like(laplacian), atol=1e-9)

    def test_blocks_orthonormal_on_point_cloud(self, basis: HarmonicGradientBasis) -> None:
        flat = basis.fields.reshape(basis.n_basis, -1)
        gram = flat @ flat.T
        assert torch.allclose(gram, torch.eye(basis.n_basis, dtype=gram.dtype), atol=1e-8)
        assert basis.n_basis == sum(2 * deg + 1 for deg in range(1, 4))


class TestAdjointAction:
    def test_matches_finite_difference_bracket(self, basis: HarmonicGradientBasis) -> None:
        gen = torch.Generator().manual_seed(1)
        coeffs = torch.randn(basis.n_basis, dtype=torch.float64, generator=gen)
        omega, t = rigid_generator([0.3, -0.5, 0.8], 0.05, translation=[1.0, -2.0, 0.5])
        analytic = torch.einsum("b,bni->ni", coeffs, basis.adjoint_action(omega, t))
        xi = torch.cross(omega.expand_as(basis.points), basis.points, dim=1) + t
        eps = 1e-3
        forward = basis.synthesize(coeffs, basis.points + eps * xi)
        backward = basis.synthesize(coeffs, basis.points - eps * xi)
        directional = (forward - backward) / (2 * eps)
        numeric = (
            torch.cross(omega.expand_as(directional), basis.synthesize(coeffs), dim=1) - directional
        )
        assert torch.allclose(analytic, numeric, atol=1e-6, rtol=1e-5)


class TestIdentifiabilityConstant:
    @pytest.mark.parametrize("theta_deg", [30.0, 60.0, 90.0])
    def test_two_axes_meet_legendre_bound(
        self, basis: HarmonicGradientBasis, theta_deg: float
    ) -> None:
        omega = 0.05
        theta = math.radians(theta_deg)
        gens = [
            rigid_generator([0.0, 0.0, 1.0], omega),
            rigid_generator([math.sin(theta), 0.0, math.cos(theta)], omega),
        ]
        constants = identifiability_constant(basis, gens)
        for degree, value in constants.items():
            bound = legendre_bound(degree, omega, theta)
            assert value >= 0.85 * bound, (degree, value, bound)
            assert value > 0.0

    def test_single_axis_has_axisymmetric_kernel(self, basis: HarmonicGradientBasis) -> None:
        constants = identifiability_constant(basis, [rigid_generator([0.0, 0.0, 1.0], 0.05)])
        for value in constants.values():
            assert value < 1e-12

    def test_three_orthogonal_translations_meet_the_poincare_constant(
        self, basis: HarmonicGradientBasis
    ) -> None:
        """Corollary 2.5: pure translations of length tau along three orthogonal axes give the
        constant tau^2 (l-1)(2l+1) / R^2 on every degree l >= 2 -- an equality on the ball, so the
        discrete constant sits within the grid's quadrature error of it."""
        tau = 0.5
        gens = [
            (torch.zeros(3, dtype=torch.float64), tau * torch.eye(3, dtype=torch.float64)[k])
            for k in range(3)
        ]
        constants = identifiability_constant(basis, gens)
        for degree, value in constants.items():
            exact = tau**2 * (degree - 1) * (2 * degree + 1) / RADIUS**2
            assert value == pytest.approx(exact, rel=0.1), (degree, value, exact)

    def test_single_translation_leaves_a_kernel_in_every_degree(
        self, basis: HarmonicGradientBasis
    ) -> None:
        gens = [
            (
                torch.zeros(3, dtype=torch.float64),
                torch.tensor([0.0, 0.0, 5.0], dtype=torch.float64),
            )
        ]
        assert all(v < 1e-12 for v in identifiability_constant(basis, gens).values())

    def test_parallel_axes_are_degenerate(self, basis: HarmonicGradientBasis) -> None:
        gens = [rigid_generator([0.0, 0.0, 1.0], 0.05), rigid_generator([0.0, 0.0, -1.0], 0.03)]
        assert all(v < 1e-12 for v in identifiability_constant(basis, gens).values())
        assert legendre_bound(2, 0.05, 0.0) == pytest.approx(0.0)

    def test_legendre_values(self) -> None:
        assert float(legendre_p(2, 0.0)) == pytest.approx(-0.5)
        assert float(legendre_p(3, 0.0)) == pytest.approx(0.0, abs=1e-12)
        assert float(legendre_p(4, 0.0)) == pytest.approx(0.375)


class TestEstimator:
    def test_planted_warp_recovered_from_noisy_commutators(self) -> None:
        basis = HarmonicGradientBasis(ball_points(13, RADIUS), max_degree=3, min_degree=2)
        gen = torch.Generator().manual_seed(2)
        truth = torch.randn(basis.n_basis, dtype=torch.float64, generator=gen)
        axes = torch.randn(6, 3, dtype=torch.float64, generator=gen)
        gens = [rigid_generator(a.tolist(), 0.05, translation=[0.5, -0.3, 0.2]) for a in axes]
        clean = torch.stack(
            [torch.einsum("b,bni->ni", truth, basis.adjoint_action(o, t)) for o, t in gens]
        )
        noise_std = 1e-3 * float(clean.abs().max())
        noisy = clean + noise_std * torch.randn(clean.shape, dtype=torch.float64, generator=gen)
        coeffs, cov = estimate_warp(basis, gens, noisy, noise_std=noise_std)
        assert torch.allclose(coeffs, truth, rtol=0.0, atol=5e-2 * float(truth.abs().max()))
        assert cov is not None and cov.shape == (basis.n_basis, basis.n_basis)
        assert torch.all(torch.diagonal(cov) > 0)

    def test_sliced_design_agrees_with_the_shared_cloud_fit(self) -> None:
        basis = HarmonicGradientBasis(ball_points(9, RADIUS), max_degree=3, min_degree=2)
        gen = torch.Generator().manual_seed(5)
        truth = torch.randn(basis.n_basis, dtype=torch.float64, generator=gen)
        gens = [
            rigid_generator([0.0, 0.0, 1.0], 0.05, translation=[0.5, 0.0, 0.0]),
            rigid_generator([1.0, 0.0, 0.0], 0.04, translation=[0.0, -0.3, 0.2]),
        ]
        n = basis.points.shape[0]
        index = torch.arange(0, n, 2)
        assert torch.allclose(
            basis.adjoint_action(*gens[0], index), basis.adjoint_action(*gens[0])[:, index]
        )
        shared = torch.stack(
            [torch.einsum("b,bni->ni", truth, basis.adjoint_action(o, t)) for o, t in gens]
        )
        coeffs_shared, _ = estimate_warp(basis, gens, shared)
        union = HarmonicGradientBasis(torch.cat([basis.points, basis.points]), 3, 2)
        slices = [slice(0, n), slice(n, 2 * n)]
        stacked = shared.reshape(-1, 3)
        coeffs_union, _ = estimate_warp(union, gens, stacked, slices=slices)
        probe = ball_points(5, 0.7 * RADIUS)
        assert torch.allclose(
            union.synthesize(coeffs_union, probe), basis.synthesize(coeffs_shared, probe), atol=1e-8
        )
        with pytest.raises(ValueError, match="slices"):
            estimate_warp(union, gens, stacked, slices=[slice(0, n)])

    def test_exact_conjugation_within_first_order_bound(self) -> None:
        basis = HarmonicGradientBasis(ball_points(11, RADIUS), max_degree=3, min_degree=2)
        gen = torch.Generator().manual_seed(3)
        coeffs = torch.randn(basis.n_basis, dtype=torch.float64, generator=gen)
        field = basis.synthesize(coeffs)
        coeffs = coeffs * (0.02 * RADIUS / float(field.norm(dim=1).max()))  # |v|_inf = 2 % of R
        omega, t = rigid_generator([0.2, 0.9, 0.4], 0.03, translation=[0.5, 0.0, -0.5])
        rot = _rotation_matrix(omega)
        pts = basis.points
        psi = pts + basis.synthesize(coeffs, pts)
        moved = psi @ rot.T + t
        inverse = moved.clone()
        for _ in range(8):  # fixed-point inversion of psi at the moved points
            inverse = moved - basis.synthesize(coeffs, inverse)
        exact = inverse - (pts @ rot.T + t)
        first_order = torch.einsum("b,bni->ni", coeffs, basis.adjoint_action(omega, t))
        rel = float((exact - first_order).norm() / first_order.norm())
        assert rel < 0.15, rel


class TestGuards:
    def test_bad_degrees_raise(self) -> None:
        pts = ball_points(5, 1.0)
        with pytest.raises(ValueError):
            HarmonicGradientBasis(pts, max_degree=1, min_degree=2)
        with pytest.raises(ValueError):
            HarmonicGradientBasis(pts, max_degree=2, min_degree=0)

    def test_bad_points_and_residuals_raise(self) -> None:
        with pytest.raises(ValueError):
            HarmonicGradientBasis(torch.zeros(4, 2), max_degree=2)
        basis = HarmonicGradientBasis(ball_points(5, 1.0), max_degree=2)
        gens = [rigid_generator([0.0, 0.0, 1.0], 0.1)]
        with pytest.raises(ValueError):
            estimate_warp(basis, gens, torch.zeros(2, basis.points.shape[0], 3))
        with pytest.raises(ValueError):
            identifiability_constant(basis, [])
        with pytest.raises(ValueError):
            rigid_generator([0.0, 0.0, 0.0], 0.1)
