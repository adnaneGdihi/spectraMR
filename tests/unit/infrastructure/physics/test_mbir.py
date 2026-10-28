"""Unit tests for the Model-Based Image Reconstruction (MBIR) and markers."""

import math

import torch

from spectramr.infrastructure.physics.mbir import positive_contrast_demodulation


def test_positive_contrast_demodulation_scalar_te() -> None:
    """Verify demodulation works with a scalar float TE."""
    kspace = torch.ones(2, 4, 32, 32, dtype=torch.complex64)
    delta_omega_hz = 150.0
    te_sec = 0.010  # 10 ms

    result = positive_contrast_demodulation(kspace, delta_omega_hz, te_sec)

    expected_phase_shift = 2.0 * math.pi * delta_omega_hz * te_sec
    expected_factor = torch.exp(torch.tensor(1j * expected_phase_shift))

    assert torch.allclose(result, kspace * expected_factor)
    assert result.shape == kspace.shape


def test_positive_contrast_demodulation_tensor_te() -> None:
    """Verify demodulation works with a 1D tensor of readout times."""
    kspace = torch.ones(2, 4, 100, dtype=torch.complex64)  # e.g. [B, C, K]
    delta_omega_hz = 150.0
    te_sec = torch.linspace(0.001, 0.010, 100)

    result = positive_contrast_demodulation(kspace, delta_omega_hz, te_sec)

    phase_shift_first = 2.0 * math.pi * delta_omega_hz * 0.001
    factor_first = torch.exp(torch.tensor(1j * phase_shift_first))
    assert torch.allclose(result[0, 0, 0], factor_first)

    phase_shift_last = 2.0 * math.pi * delta_omega_hz * 0.010
    factor_last = torch.exp(torch.tensor(1j * phase_shift_last))
    assert torch.allclose(result[0, 0, -1], factor_last)
    assert result.shape == kspace.shape


# --- FISTA as the CS solver: exact wavelet prox, 1/L step, the A^H y form ---------

import pytest  # noqa: E402

from spectramr.infrastructure.physics.mbir import FISTAMBIRSolver  # noqa: E402
from spectramr.infrastructure.physics.multicoil_ei_operator import (  # noqa: E402
    MulticoilEIOperator,
)
from spectramr.infrastructure.physics.wavelet_ops import OrthogonalWavelet2D  # noqa: E402


def _operator(mask: torch.Tensor | None = None, seed: int = 0) -> MulticoilEIOperator:
    """Four UN-normalised coils: A^H A = diag(sum_c |S_c|^2) is not the identity."""
    gen = torch.Generator().manual_seed(seed)
    mag = 0.5 + torch.rand(1, 4, 32, 32, generator=gen, dtype=torch.float64)
    phase = 6.28 * torch.rand(1, 4, 32, 32, generator=gen, dtype=torch.float64)
    return MulticoilEIOperator(torch.polar(mag, phase), mask, where="in the test")


def _solver(op: MulticoilEIOperator, **kw) -> FISTAMBIRSolver:
    kw.setdefault("regularization", "wavelet")
    if kw["regularization"] == "wavelet":
        kw.setdefault("wavelet", OrthogonalWavelet2D("haar", 2))
    return FISTAMBIRSolver(op, step_size=None, lipschitz=op.lipschitz_bound(), **kw)


def test_unknown_regularization_and_orphan_wavelet_raise() -> None:
    op = _operator()
    with pytest.raises(ValueError, match="Unknown regularization 'l1_wavelet'"):
        FISTAMBIRSolver(op, regularization="l1_wavelet")
    with pytest.raises(ValueError, match="requires a wavelet transform"):
        FISTAMBIRSolver(op, regularization="wavelet")
    with pytest.raises(ValueError, match="requires a wavelet transform"):
        FISTAMBIRSolver(op, regularization="identity", wavelet=OrthogonalWavelet2D())
    with pytest.raises(ValueError, match="positive bound"):
        FISTAMBIRSolver(op, regularization="identity", lipschitz=0.0)


def test_solver_needs_exactly_one_of_y_and_its_adjoint() -> None:
    solver = _solver(_operator(), num_iterations=1)
    y = torch.zeros(1, 4, 32, 32, dtype=torch.complex128)
    with pytest.raises(ValueError, match="exactly one"):
        solver(y, adjoint_measurement=solver.A.adjoint(y))
    with pytest.raises(ValueError, match="exactly one"):
        solver()


def test_lambda_zero_full_sampling_converges_to_the_sense_inverse() -> None:
    """M = I, lambda = 0: the minimiser is b / sum_c |S_c|^2, reached from y or from b."""
    op = _operator()
    x_true = torch.randn(1, 1, 32, 32, dtype=torch.complex128)
    y = op.forward(x_true)
    solver = _solver(op, num_iterations=300, lambda_reg=0.0)
    expected = op.adjoint(y) / op.maps.abs().pow(2).sum(dim=1, keepdim=True)
    # The SENSE path computes in complex64 (fft_ops), so agreement is to float32.
    torch.testing.assert_close(expected, x_true, rtol=0, atol=1e-5)
    from_y = solver(y)
    from_b = solver(adjoint_measurement=op.adjoint(y))
    torch.testing.assert_close(from_y, expected, rtol=0, atol=1e-5)
    torch.testing.assert_close(from_b, expected, rtol=0, atol=1e-5)
    # Not one step: the first iterate (b / L) is far from the inverse.
    first = _solver(op, num_iterations=1, lambda_reg=0.0)(y)
    assert (first - expected).abs().max() > 1e-2


def test_lipschitz_bound_dominates_the_power_iteration() -> None:
    op = _operator(mask=(torch.rand(1, 1, 1, 32) > 0.5).double())
    v = torch.randn(1, 1, 32, 32, dtype=torch.complex128)
    for _ in range(100):
        v = op.adjoint(op.forward(v))
        estimate = v.norm()
        v = v / estimate
    assert estimate <= op.lipschitz_bound() * (1 + 1e-9)


def test_wavelet_objective_decreases() -> None:
    """FISTA is not monotone, but the objective falls across the run and settles."""
    torch.manual_seed(3)
    mask = (torch.rand(1, 1, 1, 32) < 0.3).double()
    op = _operator(mask)
    y = op.forward(torch.randn(1, 1, 32, 32, dtype=torch.complex128))
    values = [
        _solver(op, num_iterations=n, lambda_reg=0.05).objective(
            _solver(op, num_iterations=n, lambda_reg=0.05)(y), y
        )
        for n in (0, 5, 40, 200)
    ]
    assert values[1] < values[0] and values[2] < values[1]
    assert values[3] <= values[2] * (1 + 1e-6)


def test_wavelet_prox_is_exact() -> None:
    """prox(v) minimises 0.5||p - v||^2 + tau ||D Psi p||_1: no perturbation does better."""
    torch.manual_seed(4)
    solver = _solver(_operator(), lambda_reg=1.0)
    v = torch.randn(1, 1, 32, 32, dtype=torch.complex128)
    tau = 0.7
    p = solver._apply_wavelet(v, tau)

    def cost(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x - v).abs().pow(2).sum() + tau * solver.regularizer(x)

    for _ in range(20):
        delta = 1e-4 * torch.randn_like(v)
        assert cost(p) <= cost(p + delta) + 1e-12
