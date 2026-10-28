"""The orthonormal wavelet Psi of the CS baseline, and the exact complex soft threshold."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.wavelet_ops import (
    WAVELET_FILTERS,
    OrthogonalWavelet2D,
    complex_soft_threshold,
)


@pytest.mark.parametrize("name", sorted(WAVELET_FILTERS))
@pytest.mark.parametrize("levels", [1, 2])
def test_perfect_reconstruction_on_complex_images(name: str, levels: int) -> None:
    torch.manual_seed(0)
    psi = OrthogonalWavelet2D(name, levels)
    x = torch.randn(2, 1, 32, 32, dtype=torch.complex128)
    coeffs = psi.forward(x)
    assert coeffs.shape == x.shape and coeffs.is_complex()
    torch.testing.assert_close(psi.inverse(coeffs), x, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", sorted(WAVELET_FILTERS))
def test_transform_matrix_is_orthonormal(name: str) -> None:
    """Psi Psi^T = I on 16x16, built column by column: Parseval, not only invertibility."""
    psi = OrthogonalWavelet2D(name, 1)
    basis = torch.eye(256, dtype=torch.float64).reshape(256, 16, 16)
    matrix = psi.forward(basis).reshape(256, 256)
    torch.testing.assert_close(matrix @ matrix.T, torch.eye(256, dtype=torch.float64))


@pytest.mark.parametrize("name", sorted(WAVELET_FILTERS))
def test_matches_pywt_periodization(name: str) -> None:
    """The names mean what pywt means by them, coefficient for coefficient."""
    pywt = pytest.importorskip("pywt")
    x = torch.randn(32, 32, dtype=torch.float64)
    ours = OrthogonalWavelet2D(name, 2).forward(x).numpy()
    reference = pywt.wavedec2(x.numpy(), name, mode="periodization", level=2)
    array, _ = pywt.coeffs_to_array(reference)
    assert abs(ours - array).max() < 1e-12


def test_detail_mask_excludes_only_the_coarsest_approximation() -> None:
    mask = OrthogonalWavelet2D("haar", 2).detail_mask(32, 32)
    assert not mask[:8, :8].any() and mask.sum() == 32 * 32 - 64


def test_unknown_wavelet_and_bad_levels_raise() -> None:
    with pytest.raises(ValueError, match="Unknown wavelet 'sym8'"):
        OrthogonalWavelet2D("sym8", 2)
    for levels in (0, -1, 2.0, True):
        with pytest.raises(ValueError, match="positive integer"):
            OrthogonalWavelet2D("haar", levels)  # type: ignore[arg-type]


def test_shape_the_levels_cannot_divide_raises() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        OrthogonalWavelet2D("haar", 3).forward(torch.zeros(1, 1, 20, 32))
    with pytest.raises(ValueError, match="at least 8 samples"):
        OrthogonalWavelet2D("db4", 4).forward(torch.zeros(1, 1, 32, 32))


def test_complex_soft_threshold_is_the_exact_l1_prox() -> None:
    """Magnitude shrinks by tau, phase kept, |z| <= tau is exactly zero."""
    z = torch.tensor([3 + 4j, 0.3 + 0.4j, 0j, -2 + 0j], dtype=torch.complex128)
    out = complex_soft_threshold(z, 1.0)
    expected = torch.tensor([3 * 0.8 + 4j * 0.8, 0, 0, -1 + 0j], dtype=torch.complex128)
    torch.testing.assert_close(out, expected)
    # Optimality: no perturbation lowers 0.5|p - z|^2 + tau |p|.
    torch.manual_seed(1)
    z = torch.randn(64, dtype=torch.complex128)
    p = complex_soft_threshold(z, 0.5)

    def cost(v: torch.Tensor) -> torch.Tensor:
        return 0.5 * (v - z).abs().pow(2).sum() + 0.5 * v.abs().sum()

    for _ in range(20):
        delta = 1e-3 * torch.randn(64, dtype=torch.complex128)
        assert cost(p) <= cost(p + delta) + 1e-12


def test_single_axis_haar_is_the_pairwise_sum_and_difference() -> None:
    psi = OrthogonalWavelet2D("haar", 1)
    x = torch.tensor([[1.0, 3.0, 2.0, 6.0]], dtype=torch.float64)
    c = psi.analyse_axis(x, -1)
    r = 2.0**0.5
    torch.testing.assert_close(c, torch.tensor([[4 / r, 8 / r, -2 / r, -4 / r]], dtype=torch.float64))
    torch.testing.assert_close(psi.synthesise_axis(c, -1), x)
    with pytest.raises(ValueError, match="odd length"):
        psi.analyse_axis(torch.zeros(3), -1)
    with pytest.raises(ValueError, match="odd length"):
        psi.synthesise_axis(torch.zeros(3), -1)
