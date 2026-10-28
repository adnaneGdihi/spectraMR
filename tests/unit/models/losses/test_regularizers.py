"""Tests for k-space regularizers.

Targets ``spectramr.models.losses.regularizers.HermitianSymmetryLoss``.

Regression: for a CENTERED k-space (DC at index N/2) the Hermitian partner
of sample ``i`` is sample ``N − i`` (DC maps to itself). A plain ``flip``
gives ``N − 1 − i`` — half a sample off for even N — so a perfectly real
image would be penalised. The fix rolls by +1 after the flip; this test
asserts the loss is ≈0 on the k-space of a real image and clearly positive
on a non-Hermitian (complex) k-space.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.physics.fft_ops import fft2c  # noqa: E402
from spectramr.models.losses.regularizers import HermitianSymmetryLoss  # noqa: E402


def test_real_image_kspace_is_hermitian_symmetric() -> None:
    """k-space of a real image scores ≈0 (it is Hermitian by construction)."""
    torch.manual_seed(0)
    real_image = torch.randn(1, 1, 16, 16)
    kspace = fft2c(real_image)  # complex, centered

    loss = HermitianSymmetryLoss(normalize=True)(kspace)
    assert loss.item() < 1e-4, f"real-image k-space should be Hermitian, got {loss.item()}"


def test_non_hermitian_kspace_is_penalised() -> None:
    """A k-space with no conjugate symmetry scores clearly above zero."""
    torch.manual_seed(0)
    complex_image = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
    kspace = fft2c(complex_image)

    loss = HermitianSymmetryLoss(normalize=True)(kspace)
    assert loss.item() > 1e-2


def test_hermitian_loss_is_differentiable() -> None:
    """The loss carries gradient back to the input k-space."""
    real_image = torch.randn(1, 1, 8, 8, requires_grad=True)
    kspace = fft2c(real_image)
    loss = HermitianSymmetryLoss(normalize=False)(kspace)
    loss.backward()
    assert real_image.grad is not None


@pytest.mark.parametrize("shape", [(15, 15), (15, 16), (16, 17), (16, 16)])
def test_real_image_kspace_is_hermitian_on_odd_and_even_axes(shape: tuple[int, int]) -> None:
    """An odd axis pairs i with N-1-i (a plain flip); cropping it to even shifted DC."""
    real_image = torch.randn(2, 1, *shape, generator=torch.Generator().manual_seed(0))
    loss = HermitianSymmetryLoss(normalize=True)(fft2c(real_image))
    assert loss.item() < 1e-10


def test_patch_low_rank_accepts_half_precision_amp_outputs() -> None:
    """The SVD has no fp16 kernel; a half input is upcast rather than raising."""
    from spectramr.models.losses.regularizers import PatchLowRankLoss

    x = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(0))
    loss_fn = PatchLowRankLoss(patch_size=4)
    torch.testing.assert_close(loss_fn(x.half()), loss_fn(x.half().float()))


def test_kspace_l1_sparsity_of_zero_kspace_is_zero_with_finite_gradient() -> None:
    """The magnitude floor keeps the gradient finite without biasing the value."""
    from spectramr.models.losses.regularizers import KspaceL1SparsityLoss

    for x in (torch.zeros(2, 2, 8, 8), torch.zeros(2, 1, 8, 8, dtype=torch.complex64)):
        x.requires_grad_(True)
        out = KspaceL1SparsityLoss()(x)
        out.backward()
        assert out.item() < 1e-10
        assert x.grad is not None and torch.isfinite(torch.view_as_real(x.grad) if x.is_complex() else x.grad).all()
