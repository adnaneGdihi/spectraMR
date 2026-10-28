import pytest
import torch

from spectramr.models.losses.pde_losses import HelmholtzPDELoss


def test_helmholtz_pde_loss():
    """Test Helmholtz PDE loss with k_sq=0 (Laplace equation).

    Uses S_real(x,y) = x^2 and S_imag(x,y) = y^2.
    Laplacian(x^2) = d^2(x^2)/dx^2 + d^2(x^2)/dy^2 = 2 + 0 = 2
    Laplacian(y^2) = d^2(y^2)/dx^2 + d^2(y^2)/dy^2 = 0 + 2 = 2
    Helmholtz residual (k_sq=0): mean(|2|^2 + |2|^2) = 8 per coil, mean over coils = 8
    """
    loss_fn = HelmholtzPDELoss(k_sq=0.0)

    batch_size = 5
    num_coils = 4
    coords = torch.rand(batch_size, 2, requires_grad=True)

    # S_real(x,y) = x^2 -> ∇²S_real = 2
    # S_imag(x,y) = y^2 -> ∇²S_imag = 2
    s_real = (coords[:, 0:1] ** 2).expand(-1, num_coils)
    s_imag = (coords[:, 1:2] ** 2).expand(-1, num_coils)

    loss = loss_fn(s_real, s_imag, coords)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad is True
    # Residual = mean(2^2 + 2^2) = mean(4 + 4) = 8
    assert torch.allclose(loss, torch.tensor(8.0), atol=1e-3)


def test_helmholtz_pde_loss_k_sq():
    """Test Helmholtz PDE loss with non-zero k_sq.

    Uses S_real = x^2, S_imag = y^2 (quadratic, so autograd keeps full graph).
    Laplacian(x^2) = 2, Laplacian(y^2) = 2.
    Residual_real = laplacian_real + k_sq * S_real = 2 + k_sq * x^2
    Residual_imag = laplacian_imag + k_sq * S_imag = 2 + k_sq * y^2
    Loss = mean(res_r^2 + res_i^2) = mean((2 + k_sq*x^2)^2 + (2 + k_sq*y^2)^2)
    """
    k_sq = 1.5
    loss_fn = HelmholtzPDELoss(k_sq=k_sq)

    batch_size = 100  # Larger batch for more stable mean
    num_coils = 1
    coords = torch.rand(batch_size, 2, requires_grad=True)

    s_real = coords[:, 0:1] ** 2  # x^2
    s_imag = coords[:, 1:2] ** 2  # y^2

    loss = loss_fn(s_real, s_imag, coords)

    # Expected:
    # res_r = 2 + k_sq * x^2
    # res_i = 2 + k_sq * y^2
    expected_loss = torch.mean(
        (2.0 + k_sq * coords[:, 0] ** 2) ** 2 + (2.0 + k_sq * coords[:, 1] ** 2) ** 2
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad is True
    assert torch.allclose(loss, expected_loss, atol=1e-2)


def _reference_grid_residual(pred: torch.Tensor, k_sq: float) -> torch.Tensor:
    """Per-channel loop the batched grid path replaces."""
    kernel = torch.tensor(
        [[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]],
        dtype=pred.real.dtype if pred.is_complex() else pred.dtype,
    )
    s_real = pred.real if pred.is_complex() else pred
    s_imag = pred.imag if pred.is_complex() else torch.zeros_like(pred)
    total = torch.zeros((), dtype=kernel.dtype)
    for c in range(pred.shape[1]):
        sr, si = s_real[:, c : c + 1], s_imag[:, c : c + 1]
        res_r = torch.nn.functional.conv2d(sr, kernel, padding=1) + k_sq * sr
        res_i = torch.nn.functional.conv2d(si, kernel, padding=1) + k_sq * si
        total = total + torch.mean(res_r**2 + res_i**2)
    return total / pred.shape[1]



@pytest.mark.parametrize("complex_input", [False, True])
def test_grid_path_matches_the_per_channel_loop(complex_input: bool) -> None:
    torch.manual_seed(0)
    if complex_input:
        pred = torch.randn(2, 3, 12, 10, dtype=torch.complex128, requires_grad=True)
    else:
        pred = torch.randn(2, 3, 12, 10, dtype=torch.float64, requires_grad=True)
    loss_fn = HelmholtzPDELoss(k_sq=0.7)
    out = loss_fn(pred, torch.zeros(1))
    ref = _reference_grid_residual(pred, 0.7)
    torch.testing.assert_close(out, ref)
    (grad,) = torch.autograd.grad(out, pred)
    (ref_grad,) = torch.autograd.grad(ref, pred)
    torch.testing.assert_close(grad, ref_grad)


def test_unknown_constructor_kwarg_raises() -> None:
    with pytest.raises(TypeError):
        HelmholtzPDELoss(k_squared=1.0)  # type: ignore[call-arg]


def test_autograd_path_without_coords_raises() -> None:
    with pytest.raises(ValueError, match="requires_grad"):
        HelmholtzPDELoss()(torch.rand(5, 2), torch.rand(5, 2))
