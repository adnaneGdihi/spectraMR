"""Unit tests for the m4_operator.py."""

import pytest
import torch

try:
    from spectramr.infrastructure.physics.m4_operator import Operator_A_m4

    TORCHKBNUFFT_AVAILABLE = True
except ImportError:
    TORCHKBNUFFT_AVAILABLE = False


@pytest.mark.skipif(not TORCHKBNUFFT_AVAILABLE, reason="TorchKbNufft required")
def test_m4_operator_adjoint_property():
    """Verify that <A x, y> == <x, A^H y>."""
    torch.manual_seed(42)
    B, C, H, W = 1, 1, 32, 32
    Nc = 4
    N_k = 500

    operator = Operator_A_m4(im_size=(H, W), readout_time=0.005, norm_nufft=True)

    # Random tissue image
    x_tissue = torch.randn(B, C, H, W, dtype=torch.complex64)

    # Random measurements
    y_meas = torch.randn(B, Nc, N_k, dtype=torch.complex64)

    # Fake maps dict
    alpha = torch.rand(B, 1, H, W) * 1.5
    theta = torch.tensor([[[1.0, 0.0, 0.1], [0.0, 1.0, -0.1]]]).expand(B, 2, 3)
    b0 = torch.randn(B, 1, H, W) * 50.0  # Hz
    s_map = torch.randn(B, Nc, H, W, dtype=torch.complex64)
    # create positive definite covariance
    psi_tmp = torch.randn(B, Nc, Nc, dtype=torch.complex64)
    psi = psi_tmp @ psi_tmp.mH + torch.eye(Nc).unsqueeze(0) * 0.1
    dk = torch.randn(B, 2, N_k) * 0.5 + torch.stack(
        [
            torch.linspace(-0.5, 0.5, N_k).unsqueeze(0).expand(B, N_k),
            torch.linspace(-0.5, 0.5, N_k).unsqueeze(0).expand(B, N_k),
        ],
        dim=1,
    )  # somewhat realistic coords roughly in [-0.5, 0.5]

    maps_dict = {
        "alpha": alpha,
        "T_kin": theta,
        "B0": b0,
        "S_map": s_map,
        "Psi": psi,
        "dK": dk,
    }

    # Forward
    Ax = operator.forward(x_tissue, maps_dict)  # [B, Nc, N_k]

    # Adjoint
    AH_y = operator.adjoint(y_meas, maps_dict)  # [B, 1, H, W]

    # Inner products
    # <Ax, y> = sum(Ax * conj(y))
    in_prod_1 = (Ax * y_meas.conj()).sum()

    # <x, AH_y> = sum(x * conj(AH_y))
    in_prod_2 = (x_tissue * AH_y.conj()).sum()

    # The adjoint of affine grid_sample is only approximate strictly speaking,
    # but the rest of the operators are exact.
    # To test pure exactness, we can test without T_kin first.
    print(f"With motion: <Ax,y> = {in_prod_1}, <x, AHy> = {in_prod_2}")

    maps_dict_no_motion = maps_dict.copy()
    maps_dict_no_motion["T_kin"] = None

    Ax_nomot = operator.forward(x_tissue, maps_dict_no_motion)
    AH_y_nomot = operator.adjoint(y_meas, maps_dict_no_motion)

    in_prod_1_nomot = (Ax_nomot * y_meas.conj()).sum()
    in_prod_2_nomot = (x_tissue * AH_y_nomot.conj()).sum()

    print(f"Without motion: <Ax,y> = {in_prod_1_nomot}, <x, AHy> = {in_prod_2_nomot}")

    # Assert they are close without motion (exact adjoint)
    torch.testing.assert_close(in_prod_1_nomot, in_prod_2_nomot, rtol=1e-3, atol=1e-3)


def _cartesian_maps(mask: torch.Tensor | None, hw: tuple[int, int]) -> dict:
    return {"dK": None, "mask": mask, "im_size": hw}


def _line_mask(w: int) -> torch.Tensor:
    mask = torch.zeros(1, 1, 1, w)
    mask[..., ::3] = 1.0
    mask[..., w // 2 - 2 : w // 2 + 2] = 1.0
    return mask


def test_cartesian_forward_is_masked_fft2c() -> None:
    """#2380: with dK=None the operator is mask * fft2c, flattened row-major."""
    from spectramr.infrastructure.physics.fft_ops import fft2c

    torch.manual_seed(0)
    H, W = 12, 10
    op = Operator_A_m4(im_size=(32, 32))
    x = torch.randn(2, 1, H, W, dtype=torch.complex64)
    mask = _line_mask(W)
    ax = op.forward(x, _cartesian_maps(mask, (H, W)))
    assert torch.allclose(ax, (fft2c(x) * mask).reshape(2, 1, H * W), atol=1e-6)


def test_cartesian_adjoint_and_dc_fixed_point() -> None:
    """<Ax, y> = <x, A^H y>, and the DC gradient vanishes at the true image."""
    from spectramr.infrastructure.physics.fft_ops import fft2c

    torch.manual_seed(1)
    H, W = 12, 10
    op = Operator_A_m4()
    maps = _cartesian_maps(_line_mask(W), (H, W))
    x = torch.randn(1, 1, H, W, dtype=torch.complex64)
    y = torch.randn(1, 1, H * W, dtype=torch.complex64)
    lhs = (op.forward(x, maps) * y.conj()).sum()
    rhs = (x * op.adjoint(y, maps).conj()).sum()
    assert torch.allclose(lhs, rhs, atol=1e-4)

    y_true = (fft2c(x) * maps["mask"]).reshape(1, 1, H * W)
    grad = op.adjoint(op.forward(x, maps) - y_true, maps)
    assert grad.abs().max() < 1e-5
    full = _cartesian_maps(None, (H, W))
    assert torch.allclose(op.adjoint(fft2c(x).reshape(1, 1, H * W), full), x, atol=1e-5)


def test_cartesian_grid_mismatch_and_missing_dk_raise() -> None:
    op = Operator_A_m4(im_size=(8, 8))
    with pytest.raises(ValueError, match="does not match the grid"):
        op.adjoint(torch.zeros(1, 1, 60, dtype=torch.complex64), {"dK": None})
    with pytest.raises(KeyError):
        op.forward(torch.zeros(1, 1, 8, 8, dtype=torch.complex64), {})


def test_mask_on_a_nufft_trajectory_raises() -> None:
    op = Operator_A_m4(im_size=(8, 8))
    maps = {"dK": torch.zeros(1, 2, 64), "mask": torch.ones(1, 1, 8, 8)}
    with pytest.raises(ValueError, match="Cartesian grid only"):
        op.forward(torch.zeros(1, 1, 8, 8, dtype=torch.complex64), maps)
