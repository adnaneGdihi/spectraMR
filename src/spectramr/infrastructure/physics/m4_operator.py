"""Global Forward Physics Operator for PMA-VarNet (m4raw topology).

Constructs the comprehensive operator A_m4 and its backpropagation-exact
adjoint A_m4^H as a differentiable nn.Module.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

try:
    import torchkbnufft as tkbn

    TORCHKBNUFFT_AVAILABLE = True
except ImportError:
    TORCHKBNUFFT_AVAILABLE = False
    tkbn = None

logger = logging.getLogger(__name__)

GAMMA = 267.513e6  # Gyromagnetic ratio for 1H in rad/(s*T)


def apply_rigid_transform(
    x: torch.Tensor,
    theta: torch.Tensor | None,
) -> torch.Tensor:
    """Apply inverse rigid kinematic affine transform.

    Args:
        x: Image [B, C, H, W]
        theta: Affine matrix [B, 2, 3], or None to pass-through unchanged

    Returns:
        Warped image [B, C, H, W]
    """
    if theta is None:
        return x

    # We must treat complex x as two channels for grid_sample
    orig_shape = x.shape
    if x.is_complex():
        x_real = (
            torch.view_as_real(x)
            .permute(0, 1, 4, 2, 3)
            .reshape(orig_shape[0], orig_shape[1] * 2, orig_shape[2], orig_shape[3])
        )
    else:
        x_real = x

    grid = F.affine_grid(theta, x_real.size(), align_corners=False)
    x_warped = F.grid_sample(
        x_real, grid, mode="bilinear", padding_mode="reflection", align_corners=False
    )

    if x.is_complex():
        x_warped = x_warped.view(
            orig_shape[0], orig_shape[1], 2, orig_shape[2], orig_shape[3]
        ).permute(0, 1, 3, 4, 2)
        x_warped = torch.view_as_complex(x_warped.contiguous())

    return x_warped


def pre_whiten(
    x_kspace: torch.Tensor,
    psi: torch.Tensor,
) -> torch.Tensor:
    """Apply coil pre-whitening using noise covariance.

    Args:
        x_kspace: Coil data [B, Nc, N_k]
        psi: Noise covariance matrix [B, Nc, Nc]

    Returns:
        Whitened data [B, Nc, N_k]
    """
    if psi is None:
        return x_kspace

    # Cholesky decomposition L * L^H = Psi
    # Whitening is L^-1 * x
    # Psi is [B, Nc, Nc]
    L = torch.linalg.cholesky(psi)

    # solve_triangular expects (B, m, m) and (B, m, k)
    whitened = torch.linalg.solve_triangular(L, x_kspace, upper=False)
    return whitened


def _reject_mask_off_grid(maps_dict: dict) -> None:
    """A Cartesian line mask has no meaning on a NUFFT trajectory; raise rather than drop it."""
    if maps_dict.get("mask") is not None:
        raise ValueError("maps_dict['mask'] applies to the Cartesian grid only (dK=None).")


class Operator_A_m4(nn.Module):
    """The Complete Forward Physics Simulator for the m4raw anatomy space.

    Unifies Flip Angle Shading, Kinematic Shift, B0 Phase Roll, Coil Sensitivities,
    Pre-Whitening, and Trajectory-Corrected NUFFT into a single differentiable operator.
    Mathematical Formulation:
    .. math::

        \\mathcal{O}_{A\\_m4}(x) = \\mathcal{F}_{NUFFT}(x \\odot S \\odot e^{-i \\gamma B_0 t} \\odot \\sin(\alpha))"""

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        readout_time: float = 0.005,
        norm_nufft: bool = True,
    ) -> None:
        super().__init__()
        self.im_size = im_size
        self.readout_time = readout_time
        # torchkbnufft >=1.4 removed `norm` from __init__;
        # it must be passed at forward-time instead.
        self._nufft_norm: str | None = "ortho" if norm_nufft else None

        if TORCHKBNUFFT_AVAILABLE:
            self.nufft = tkbn.KbNufft(im_size=im_size)
            self.adjnufft = tkbn.KbNufftAdjoint(im_size=im_size)
        else:
            logger.warning("TorchKbNufft not found. M4 Operator will fail on execution.")

    def forward(
        self,
        x_tissue: torch.Tensor,
        maps_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply full forward model A_{m4}(x).

        Args:
            x_tissue: Clean image [B, 1, H, W]
            maps_dict: Dictionary containing:
                - alpha: [B, 1, H, W]
                - T_kin: [B, 2, 3] affine matrix
                - B0: [B, 1, H, W] in Tesla
                - S_map: [B, Nc, H, W]
                - Psi: [B, Nc, Nc]
                - dK: [B, 2, N_k] trajectory (nominal + tracking adjustment), or
                  None for the Cartesian grid of ``fft2c`` (required key)
                - mask: binary sampling mask broadcastable to [B, Nc, H, W]
                  (Cartesian only, optional)
                - im_size: (H, W) of the Cartesian grid (optional, else self.im_size)
                - dcf: [B, 1, N_k] density compensation (optional)

        Returns:
            Simulated k-space y [B, Nc, N_k]

        forward method for Operator_A_m4.

        Executes PyTorch tensor operations.

        Args:
            x_tissue (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            maps_dict (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Flip Angle Shading (Task 4.8)
        if "alpha" in maps_dict and maps_dict["alpha"] is not None:
            x = x_tissue * torch.sin(maps_dict["alpha"])
        else:
            x = x_tissue

        # 2. Kinematic Affine Shift (Task 4.1)
        if "T_kin" in maps_dict and maps_dict["T_kin"] is not None:
            x = apply_rigid_transform(x, maps_dict["T_kin"])

        # 3. B0 Off-Resonance Phase Roll (Task 4.2)
        if "B0" in maps_dict and maps_dict["B0"] is not None:
            phase = -1j * GAMMA * maps_dict["B0"] * self.readout_time
            x = x * torch.exp(phase)

        # 4. Coil Sensitivities
        if "S_map" in maps_dict and maps_dict["S_map"] is not None:
            x = x * maps_dict["S_map"]  # [B, Nc, H, W]

        # 5. Trajectory-Corrected NUFFT (Task 4.3), or the Cartesian fft2c grid
        ktraj = maps_dict["dK"]  # [B, 2, N_k]
        if ktraj is None:
            kx = self._cartesian_forward(x, maps_dict)
            if "Psi" in maps_dict and maps_dict["Psi"] is not None:
                kx = pre_whiten(kx, maps_dict["Psi"])
            return kx

        _reject_mask_off_grid(maps_dict)
        if not TORCHKBNUFFT_AVAILABLE:
            raise RuntimeError("TorchKbNufft required for NUFFT.")

        kx = self.nufft(x, ktraj, norm=self._nufft_norm)  # [B, Nc, N_k]

        # 6. Pre-Whitening (Task 4.6)
        if "Psi" in maps_dict and maps_dict["Psi"] is not None:
            kx = pre_whiten(kx, maps_dict["Psi"])

        return kx

    def adjoint(
        self,
        y: torch.Tensor,
        maps_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply exact adjoint A_{m4}^H(y).

        Args:
            y: K-space measurements [B, Nc, N_k]
            maps_dict: Same fields as forward.

        Returns:
            Adjoint reconstructed image [B, 1, H, W]
        """
        kx = y

        # 0. Density Compensation (if provided)
        if "dcf" in maps_dict and maps_dict["dcf"] is not None:
            kx = kx * maps_dict["dcf"]

        # 1. Un-Whitening (Adjoint of Whitening)
        if "Psi" in maps_dict and maps_dict["Psi"] is not None:
            # Whitening is L^-1 * x. Adjoint is (L^-1)^H * x = L^-H * x
            L = torch.linalg.cholesky(maps_dict["Psi"])
            kx = torch.linalg.solve_triangular(L.mH, kx, upper=True)

        # 2. NUFFT Adjoint, or the Cartesian ifft2c grid
        ktraj = maps_dict["dK"]
        if ktraj is None:
            x = self._cartesian_adjoint(kx, maps_dict)
        else:
            _reject_mask_off_grid(maps_dict)
            if not TORCHKBNUFFT_AVAILABLE:
                raise RuntimeError("TorchKbNufft required for NUFFT.")
            x = self.adjnufft(kx, ktraj, norm=self._nufft_norm)  # [B, Nc, H, W]

        # 3. Adjoint Coil Sensitivities (Coil Combination)
        if "S_map" in maps_dict and maps_dict["S_map"] is not None:
            x = torch.sum(x * maps_dict["S_map"].conj(), dim=1, keepdim=True)  # [B, 1, H, W]

        # 4. Adjoint B0 Phase Roll
        if "B0" in maps_dict and maps_dict["B0"] is not None:
            phase = 1j * GAMMA * maps_dict["B0"] * self.readout_time
            x = x * torch.exp(phase)

        # 5. Adjoint Kinematic Shift
        if "T_kin" in maps_dict and maps_dict["T_kin"] is not None:
            T = maps_dict["T_kin"]  # [B, 2, 3]
            R = T[:, :, :2]  # [B, 2, 2]
            t = T[:, :, 2:]  # [B, 2, 1]
            R_inv = R.transpose(1, 2)
            t_inv = -torch.bmm(R_inv, t)
            T_inv = torch.cat([R_inv, t_inv], dim=2)
            x = apply_rigid_transform(x, T_inv)

        # 6. Adjoint Flip Angle Shading
        if "alpha" in maps_dict and maps_dict["alpha"] is not None:
            x = x * torch.sin(maps_dict["alpha"])  # sin is real, self-adjoint

        return x

    def _cartesian_forward(self, x: torch.Tensor, maps_dict: dict) -> torch.Tensor:
        """``mask * fft2c(x)`` flattened row-major to ``[B, Nc, H*W]``."""
        k = fft2c(x)
        mask = maps_dict.get("mask")
        if mask is not None:
            k = k * mask
        return k.flatten(-2)

    def _cartesian_adjoint(self, y: torch.Tensor, maps_dict: dict) -> torch.Tensor:
        """``ifft2c(mask * y)`` on the declared ``(H, W)`` grid; the mask is diagonal."""
        H, W = maps_dict.get("im_size") or self.im_size
        if y.shape[-1] != H * W:
            raise ValueError(
                f"Cartesian adjoint: N_k={y.shape[-1]} does not match the grid {H}x{W}; "
                "pass maps_dict['im_size'] for a grid other than the operator's im_size."
            )
        k = y.reshape(*y.shape[:-1], H, W)
        mask = maps_dict.get("mask")
        if mask is not None:
            k = k * mask
        return ifft2c(k)


__all__ = ["Operator_A_m4", "apply_rigid_transform", "pre_whiten"]
