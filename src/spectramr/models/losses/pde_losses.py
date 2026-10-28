from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss(name="helmholtz_pde", aliases=["HelmholtzLoss", "pde"])
class HelmholtzPDELoss(nn.Module):
    r"""Computes the Helmholtz Wave Equation residual loss via analytical 2nd-order derivatives.

    On a grid the Laplacian is the 5-point stencil in pixel units with zero
    padding, so ``k_sq`` is in pixel⁻².
    """

    laplacian_kernel: torch.Tensor

    def __init__(self, k_sq: float = 0.0) -> None:
        """__init__.

        Args:
            k_sq: Squared wavenumber of ``∇²S + k²S = 0``.
        """
        super().__init__()
        self.k_sq = k_sq
        self.register_buffer(
            "laplacian_kernel",
            torch.tensor([[[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]]]),
            persistent=False,
        )

    def forward(
        self,
        arg1: torch.Tensor,
        arg2: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            arg1: S_real (legacy) or complex pred [B, C, H, W] (CNN).
            arg2: S_imag (legacy) or target [B, C, H, W] (CNN).
            coords: Spatial coordinates for explicit autograd (N, 2).

        Returns:
            Scalar tensor containing the Mean Squared PDE residual.

        forward method for HelmholtzPDELoss.

        Executes PyTorch tensor operations.

        Args:
            arg1 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            arg2 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            coords (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # If arg1 is 4D grid tensor, we compute finite difference Laplacian
        if arg1.dim() == 4:
            return self._grid_residual(arg1)

        # Legacy SIREN path
        s_real = arg1
        s_imag = arg2
        if coords is None:
            coords = kwargs.get("coords")
        if s_imag is None or coords is None or not coords.requires_grad:
            raise ValueError(
                "The autograd PDE path needs S_imag and coords with requires_grad=True."
            )

        num_coils = s_real.shape[1]
        total_pde_res = s_real.new_zeros(())

        for c in range(num_coils):
            sr = s_real[..., c].unsqueeze(-1)
            si = s_imag[..., c].unsqueeze(-1)

            grad_r = torch.autograd.grad(
                sr, coords, grad_outputs=torch.ones_like(sr), create_graph=True
            )[0]
            grad_i = torch.autograd.grad(
                si, coords, grad_outputs=torch.ones_like(si), create_graph=True
            )[0]

            grad_r_x = grad_r[..., 0].unsqueeze(-1).contiguous()
            grad_r_y = grad_r[..., 1].unsqueeze(-1).contiguous()

            d2r_dx2 = (
                torch.autograd.grad(
                    grad_r_x,
                    coords,
                    grad_outputs=torch.ones_like(grad_r_x),
                    create_graph=True,
                )[0][..., 0]
                .unsqueeze(-1)
                .contiguous()
            )

            d2r_dy2 = (
                torch.autograd.grad(
                    grad_r_y,
                    coords,
                    grad_outputs=torch.ones_like(grad_r_y),
                    create_graph=True,
                )[0][..., 1]
                .unsqueeze(-1)
                .contiguous()
            )

            grad_i_x = grad_i[..., 0].unsqueeze(-1).contiguous()
            grad_i_y = grad_i[..., 1].unsqueeze(-1).contiguous()

            d2i_dx2 = (
                torch.autograd.grad(
                    grad_i_x,
                    coords,
                    grad_outputs=torch.ones_like(grad_i_x),
                    create_graph=True,
                )[0][..., 0]
                .unsqueeze(-1)
                .contiguous()
            )

            d2i_dy2 = (
                torch.autograd.grad(
                    grad_i_y,
                    coords,
                    grad_outputs=torch.ones_like(grad_i_y),
                    create_graph=True,
                )[0][..., 1]
                .unsqueeze(-1)
                .contiguous()
            )

            laplacian_r = d2r_dx2 + d2r_dy2
            laplacian_i = d2i_dx2 + d2i_dy2

            res_r = laplacian_r + self.k_sq * sr
            res_i = laplacian_i + self.k_sq * si

            total_pde_res = total_pde_res + torch.mean(res_r**2 + res_i**2)

        return total_pde_res / num_coils

    def _grid_residual(self, pred: torch.Tensor) -> torch.Tensor:
        """Mean-squared Helmholtz residual of a ``[B, C, H, W]`` grid, averaged over channels.

        Every channel goes through one convolution as a ``[B*C, 1, H, W]`` batch; a
        real input has no imaginary residual, so none is computed for it.
        """
        b, c, h, w = pred.shape
        parts = (pred.real, pred.imag) if pred.is_complex() else (pred,)
        kernel = self.laplacian_kernel.to(device=pred.device, dtype=parts[0].dtype)
        sq_residual = parts[0].new_zeros(())
        for part in parts:
            flat = part.reshape(b * c, 1, h, w)
            residual = F.conv2d(flat, kernel, padding=1) + self.k_sq * flat
            sq_residual = sq_residual + residual.pow(2).mean()
        return sq_residual
