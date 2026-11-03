"""Hermitian-tied spectral blocks (Idea 1 of audit_plan_novel).

A standard ``SpectralConv2d`` learns an unconstrained complex kernel on the
retained rfft modes. Here the negative-row modes are tied to the positive
rows by the half-spectrum reflection ``W(-k1, k2) = conj W(k1, k2)``, with the
``k1 = 0`` row real and the unpaired ``-modes1`` row zero. Combined with the
Hermitian symmetry that ``irfft2`` imposes along the width axis, this makes
the circular kernel even in width, so the layer commutes with a width flip
(``x.flip(-1)``) at the parameter count of the untied kernel. It is not
height-flip equivariant. The output is real for any kernel, since ``irfft2``
of a half-spectrum multiplier is a real circular convolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HermitianSpectralConv2d(nn.Module):
    """Spectral convolution whose negative rows reflect the positive rows.

    Stores two real tensors for rows ``k1 = 0 .. modes1-1`` and assembles
    rows ``-1 .. -(modes1-1)`` as their conjugates, which makes the layer
    width-flip equivariant. Parameter count is ``2 * C_in * C_out * modes1 *
    modes2``; the imaginary part of row 0 is stored but masked to zero.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes1: Spectral modes along height (must be ≤ H/2).
        modes2: Spectral modes along width (must be ≤ W/2 + 1 for rfft).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = (2.0 / (in_channels + out_channels)) ** 0.5
        self.w_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))
        self.w_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2))

    def _assemble_kernel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Kernels for rfft rows ``[:modes1]`` and ``[-modes1:]`` (frequencies ``-modes1..-1``)."""
        imag = torch.cat([torch.zeros_like(self.w_imag[:, :, :1]), self.w_imag[:, :, 1:]], 2)
        k_pos = torch.complex(self.w_real, imag)
        # Slice row r holds frequency r - modes1: row 0 (-modes1) has no stored partner.
        k_neg = torch.cat([torch.zeros_like(k_pos[:, :, :1]), k_pos[:, :, 1:].flip(-2).conj()], 2)
        return k_pos, k_neg

    @staticmethod
    def _compl_mul2d(x_ft: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, h, w) × (C_in, C_out, h, w) → (B, C_out, h, w)`` complex."""
        return torch.einsum("bixy,ioxy->boxy", x_ft, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Hermitian-equivariant spectral conv.

        Args:
            x: Real input ``[B, C, H, W]``.

        Returns:
            Real output ``[B, C_out, H, W]``.
        """
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            b,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.promote_types(x_ft.dtype, torch.cfloat),
            device=x.device,
        )
        k_pos, k_neg = self._assemble_kernel()
        out_ft[:, :, : self.modes1, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], k_pos
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], k_neg
        )
        return torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")


class HermitianFNOBlock(nn.Module):
    """FNO block with the Hermitian-tied spectral conv inside.

    Composition: ``SpectralConv → 1x1 Conv (residual) → activation``. Every
    part commutes with a width flip, so the block is width-flip equivariant.
    """

    def __init__(
        self,
        width: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.spectral = HermitianSpectralConv2d(width, width, modes1, modes2)
        self.residual = nn.Conv2d(width, width, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.spectral(x) + self.residual(x))


__all__ = ["HermitianFNOBlock", "HermitianSpectralConv2d"]
