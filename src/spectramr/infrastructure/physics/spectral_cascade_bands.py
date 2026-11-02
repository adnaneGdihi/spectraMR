"""Low-pass circle / high-pass residual split of a real image for the spectral cascade.

The mask keeps the centred k-space samples within ``cutoff`` of DC, measured on
integer frequency coordinates ``i - N // 2``. That set is Hermitian-symmetric
whenever ``cutoff < min(H, W) / 2``, so the low band of a real image is real and
the split ``x = low + high`` is an orthogonal decomposition (Parseval): the
squared error of a band-limited estimate splits exactly into the two bands.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

__all__ = ["low_pass_circle_mask", "split_low_high"]


def low_pass_circle_mask(
    height: int, width: int, cutoff: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Real ``(H, W)`` mask, 1 inside the radius-``cutoff`` disc around centred DC."""
    if 2 * cutoff >= min(height, width):
        raise ValueError(
            f"cutoff_frequency={cutoff} must be < min(H, W) / 2 = {min(height, width) / 2} "
            "for a Hermitian-symmetric low band."
        )
    fy = torch.arange(height, device=device, dtype=torch.float32) - height // 2
    fx = torch.arange(width, device=device, dtype=torch.float32) - width // 2
    radius_sq = fy[:, None].square() + fx[None, :].square()
    return (radius_sq <= float(cutoff) ** 2).to(torch.float32)


def split_low_high(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(low, x - low)`` with ``low = Re(ifft2c(mask * fft2c(x)))`` in float32."""
    x32 = x.float()
    # Explicit complex cast: fft2c reads a real 2-channel tensor as (real, imag).
    k = fft2c(torch.complex(x32, torch.zeros_like(x32)))
    low = ifft2c(k * mask).real
    return low, x32 - low
