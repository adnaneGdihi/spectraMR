"""Render a blur kernel as a model-input channel (VFU-01).

``forward_psf.source`` names the kernel the network is CONDITIONED on: the
declared Gaussian (``assumed``) or the one solved from the fiducial
(``measured``). Until 2026-10 neither reached the network, so the two arms ran
the same computation. This module turns one kernel per sample into one
full-field channel, so both arms feed the model the same input geometry and
differ only in the kernel's content.

The channel is the modulation transfer function, ``|OTF|``, normalised to 1 at
DC and centred on the grid. A raw ``k x k`` kernel pasted into an ``h x w``
canvas would be zero almost everywhere, which a convolutional encoder barely
sees; the MTF spans the whole field and is exactly the per-frequency
attenuation the network has to invert.
"""

from __future__ import annotations

from torch import Tensor

from spectramr.infrastructure.physics.fft_ops import fft2c


def otf_channel(kernels: Tensor, size: tuple[int, int], *, eps: float = 1e-8) -> Tensor:
    """``|OTF|`` of one kernel per sample, as a ``[B, 1, h, w]`` map.

    Args:
        kernels: ``[B, R, C, k, k]`` kernel field, the layout
            :func:`~spectramr.infrastructure.physics.psf_estimation.estimate_psf`
            returns. Only a single control point (``R = C = 1``) is accepted.
        size: The ``(h, w)`` grid of the stack the channel joins.
        eps: Floor on the DC magnitude before dividing by it.

    Returns:
        Real ``[B, 1, h, w]`` map, equal to 1 at DC (index ``h // 2, w // 2``).

    Raises:
        ValueError: on a spatially varying kernel field (one channel cannot
            carry several kernels), or a kernel larger than the grid.
    """
    if kernels.ndim != 5:
        raise ValueError(f"expected [B, R, C, k, k] kernels, got {tuple(kernels.shape)}")
    b, rows, cols, kh, kw = kernels.shape
    if rows != 1 or cols != 1:
        raise ValueError(
            f"PSF conditioning renders ONE kernel per sample, got a {rows}x{cols} "
            "control grid. A spatially varying operator has no single OTF; set "
            "forward_psf.control_rows = control_cols = 1 for a conditioned arm."
        )
    h, w = int(size[0]), int(size[1])
    if kh > h or kw > w:
        raise ValueError(f"kernel {kh}x{kw} does not fit the {h}x{w} grid")
    canvas = kernels.new_zeros((b, 1, h, w))
    top, left = h // 2 - kh // 2, w // 2 - kw // 2
    canvas[:, :, top : top + kh, left : left + kw] = kernels[:, 0, 0].unsqueeze(1)
    mtf = fft2c(canvas).abs()
    dc = mtf[..., h // 2 : h // 2 + 1, w // 2 : w // 2 + 1].clamp(min=eps)
    return (mtf / dc).to(kernels.dtype)


__all__ = ["otf_channel"]
