r"""Focal Frequency Loss (FFL) for MRI Reconstruction.

Penalises reconstructions that fail to recover high-frequency k-space
components, which standard L1/L2 losses treat uniformly. Critical for
correcting Gibbs truncation artefacts and super-resolution.

.. math::

    \mathcal{L}_{\text{FFL}} = \sum_{u,v}
        W(u,v)\,\bigl\|\mathcal{F}(\hat{x}_0)(u,v)
                       - \mathcal{F}(x_0)(u,v)\bigr\|_2^2

The weight matrix :math:`W(u,v)` up-weights the k-space periphery (high
frequencies) using a radial power law: ``W = r^alpha`` where *r* is the
normalised distance from DC.

Reference:
    Jiang et al., "Focal Frequency Loss for Image Reconstruction and
    Synthesis", *ICCV* (2021).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from spectramr.infrastructure.physics.fft_ops import _to_complex, fft2c
from spectramr.models.losses.registry import register_loss

_WEIGHT_MODES = ("radial", "parabolic")


def _build_radial_weight(
    height: int,
    width: int,
    alpha: float = 2.0,
    mode: str = "radial",
    device: torch.device | None = None,
) -> Tensor:
    """Build radial frequency weight matrix.

    Returns:
        Weight tensor ``(1, 1, H, W)`` with values in ``[0, 1]``, raised to
        the power *alpha* so that periphery (r ≈ 1) gets maximum weight and
        DC (r ≈ 0) gets near-zero weight.
    """
    ky = torch.linspace(-1.0, 1.0, height, device=device)
    kx = torch.linspace(-1.0, 1.0, width, device=device)
    grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")

    if mode == "parabolic":
        # W(k) = 1 + alpha * (kx^2 + ky^2)
        w = 1.0 + alpha * (grid_x**2 + grid_y**2)
    else:
        r = torch.sqrt(grid_y**2 + grid_x**2).clamp(max=1.0)
        w = r.pow(alpha)

    return w.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


@register_loss(
    name="focal_frequency_radial",
    aliases=["RadialFocalFrequencyLoss", "ffl_radial", "focal_frequency_parabolic"],
    domain="agnostic",
)
class FocalFrequencyLoss(nn.Module):
    r"""Focal Frequency Loss with configurable peripheral k-space weighting.

    **DOMAIN**: AGNOSTIC (works on complex k-space or real images)
    **Input**: ``[B, C, H, W]`` (complex, 2-channel real/imag, or magnitude)
    **3D Support**: No (2D FFT only)

    Example::

        loss_fn = FocalFrequencyLoss(alpha=2.0)
        loss = loss_fn(pred, target)
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{FocalFrequency} = \mathbb{E}[ W(k_x, k_y) \odot | \mathcal{F}(\hat{y}) - \mathcal{F}(y) |^2 ]"""

    def __init__(self, alpha: float = 2.0, mode: str = "radial") -> None:
        """Initialise FFL.

        Args:
            alpha: Radial power exponent.  Higher values penalise periphery
                more aggressively.  ``alpha=0`` recovers uniform spectral L2.
            mode: "radial" or "parabolic".
        """
        super().__init__()
        if mode not in _WEIGHT_MODES:
            raise ValueError(f"mode must be one of {_WEIGHT_MODES}, got {mode!r}")
        self.alpha = alpha
        self.mode = mode
        # Cache weight matrix per spatial size to avoid recomputation
        self._cached_weight: Tensor | None = None
        self._cached_size: tuple[int, int] = (0, 0)

    def _get_weight(self, h: int, w: int, device: torch.device) -> Tensor:
        """Return (possibly cached) weight matrix."""
        if self._cached_size != (h, w) or self._cached_weight is None:
            self._cached_weight = _build_radial_weight(
                h,
                w,
                self.alpha,
                self.mode,
                device=device,
            )
            self._cached_size = (h, w)
        return self._cached_weight.to(device)

    def forward(self, pred: Tensor, target: Tensor, **kwargs: Any) -> Tensor:
        """Compute the Focal Frequency Loss.

        Args:
            pred: Model prediction ``(B, C, H, W)``.
            target: Ground truth ``(B, C, H, W)``.

        Returns:
            Scalar loss.
        """
        # Ensure complex
        pred_c = _to_complex(pred)
        tgt_c = _to_complex(target)

        # FFT
        pred_k = fft2c(pred_c)
        tgt_k = fft2c(tgt_c)

        # Spectral error  (complex magnitude squared per frequency)
        diff = pred_k - tgt_k
        err = diff.real**2 + diff.imag**2  # (B, C, H, W) real

        # Weight matrix
        H, W = err.shape[-2:]
        weight = self._get_weight(H, W, err.device)

        # Weighted mean
        return (weight * err).mean()


__all__ = ["FocalFrequencyLoss"]
