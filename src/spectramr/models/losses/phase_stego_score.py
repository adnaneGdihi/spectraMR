r"""Phase-stego marker-score loss for Twin-DPS (Phase 3 / Idea 3).

The Twin-DPS reverse-SDE adds two likelihood gradients to the score
model's drift: one from the acquired k-space measurement and one from
the phase-stego marker. This module provides the latter — the
gradient of :math:`\log p(M \mid x_t)` under the phase-stego forward
map :math:`M(x) = \mathcal{F}_{\text{stego}}(\arg(x))`, assuming a
Gaussian observation noise model on the marker:

.. math::

    \log p(M \mid x_t) \;=\; -\frac{1}{2\sigma_M^2}
        \|M(\arg(\hat x_0(x_t))) - M\|_2^2 + \text{const.}

where :math:`\hat x_0(x_t)` is Tweedie's posterior-mean estimate.

The loss is registered as ``phase_stego_score`` under ``domain='complex'``
so it composes with the existing ``kspace_likelihood_score`` already
used by the diffusion subsystem.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import _to_complex
from spectramr.models.losses.registry import register_loss


@register_loss(name="phase_stego_score", domain="complex")
class PhaseStegoScoreLoss(nn.Module):
    """Squared-residual under the phase-stego forward map.

    Args:
        sigma_M: standard deviation of the marker observation noise.
            Lower → tighter coupling to the marker; values <= 0.01 risk
            numerical instability when the marker is far from the
            current estimate.
        basis: ``"fourier"`` or ``"wavelet"`` — selects the phase-stego
            forward transform. ``"fourier"`` matches the canonical
            ``vf_m7_phase_steganography`` arm.

    The loss returns the per-sample residual norm; the strategy
    multiplies by the ``lambda_M`` weight before adding to the total
    sampling drift.
    """

    def __init__(self, sigma_M: float = 0.05, basis: str = "fourier") -> None:  # noqa: N803
        super().__init__()
        if sigma_M <= 0:
            raise ValueError(f"sigma_M must be > 0; got {sigma_M}.")
        if basis not in ("fourier", "wavelet"):
            raise ValueError(f"basis must be 'fourier' or 'wavelet'; got {basis!r}.")
        self.sigma_M = float(sigma_M)
        self.basis = basis

    def _phase_stego_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phase-only forward map M(arg(x)).

        For the Fourier basis this is FFT(exp(i * arg(x))). For the
        wavelet basis it's the 2D wavelet transform of the same phase
        unit vector. The wavelet branch falls back to a Haar transform
        implemented via 2-D conv if PyWavelets isn't installed.
        """
        # `x.to(torch.complex64)` was a SILENT upcast, and this loss is
        # registered `domain="complex"`. On a real tensor it set the imaginary
        # part to zero, so `torch.angle` was identically 0 (or pi) -- a constant
        # -- and the whole loss had a zero gradient while still reporting a
        # large non-zero value: a term that contributes to the total and
        # nothing to the update (pitfall #16). On interleaved real/imag input
        # (the framework's canonical complex layout, even C) it was worse: it
        # read 2 channels as 2 separate zero-imaginary images and discarded the
        # phase entirely.
        #
        # `_to_complex` is the SSOT for that conversion -- it understands both
        # the last-dim-2 and the interleaved-channel encodings -- and
        # `strict=True` raises on a genuinely real tensor instead of degrading
        # (non-negotiable #3).
        x = _to_complex(x, strict=True)
        phase = torch.angle(x)
        # exp(i*phase) lives on the unit circle — captures the phase
        # geometry without the magnitude confound.
        phase_unit = torch.complex(torch.cos(phase), torch.sin(phase))
        if self.basis == "fourier":
            spectrum: torch.Tensor = torch.fft.fftn(phase_unit, dim=(-2, -1), norm="ortho")
            return spectrum
        # Haar-style 2-D wavelet via stride-2 convolutions (LL band).
        # Avoids a hard dependency on PyWavelets.
        device, dtype = phase_unit.device, phase_unit.dtype
        kernel = 0.5 * torch.ones(1, 1, 2, 2, dtype=dtype, device=device)
        # Broadcast to channel dim.
        if phase_unit.ndim == 4:
            n = phase_unit.shape[1]
            kernel = kernel.expand(n, 1, 2, 2)
            return torch.nn.functional.conv2d(phase_unit, kernel, stride=2, groups=n)
        raise ValueError(
            f"Phase-stego wavelet branch expects (B, C, H, W); got {tuple(phase_unit.shape)}."
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        r"""Compute :math:`\| M(\arg(\hat{x}_0)) - M \|_2^2 / (2 \sigma_M^2)`.

        Args:
            prediction: Tweedie estimate :math:`\hat{x}_0(x_t)`,
                complex tensor.
            target: marker template :math:`M`, same shape as the
                stego-transform of ``prediction``.

        Returns:
            Scalar tensor — the negative log-likelihood (up to const)
            of the marker under the current estimate.
        """
        m_hat = self._phase_stego_forward(prediction)
        target_typed = target.to(m_hat.dtype)
        try:
            target_b = target_typed.expand(m_hat.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Marker template shape {tuple(target.shape)} does not "
                f"broadcast to phase-stego forward output {tuple(m_hat.shape)}."
            ) from exc
        residual = m_hat - target_b
        return (residual.abs().pow(2).sum()) / (2.0 * self.sigma_M**2)


__all__ = ["PhaseStegoScoreLoss"]
