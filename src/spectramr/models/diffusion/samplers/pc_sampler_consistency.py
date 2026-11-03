"""Predictor-corrector sampling of a VE-SDE with k-space data consistency.

Reverse-diffusion predictor and Langevin corrector (Song et al., "Score-Based Generative
Modeling through Stochastic Differential Equations", ICLR 2021, Alg. 2), with the measured
k-space bins re-imposed after every update (Chung and Ye, "Score-based diffusion models for
accelerated MRI", MedIA 2022, Alg. 1). The image is real-valued, so each consistency step
keeps the real part of the blended image.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

from .registry import register_sampler

#: Song et al. (2021) VE-SDE defaults: Langevin signal-to-noise ratio, corrector steps per
#: noise level, and the smallest time the reverse chain reaches.
DEFAULT_SNR = 0.16
DEFAULT_CORRECTOR_STEPS = 1
SAMPLING_EPS = 1e-5

ScoreFn = Callable[[Tensor, Tensor], Tensor]


class VarianceExplodingSDE(Protocol):
    """What the sampler reads off ``score_sde.sde.VESDE`` (structural, to keep imports acyclic)."""

    sigma_max: float

    def sigma(self, t: Tensor) -> Tensor: ...

    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> Tensor: ...


@register_sampler("pc_sampler_consistency")
class PCSamplerConsistency:
    """VE-SDE predictor-corrector sampler that keeps the acquired k-space.

    Args:
        score_fn: ``score_fn(x, t)`` returns the score of ``p_t`` at ``x`` for times ``t``
            of shape ``[B]``.
        sde: The VE-SDE the score was trained on; its ``sigma(t)`` sets every step size.
        n_steps: Number of noise levels between ``t = 1`` and ``t = SAMPLING_EPS``.
        snr: Langevin signal-to-noise ratio.
        corrector_steps: Langevin updates per noise level.
        dc_weight: Weight of the measured bins in the blend ``k + w * m * (y - k)``; 1.0 is a
            hard replacement.
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        sde: VarianceExplodingSDE,
        n_steps: int,
        snr: float = DEFAULT_SNR,
        corrector_steps: int = DEFAULT_CORRECTOR_STEPS,
        dc_weight: float = 1.0,
    ) -> None:
        if not (hasattr(sde, "sigma") and hasattr(sde, "prior_sampling")):
            raise TypeError(
                f"pc_sampler_consistency integrates a VE-SDE, got {type(sde).__name__}."
            )
        if int(n_steps) < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        if snr <= 0.0:
            raise ValueError(f"snr must be > 0, got {snr}.")
        if int(corrector_steps) < 0:
            raise ValueError(f"corrector_steps must be >= 0, got {corrector_steps}.")
        if not 0.0 < dc_weight <= 1.0:
            raise ValueError(f"dc_weight must be in (0, 1], got {dc_weight}.")
        self.score_fn = score_fn
        self.sde = sde
        self.n_steps = int(n_steps)
        self.snr = float(snr)
        self.corrector_steps = int(corrector_steps)
        self.dc_weight = float(dc_weight)

    def _consistency(self, x: Tensor, measurement: Tensor, mask: Tensor) -> Tensor:
        k = fft2c(x)
        k = k + (self.dc_weight * mask) * (measurement - k)
        return ifft2c(k).real.to(x.dtype)

    def _langevin(self, x: Tensor, t_vec: Tensor) -> Tensor:
        score = self.score_fn(x, t_vec)
        noise = torch.randn_like(x)
        batch = x.shape[0]
        grad_norm = torch.linalg.vector_norm(score.reshape(batch, -1), dim=-1).mean()
        noise_norm = torch.linalg.vector_norm(noise.reshape(batch, -1), dim=-1).mean()
        step = 2.0 * (self.snr * noise_norm / grad_norm.clamp_min(1e-12)) ** 2
        return x + step * score + torch.sqrt(2.0 * step) * noise

    def sample(
        self,
        measurement: Tensor,
        mask: Tensor,
        shape: tuple[int, ...] | torch.Size | None = None,
    ) -> Tensor:
        """Draw ``x ~ p(x | y)`` from the prior ``N(0, sigma_max^2 I)`` down to ``t = eps``.

        Args:
            measurement: Undersampled complex k-space ``y = m * F(x)``, ``[B, C, H, W]``.
            mask: Real sampling mask broadcastable to ``measurement``.
            shape: Image shape; defaults to ``measurement.shape``.

        Returns:
            The denoised final mean, real-valued, of ``shape``.
        """
        if measurement is None or mask is None:
            raise ValueError("pc_sampler_consistency needs both a measurement and its mask.")
        if not torch.is_complex(measurement):
            raise TypeError("measurement must be complex k-space (fft2c of the image).")
        shape = tuple(measurement.shape if shape is None else shape)
        device = measurement.device
        mask = mask.to(device=device, dtype=torch.float32)
        times = torch.linspace(1.0, SAMPLING_EPS, self.n_steps, device=device)
        sigmas = self.sde.sigma(times)
        # The final step steps to sigma = 0, so the chain ends on the noise-free mean.
        sigmas_next = torch.cat([sigmas[1:], sigmas.new_zeros(1)])
        x = self.sde.prior_sampling(shape, device)
        x_mean = x
        for i in range(self.n_steps):
            t_vec = times[i].expand(shape[0])
            for _ in range(self.corrector_steps):
                x = self._consistency(self._langevin(x, t_vec), measurement, mask)
            variance = sigmas[i] ** 2 - sigmas_next[i] ** 2
            x_mean = x + variance * self.score_fn(x, t_vec)
            x = x_mean + torch.sqrt(variance) * torch.randn_like(x)
            x_mean = self._consistency(x_mean, measurement, mask)
            x = self._consistency(x, measurement, mask)
        return x_mean
