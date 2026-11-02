"""Annealed Langevin dynamics with hard k-space data consistency.

Reconstruction sampler for a variance-exploding score model trained by denoising
score matching over ``sigma ~ LogUniform[sigma_min, sigma_max]`` (Song & Ermon,
NeurIPS 2019; Jalal et al., NeurIPS 2021 for the MRI posterior variant). At each
level ``sigma_i`` of a geometric ladder the chain takes ``steps_per_level`` steps

    x <- x + alpha_i * s(x, sigma_i) + sqrt(2 * alpha_i) * z,   alpha_i = eps * sigma_i**2

followed by the projection ``k <- (1 - M) * k + M * y`` onto the measured lines,
and ends with one Tweedie denoising step at ``sigma_min`` and a last projection.
``eps * sigma**2`` is Song's ``eps / sigma_L**2`` rewritten scale-free: for a
Gaussian score the chain's stationary variance is ``sigma**2 / (1 - eps / 2)``,
so the iteration is stable for ``eps < 2``.

Images are real tensors ``[B, 2C, H, W]`` in the project's interleaved layout
(even channels real, odd imaginary); k-space goes through ``fft2c``/``ifft2c``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

from .registry import register_sampler

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def interleaved_to_complex(x: torch.Tensor) -> torch.Tensor:
    """``[B, 2C, H, W]`` real interleaved -> ``[B, C, H, W]`` complex."""
    if torch.is_complex(x):
        return x
    if x.shape[1] % 2 != 0:
        raise ValueError(
            f"annealed_langevin_dc needs a real/imag interleaved image (even channel "
            f"count); got {x.shape[1]} channels."
        )
    return torch.complex(x[:, 0::2].float(), x[:, 1::2].float())


def complex_to_interleaved(z: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` complex -> ``[B, 2C, H, W]`` real interleaved."""
    return torch.stack((z.real, z.imag), dim=2).flatten(1, 2)


def project_measured(x: torch.Tensor, measured: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace the measured k-space lines of interleaved image ``x`` with ``measured``."""
    k = fft2c(interleaved_to_complex(x))
    m = mask.to(k.real.dtype)
    k = k + m * (measured - k)
    return complex_to_interleaved(ifft2c(k))


@register_sampler("annealed_langevin_dc", aliases=["annealed_langevin"])
class AnnealedLangevinDCSampler:
    """Annealed Langevin reconstruction from undersampled Cartesian k-space.

    Args:
        sigma_min: Smallest noise level of the ladder (the training minimum).
        sigma_max: Largest noise level of the ladder (the training maximum).
        num_levels: Number of geometric levels, ``>= 2``.
        steps_per_level: Langevin steps per level, ``>= 1``.
        step_size: ``eps`` in ``alpha_i = eps * sigma_i**2``, in ``(0, 2)``.
        seed: Seed of the chain's private noise generator, so repeated
            validation passes of the same weights give the same number.
    """

    def __init__(
        self,
        *,
        sigma_min: float,
        sigma_max: float,
        num_levels: int,
        steps_per_level: int,
        step_size: float,
        seed: int,
    ) -> None:
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError(f"need 0 < sigma_min < sigma_max; got {sigma_min}, {sigma_max}")
        if num_levels < 2 or steps_per_level < 1:
            raise ValueError(
                f"need num_levels >= 2 and steps_per_level >= 1; got {num_levels}, "
                f"{steps_per_level}"
            )
        if not 0.0 < step_size < 2.0:
            raise ValueError(f"step_size must lie in (0, 2); got {step_size}")
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.num_levels = int(num_levels)
        self.steps_per_level = int(steps_per_level)
        self.step_size = float(step_size)
        self.seed = int(seed)

    @property
    def num_score_evaluations(self) -> int:
        """Score-network calls per :meth:`sample` (the Tweedie step included)."""
        return self.num_levels * self.steps_per_level + 1

    def sigmas(self) -> list[float]:
        """The geometric ladder, ``sigma_max`` first."""
        lo, hi = math.log(self.sigma_min), math.log(self.sigma_max)
        n = self.num_levels - 1
        return [math.exp(hi + (lo - hi) * i / n) for i in range(self.num_levels)]

    @torch.no_grad()
    def sample(
        self,
        score_fn: ScoreFn,
        x_init: torch.Tensor,
        measured: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the chain from ``x_init + sigma_max * z``.

        Args:
            score_fn: ``score_fn(x, sigma[B]) -> s_x`` with ``x``'s shape.
            x_init: Interleaved starting image ``[B, 2C, H, W]`` (zero-filled).
            measured: Complex measured k-space ``[B, C, H, W]``, zero off the mask.
            mask: Real sampling mask broadcastable to ``measured``.

        Returns:
            The reconstructed interleaved image, consistent with ``measured``.
        """
        gen = torch.Generator(device=x_init.device).manual_seed(self.seed)
        b = x_init.shape[0]

        def noise() -> torch.Tensor:
            return torch.randn(x_init.shape, generator=gen, device=x_init.device)

        x = x_init.float() + self.sigma_max * noise()
        for sigma in self.sigmas():
            sig = x.new_full((b,), sigma)
            alpha = self.step_size * sigma * sigma
            for _ in range(self.steps_per_level):
                x = x + alpha * score_fn(x, sig) + math.sqrt(2.0 * alpha) * noise()
                x = project_measured(x, measured, mask)
        sig = x.new_full((b,), self.sigma_min)
        x = x + self.sigma_min**2 * score_fn(x, sig)
        return project_measured(x, measured, mask)


__all__ = [
    "AnnealedLangevinDCSampler",
    "complex_to_interleaved",
    "interleaved_to_complex",
    "project_measured",
]
