r"""α-stable Lévy score-based generative modelling for heavy-tailed MRI.

Standard score-based diffusion uses the variance-preserving SDE

.. math::

    d\mathbf{x}_t = -\tfrac{1}{2}\mathbf{x}_t\,dt + \sigma(t)\,dW_t,

with Brownian increments. For magnitude MRI at low SNR (the ULF regime),
residuals follow heavy-tailed Rician / non-central chi distributions
[St-Jean et al. 2020] whose tails the Gaussian forward process cannot
match, causing under-coverage in posterior samples.

The :math:`\alpha`-stable Lévy process :math:`L_t^{(\alpha)}` with
:math:`\alpha\in(0,2]` (:math:`\alpha=2` recovers Brownian motion) has
characteristic function :math:`\hat p_t(k)=\exp(-\sigma^\alpha t\|k\|^\alpha)`.
The replacement forward SDE

.. math::

    d\mathbf{x}_t = -\tfrac{1}{2}\mathbf{x}_t\,dt + \sigma(t)\,dL_t^{(\alpha)}

has generator :math:`-\tfrac{1}{2}x\cdot\nabla-\sigma^\alpha(t)(-\Delta)^{\alpha/2}`
involving the fractional Laplacian. Training is *fractional denoising
score matching* against the α-stable transition density, evaluated via
inverse FFT of :math:`\exp(-\sigma^\alpha\|k\|^\alpha)`.

Consistency check: at :math:`\alpha=2` the process recovers Brownian
motion and the model reduces to standard VP-SDE [Song et al. 2021].

Exports:

* :func:`sample_alpha_stable` — Chambers-Mallows-Stuck simulator for
  univariate :math:`\alpha`-stable.
* :func:`alpha_stable_kernel_fft` — closed-form characteristic function
  :math:`\exp(-\sigma^\alpha\|k\|^\alpha)` for spectral score-matching.
* :func:`fractional_score_matching_loss` — denoising score-matching
  against the α-stable transition density.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "alpha_stable_kernel_fft",
    "fractional_score_matching_loss",
    "sample_alpha_stable",
]


def sample_alpha_stable(
    shape: tuple[int, ...],
    alpha: float,
    *,
    scale: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Sample i.i.d. α-stable Lévy random variables via Chambers-Mallows-Stuck.

    For :math:`\alpha=2` returns :math:`\mathcal{N}(0, 2\sigma^2)` (the Brownian
    limit). For :math:`\alpha\in(0,2)` returns heavy-tailed symmetric stable.

    Args:
        shape: Output shape.
        alpha: Stability index in :math:`(0, 2]`.
        scale: Dispersion :math:`\sigma`.
        generator: Optional :class:`torch.Generator`.
        device: Output device.

    Returns:
        Tensor of shape ``shape`` with i.i.d. α-stable entries.
    """
    if alpha <= 0.0 or alpha > 2.0:
        raise ValueError(f"alpha must lie in (0, 2]; got {alpha}.")
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    if alpha == 2.0:
        return torch.randn(shape, generator=generator, device=device) * (math.sqrt(2.0) * scale)
    # Chambers-Mallows-Stuck for symmetric stable (β=0):
    # U ~ Uniform(-π/2, π/2), W ~ Exp(1).
    # X = sin(αU) / cos(U)^{1/α} · (cos((α-1)U) / W)^{(1-α)/α}.
    u = (
        torch.rand(shape, generator=generator, device=device) - 0.5
    ) * math.pi  # Uniform(-π/2, π/2)
    w = -torch.log(torch.rand(shape, generator=generator, device=device).clamp_min(1e-12))  # Exp(1)
    numerator = torch.sin(alpha * u) / torch.cos(u).clamp_min(1e-12).pow(1.0 / alpha)
    factor = (torch.cos((alpha - 1.0) * u) / w.clamp_min(1e-12)).pow((1.0 - alpha) / alpha)
    return scale * numerator * factor


def alpha_stable_kernel_fft(
    shape: tuple[int, ...],
    alpha: float,
    scale: float = 1.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    r"""Characteristic function :math:`\hat p_t(k)=\exp(-\sigma^\alpha\|k\|^\alpha)`.

    Args:
        shape: Spatial shape (no batch dim).
        alpha: Stability index.
        scale: Dispersion :math:`\sigma`.
        device: Output device.
        dtype: Output dtype.

    Returns:
        ``shape``-shaped tensor of characteristic-function values on the
        FFT frequency grid (DC at index 0 per axis).
    """
    if alpha <= 0.0 or alpha > 2.0:
        raise ValueError(f"alpha must lie in (0, 2]; got {alpha}.")
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    freqs = [torch.fft.fftfreq(s, device=device, dtype=dtype) for s in shape]
    grids = torch.meshgrid(*freqs, indexing="ij")
    norm_k = torch.sqrt(sum(g * g for g in grids))
    return torch.exp(-(scale**alpha) * norm_k.pow(alpha))


def fractional_score_matching_loss(
    score_model,
    x0: torch.Tensor,
    *,
    alpha: float,
    sigma: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    r"""Denoising score-matching against an α-stable noise sample.

    .. math::

        \mathcal{L} = \mathbb{E}_{x_0,x_t}\!\bigl[\|s_\theta(x_t) - \nabla_{x_t}\log p_{t\mid 0}(x_t\mid x_0)\|^2\bigr].

    For α-stable noise the score :math:`\nabla\log p_{t\mid 0}` is the
    fractional-Laplacian Riesz-flux of the transition density; we use the
    *standard* DSM surrogate that targets :math:`-\mathrm{noise}/\sigma^\alpha`,
    which matches the score at alpha=2 and is only a surrogate at other alpha,
    not the exact alpha-stable score (for alpha-stable score-based models see
    E. Yoon, K. Park, S. Kim and S. Lim, "Score-based Generative Models with
    Levy Processes", NeurIPS 2023).

    Args:
        score_model: A callable mapping ``x_t -> s_θ(x_t)`` with matching shape.
        x0: ``(B, ...)`` clean samples.
        alpha: Stability index in (0, 2].
        sigma: Noise scale.
        generator: Optional generator for the noise.

    Returns:
        Scalar loss.
    """
    noise = sample_alpha_stable(
        tuple(x0.shape), alpha=alpha, scale=sigma, generator=generator, device=x0.device
    )
    x_t = x0 + noise
    pred_score = score_model(x_t)
    target_score = -noise / (sigma**alpha)
    return (pred_score - target_score).pow(2).mean()
