r"""Reverse variance-exploding SDE sampler with Poisson stochastic resetting.

The conditional denoiser :math:`D\approx\mathbb E[x_0\mid y,x,\sigma]` trained by
:class:`LevyDenoisingStrategy` at :math:`\alpha=2` gives the score by Tweedie's
identity. The corruption is :math:`\mathcal N(0,2\sigma^2)` in the stable
parameterisation, so :math:`\nabla\log p_\sigma(x)=(D-x)/(2\sigma^2)`. One
Euler-Maruyama step of the reverse VE SDE from :math:`\sigma_i` to :math:`\sigma_{i+1}` is

.. math::

    x \leftarrow x+\frac{\sigma_i^2-\sigma_{i+1}^2}{\sigma_i^2}\,(D-x)
    +\sqrt{\sigma_i^2-\sigma_{i+1}^2}\;z,\qquad z\sim\mathcal N(0,2).

Resetting [Evans & Majumdar, PRL 106, 160601 (2011)] interrupts the reverse
process at Poisson times of rate :math:`r` per unit of log-noise time
:math:`\tau=-\log\sigma`. A step therefore resets with probability
:math:`1-(\sigma_{i+1}/\sigma_i)^r`. A reset sends the sample back to the ladder
level nearest ``reset_sigma`` and re-noises it there,
:math:`x\leftarrow x+\sqrt{\sigma_r^2-\sigma^2}\,z`, which keeps the marginal at
:math:`\sigma_r` exact when :math:`x` was on the marginal at :math:`\sigma`. This
is the restart move of Restart sampling [Xu et al., NeurIPS 2023] at random,
memoryless times. ``max_resets`` bounds the work at
:math:`n+\text{max\_resets}\cdot(n-1-j_r)` evaluations per sample. Rate 0 is
the plain sampler, draw for draw.

Every sample carries its own ladder index, so the batch is denoised in one
forward per iteration with a per-sample noise level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .levy_denoising_strategy import LevyDenoisingStrategy


def noise_ladder(sigma_max: float, sigma_min: float, steps: int) -> list[float]:
    """Geometric noise scales from ``sigma_max`` down to ``sigma_min`` (``steps`` values)."""
    if steps == 1:
        return [sigma_max]
    ratio = (sigma_min / sigma_max) ** (1.0 / (steps - 1))
    return [sigma_max * ratio**i for i in range(steps)]


def reset_index(sigmas: torch.Tensor, reset_sigma: float) -> int:
    """Ladder index whose noise level is nearest ``reset_sigma`` in log space."""
    return int((sigmas.log() - torch.log(torch.tensor(reset_sigma))).abs().argmin())


def resetting_sde_sampler(
    strategy: LevyDenoisingStrategy, cond: torch.Tensor, gen: torch.Generator
) -> torch.Tensor:
    """Euler-Maruyama reverse SDE with per-sample Poisson resets; returns the final x0."""
    cfg = strategy.levy_cfg
    device = cond.device
    sigmas = torch.tensor(
        noise_ladder(cfg.sigma_max, cfg.sigma_min, cfg.sampling_steps), dtype=cond.dtype
    )
    jr = reset_index(sigmas, cfg.reset_sigma) if cfg.reset_rate > 0 else 0
    sigmas = sigmas.to(device)
    n, batch = sigmas.numel(), cond.shape[0]
    shape = strategy.target_shape(cond)
    view = (batch,) + (1,) * (len(shape) - 1)
    reset_gen = torch.Generator(device="cpu").manual_seed(gen.initial_seed() + 1)

    x = sigmas[0] * strategy.draw_noise(shape, device, gen)
    out = torch.zeros(shape, device=device, dtype=cond.dtype)
    idx = torch.zeros(batch, dtype=torch.long, device=device)
    resets = torch.zeros(batch, dtype=torch.long, device=device)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    while not bool(done.all()):
        sigma = sigmas[idx]
        x0 = strategy.denoise(cond, x, sigma)
        last = idx == n - 1
        out = torch.where((last & ~done).view(view), x0, out)
        done = done | last
        nxt = (idx + 1).clamp(max=n - 1)
        var = (sigma.square() - sigmas[nxt].square()).clamp_min(0.0)
        z = strategy.draw_noise(shape, device, gen)
        stepped = x + (var / sigma.square()).view(view) * (x0 - x) + var.sqrt().view(view) * z
        x = torch.where(done.view(view), x, stepped)
        idx = torch.where(done, idx, nxt)
        if cfg.reset_rate > 0:
            p = 1.0 - (sigmas[nxt] / sigma).pow(cfg.reset_rate)
            u = torch.rand(batch, generator=reset_gen).to(device, non_blocking=True)
            fire = ~done & (idx > jr) & (idx < n - 1) & (resets < cfg.max_resets) & (u < p)
            lift = (sigmas[jr].square() - sigmas[idx].square()).clamp_min(0.0).sqrt()
            z_r = strategy.draw_noise(shape, device, reset_gen)
            x = torch.where(fire.view(view), x + lift.view(view) * z_r, x)
            idx = torch.where(fire, torch.full_like(idx, jr), idx)
            resets = resets + fire.long()
    return out
