"""Reverse diffusion on a subsequence of the training schedule (DDIM).

A model trained on a ``T``-step noise schedule has learned one noise level per
training index ``t in [0, T)``. Sampling it in ``S < T`` steps must keep that
schedule and visit a subsequence of its indices, giving the model the training
index at every step. Building a fresh ``S``-step schedule instead feeds the
model indices ``0..S-1`` that name different noise levels, which is what
``spectramr infer`` did (#2305).

The update is DDIM (Song, Meng & Ermon, ICLR 2021, eq. 12), which is defined
between any two indices of the schedule: ``eta = 0`` is deterministic and
``eta = 1`` matches the DDPM posterior variance.
"""

from __future__ import annotations

import torch

__all__ = ["ddim_step", "strided_timesteps"]


def strided_timesteps(num_train_timesteps: int, num_steps: int) -> list[int]:
    """Training indices to visit, noisiest first, ending one stride above 0.

    Trailing spacing: the first index is ``T - 1``, so sampling starts at the
    noise level the Gaussian initial state corresponds to. The step after the
    last index lands on the clean sample (``alpha_bar = 1``).

    Raises:
        ValueError: ``num_steps`` is not in ``[1, num_train_timesteps]``.
    """
    if not 1 <= num_steps <= num_train_timesteps:
        raise ValueError(
            f"num_steps={num_steps} must be in [1, {num_train_timesteps}] "
            "(the training schedule's length): a sampler cannot visit more "
            "noise levels than the model was trained on."
        )
    ratio = num_train_timesteps / num_steps
    return [round((i + 1) * ratio) - 1 for i in reversed(range(num_steps))]


def ddim_step(
    x_t: torch.Tensor,
    eps: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    alpha_bar_prev: torch.Tensor | float,
    eta: float,
) -> torch.Tensor:
    """One DDIM update from index ``t`` to the previous visited index.

    Args:
        x_t: Current sample.
        eps: The model's noise prediction at ``t``.
        alpha_bar_t: ``alpha_bar`` at ``t``.
        alpha_bar_prev: ``alpha_bar`` at the next index to visit, or ``1.0``
            after the last one.
        eta: Stochasticity, ``0`` (deterministic) to ``1`` (DDPM variance).
    """
    alpha_bar_prev = torch.as_tensor(alpha_bar_prev, dtype=x_t.dtype, device=x_t.device)
    x0 = (x_t - (1 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
    sigma = (
        eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
    )
    direction = (1 - alpha_bar_prev - sigma**2).clamp_min(0).sqrt() * eps
    x_prev = alpha_bar_prev.sqrt() * x0 + direction
    if eta > 0:
        x_prev = x_prev + sigma * torch.randn_like(x_t)
    return x_prev
