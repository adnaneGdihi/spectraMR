"""Data-calibrated latent scale for a from-scratch LDM autoencoder (Rombach et al.).

Rombach et al. (CVPR 2022) scale the first-stage latent to unit variance with
``s = 1 / std(z)``, measured over encoder outputs on the first training batch.
The stage-2 ULF->HF arm hand-set ``s = 2.45`` from stage 1's logged
``latent_std``, but that metric is ``exp(0.5 * logvar).mean()``, the posterior
sigma of q(z|x), not the spread of the encoder mean ``mu`` that
``encode_to_latent`` standardises. ``latent_scaling_factor: auto`` measures the
right quantity instead.

The statistic is all-reduced across ranks, so every rank standardises with the
same scale. Each rank must therefore reach the first calibrating encode in the
same step, which the training loop guarantees (every rank encodes its first
target batch at step 1).
"""

from __future__ import annotations

import torch

__all__ = ["AUTO_LATENT_SCALE", "global_latent_std", "parse_latent_scaling_factor"]

#: The YAML spelling that asks for calibration rather than a fixed scale.
AUTO_LATENT_SCALE = "auto"


def parse_latent_scaling_factor(raw: object) -> float | str | None:
    """``None`` (absent), ``"auto"``, or a positive float. Anything else raises."""
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.strip().lower() == AUTO_LATENT_SCALE:
            return AUTO_LATENT_SCALE
        raise ValueError(
            f"latent_scaling_factor={raw!r}: expected a positive number or 'auto' "
            "(calibrate 1/std(mu) on the first training batch)."
        )
    value = float(raw)  # type: ignore[arg-type]
    if value <= 0:
        raise ValueError(f"latent_scaling_factor must be > 0; got {value}.")
    return value


def global_latent_std(mu: torch.Tensor) -> torch.Tensor:
    """Scalar std of *mu* over every element, pooled across distributed ranks.

    Pooled from (sum, sum of squares, count) rather than averaging per-rank
    stds, so an uneven split still yields the std of the union. Returns a
    0-d tensor on ``mu``'s device; no host sync.
    """
    flat = mu.detach().to(torch.float64).reshape(-1)
    stats = torch.stack(
        [
            flat.sum(),
            flat.square().sum(),
            torch.tensor(float(flat.numel()), dtype=torch.float64, device=flat.device),
        ]
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats)
    total, total_sq, count = stats[0], stats[1], stats[2]
    mean = total / count
    var = (total_sq / count - mean.square()).clamp_min(0.0)
    return var.sqrt().to(mu.dtype)
