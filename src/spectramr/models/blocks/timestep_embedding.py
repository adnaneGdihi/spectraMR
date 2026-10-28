"""Sinusoidal timestep embedding — canonical, parameter-free.

Why this module exists
----------------------
A diffusion backbone has to tell neighbouring timesteps apart. Three mistakes
recur in this tree and all are silent:

1. **Encoding ``t`` linearly.** ``nn.Linear(1, d)`` applied to the raw scalar is
   a rank-1 map of ``t``: every timestep code lies on a single line in embedding
   space, so the network can represent "large t" but not "which t". A sinusoidal
   basis spreads the codes over ``d`` dimensions at geometrically spaced
   frequencies, which is what lets a model condition on a specific step.

2. **Dividing ``t`` by the horizon before the sinusoid.** The frequency ladder
   ``10000^(-k/(d/2-1))`` is built for unit steps: its highest frequency is one
   radian per step. Dividing by ``max_timesteps`` compresses the whole schedule
   into one unit, so only the few highest frequencies move. At ``T = 29`` and
   ``d = 256``, ``t / T`` leaves adjacent rungs 0.094 apart (norm 11.3), the two
   ends at cosine 0.97, and 43 of 256 dimensions varying; integer ``t`` gives
   2.66, 0.57 and 135 (#2315). ``scale="integer"`` embeds the step index as is;
   ``"unit_interval"`` is the dividing form, kept so trained models keep the
   codes they were fitted to.

3. **Borrowing DDPM's horizon.** Stretching ``t`` by ``1000 / T`` reuses the
   step count DDPM's ladder was tuned on; the spread it buys comes from
   aliasing the fast channels, and it changes with ``T``. ``"horizon_band"``
   instead derives the ladder from the schedule itself: geometric frequencies
   from a quarter turn per step (half the per-step Nyquist rate, so adjacent
   rungs never alias) down to a quarter turn over the whole horizon (so the
   slowest channel is monotone from ``t = 0`` to ``t = T - 1``). At ``T = 29``
   and ``d = 256`` adjacent rungs sit 6.59 apart, 2.5x ``integer``, with 0.991
   of distance triples ordered by step gap (``integer`` 0.999); at ``T = 31``
   the spacing is 6.52.

Scope note
----------
Ten near-identical private copies of this function exist across
``models/generators/``. This module is deliberately introduced for NEW consumers
only — migrating the existing copies would need each one proven bit-identical
first, or it silently changes trained-model behaviour. Consolidating them is
tracked separately.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = [
    "VALID_TIMESTEP_SCALES",
    "sinusoidal_timestep_embedding",
    "validate_timestep_scale",
]


#: Base of the transformer ladder the two legacy scales keep, so trained
#: models keep the codes they were fitted to.
_LEGACY_BASE = 10000.0

#: A quarter turn: the fastest ``horizon_band`` channel's advance per step and
#: the slowest channel's advance over the whole horizon.
_QUARTER_TURN = math.pi / 2


def _positive(horizon: float) -> float:
    if horizon <= 0:
        raise ValueError(f"max_timesteps must be > 0, got {horizon}")
    return float(horizon)


def _legacy_ladder(t: torch.Tensor, half_dim: int) -> torch.Tensor:
    freqs = torch.exp(
        -math.log(_LEGACY_BASE)
        * torch.arange(half_dim, device=t.device, dtype=t.dtype)
        / max(half_dim - 1, 1)
    )
    return t[:, None] * freqs[None, :]


def _unit_interval(t: torch.Tensor, half_dim: int, horizon: float | None) -> torch.Tensor:
    return _legacy_ladder(t if horizon is None else t / _positive(horizon), half_dim)


def _integer(t: torch.Tensor, half_dim: int, horizon: float | None) -> torch.Tensor:
    return _legacy_ladder(t, half_dim)


def _horizon_band(t: torch.Tensor, half_dim: int, horizon: float | None) -> torch.Tensor:
    if horizon is None:
        raise ValueError(
            "timestep scale 'horizon_band' needs max_timesteps: the band is derived "
            "from the schedule's own horizon."
        )
    span = _positive(horizon) - 1.0
    if span < 1.0:
        raise ValueError(
            f"timestep scale 'horizon_band' needs at least two timesteps, got "
            f"max_timesteps={horizon}."
        )
    freqs = torch.exp(
        torch.linspace(
            math.log(_QUARTER_TURN),
            math.log(_QUARTER_TURN / span),
            half_dim,
            device=t.device,
            dtype=t.dtype,
        )
    )
    return t[:, None] * freqs[None, :]


_PHASE_MAPS = {
    "unit_interval": _unit_interval,
    "integer": _integer,
    "horizon_band": _horizon_band,
}

#: How scalar timesteps become sinusoid phases. ``unit_interval`` divides by the
#: horizon and ``integer`` keeps the step index, both on the legacy ladder;
#: ``horizon_band`` keeps the step index on a ladder derived from the horizon.
VALID_TIMESTEP_SCALES: frozenset[str] = frozenset(_PHASE_MAPS)


def validate_timestep_scale(scale: str) -> str:
    """Return ``scale`` if registered; raise otherwise (no silent default)."""
    if scale not in VALID_TIMESTEP_SCALES:
        raise ValueError(
            f"Unknown timestep embedding scale {scale!r}. "
            f"Valid: {sorted(VALID_TIMESTEP_SCALES)}."
        )
    return scale


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    *,
    max_timesteps: float | None = None,
    scale: str = "unit_interval",
) -> torch.Tensor:
    """Map a batch of scalar timesteps onto a sinusoidal basis.

    Args:
        timesteps: ``[B]`` (or ``[B, 1]``) scalar step indices. Floats are fine —
            a continuous acceleration factor is a legitimate conditioning signal
            here, not only an integer step.
        dim: Width of the returned embedding. Odd widths are right-padded.
        max_timesteps: The number of timesteps ``T``. ``unit_interval``
            divides by it (``None`` leaves the values untouched), ``integer``
            ignores it and ``horizon_band`` requires it, spanning ``T - 1`` steps.
        scale: ``"unit_interval"``, ``"integer"`` or ``"horizon_band"``; see
            :data:`VALID_TIMESTEP_SCALES`.

    Returns:
        ``[B, dim]``, ``[sin(...) ‖ cos(...)]``.

    Raises:
        ValueError: If ``dim < 2`` (``half_dim - 1`` would divide by zero) or
            ``max_timesteps`` is non-positive (both silent NaN sources
            otherwise), or ``horizon_band`` gets no horizon or fewer than two
            timesteps.
    """
    if dim < 2:
        raise ValueError(f"dim must be >= 2 for a sin/cos split, got {dim}")

    t = timesteps.float()
    if t.ndim > 1:
        t = t.reshape(t.shape[0], -1)[:, 0]
    args = _PHASE_MAPS[validate_timestep_scale(scale)](t, dim // 2, max_timesteps)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb
