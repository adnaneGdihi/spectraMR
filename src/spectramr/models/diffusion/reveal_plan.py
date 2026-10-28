"""The reveal schedule of a freeze-family reverse trajectory, computed before the loop.

Which coefficients step ``i`` writes depends on the keying masks, the observed
support and what earlier steps committed, never on the model's output. The whole
schedule is therefore known before the first model call. Computing it up front
lets the loop decide which steps are inert with one host read per trajectory
instead of one ``bool(reveal.any())`` per step (#1162, #1388), and builds each
keying mask once instead of twice on every step that runs.

The arithmetic stays in ``PhysicsInformedColdDiffusion._reveal_support``; this
module only walks it over the schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class RevealPlan:
    """Per-coefficient step of first write, and which steps write anything.

    Attributes:
        first: ``int16`` tensor shaped like a reveal; the step that writes each
            coefficient, ``-1`` for one no step writes (it was committed or
            observed before the loop).
        active: ``active[i]`` is True when step ``i`` writes at least one
            coefficient.
    """

    first: Tensor
    active: tuple[bool, ...]

    def reveal(self, i: int, dtype: torch.dtype) -> Tensor:
        """The coefficients step ``i`` writes, as a 0/1 tensor of ``dtype``."""
        return (self.first == i).to(dtype)

    def first_active(self) -> int | None:
        """Index of the first step that writes anything, or None."""
        return next((i for i, hit in enumerate(self.active) if hit), None)


def plan_reveals(
    sampler: Any,
    timestep_schedule: list[int],
    committed: Tensor,
    obs: Tensor,
    x: Tensor,
    batch: int,
    device: torch.device,
) -> RevealPlan:
    """Walk ``sampler._reveal_support`` over the schedule without reading the host per step.

    ``committed`` and ``obs`` are the loop's own starting values; the loop must
    advance ``committed`` by exactly the reveals this returns, which it does by
    reading them from the plan. The one host transfer is the final ``tolist``.
    """
    if len(timestep_schedule) >= torch.iinfo(torch.int16).max:
        raise ValueError(
            f"a {len(timestep_schedule)}-step schedule does not fit the int16 reveal plan"
        )
    first: Tensor | None = None
    hits: list[Tensor] = []
    for i in range(len(timestep_schedule)):
        reveal = sampler._reveal_support(i, timestep_schedule, committed, obs, x, batch, device)
        written = reveal > 0
        if first is None:
            first = torch.full(written.shape, -1, dtype=torch.int16, device=written.device)
        first.masked_fill_(written, i)
        hits.append(written.any())
        committed = torch.clamp(committed + reveal, max=1.0)
    if first is None:
        raise ValueError("cannot plan the reveals of an empty timestep schedule")
    active = tuple(bool(v) for v in torch.stack(hits).tolist())
    return RevealPlan(first=first, active=active)
