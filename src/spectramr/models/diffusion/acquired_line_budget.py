"""Count the acceleration R over the phase-encode lines a sample actually acquired.

M4Raw acquires 195 of 256 phase-encode lines (T1/T2) or 198 (FLAIR), and fastMRI
brain a per-volume share; the rest are exact zeros. A line mask that ranks all
256 lines spends part of every rung's budget on lines that hold nothing, so its
realised R over the acquired lines falls short of the declared one (#2160).

**One fixed ranking.** The full-support cascade is read once per plane shape:
``first_drop[l]`` is the first rung at which line ``l`` leaves the mask. Lines
are ranked by ``first_drop`` (latest first), then by distance to the centre line,
so the ACS core outranks the other lines that survive to the top rung. At rung
``t`` a sample keeps the ``round(kept_t * n_acquired / N)`` best-ranked lines of
its own acquired set. The ranking and each sample's support are fixed, so the
cascade stays nested; a full support reproduces the cascade bit for bit.

Everything per batch is tensor arithmetic on the mask device: no host read, and
nothing is cached per sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "LineRanking",
    "acquired_lines_of",
    "budget_lines",
    "check_declared_lines",
    "dynamic_lines",
    "lines_to_plane",
    "rank_lines",
]


@dataclass(frozen=True)
class LineRanking:
    """A full-support cascade, reduced to what the budget needs.

    ``order`` lists line indices best-first ``[N]``; ``kept`` is the number of
    lines each rung keeps at full support ``[T]``; ``core`` marks the ACS core,
    the contiguous run through the centre line at the top rung ``[N]``.
    """

    order: Tensor
    kept: Tensor
    core: Tensor

    @property
    def num_lines(self) -> int:
        return int(self.order.shape[0])


def rank_lines(cascade_lines: Tensor) -> LineRanking:
    """The fixed ranking of a nested ``[T, N]`` bool cascade of kept lines."""
    num_t, n = cascade_lines.shape
    device = cascade_lines.device
    first_drop = torch.full((n,), num_t, dtype=torch.long, device=device)
    for t in range(num_t):
        first_drop = torch.where(~cascade_lines[t] & (first_drop == num_t), t, first_drop)
    centre = n // 2
    index = torch.arange(n, device=device)
    distance = (index - centre).abs()
    order = torch.argsort(first_drop * (2 * n) + (2 * n - 1 - distance), descending=True)
    top = cascade_lines[-1]
    left = torch.cumprod(top[: centre + 1].flip(0).long(), 0).sum()
    right = torch.cumprod(top[centre:].long(), 0).sum()
    core = (index > centre - left) & (index < centre + right)
    return LineRanking(order=order, kept=cascade_lines.sum(-1), core=core)


def acquired_lines_of(kspace: Tensor, line_dim: int) -> Tensor:
    """``[B, N]`` bool: the lines along ``line_dim`` that carry energy, per sample.

    ``kspace`` is ``[B, ..., H, W]``. A never-acquired phase-encode line is exactly
    zero in every coil and slice (``pe_support``'s rule), so a zero sum marks it.
    """
    plane = kspace.detach().abs()
    if line_dim == -2:
        plane = plane.transpose(-1, -2)
    return plane.flatten(1, -2).sum(1) > 0


def budget_lines(ranking: LineRanking, timesteps: Tensor, acquired: Tensor) -> Tensor:
    """``[B, N]`` bool: each sample's kept lines at its rung, over acquired lines only.

    Args:
        ranking: the fixed ranking of this plane shape.
        timesteps: ``[B]`` rung per sample.
        acquired: ``[B, N]`` bool acquired lines per sample.
    """
    n = ranking.num_lines
    n_acquired = acquired.sum(-1)
    kept = ranking.kept.to(acquired.device)[timesteps.to(acquired.device).long()]
    budget = torch.round(kept * n_acquired / n).long()
    order = ranking.order.to(acquired.device)
    ranked = acquired[:, order]
    take = ranked & (torch.cumsum(ranked.long(), dim=-1) <= budget[:, None])
    return torch.zeros_like(acquired).scatter(1, order.expand_as(take), take)


def dynamic_lines(draw: Tensor, acquired: Tensor, core: Tensor, seed: int) -> Tensor:
    """One dynamic-mask sample's kept lines, ``[N]`` bool, over its acquired lines.

    The per-sample draw has no cascade to rank against, so its own lines rank
    first, the ACS core above them, and the rest follow a permutation seeded by the
    same per-sample seed. The budget is the draw's line count scaled to the
    acquired share, so the draw's R holds over acquired lines.
    """
    n = draw.shape[-1]
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shuffle = torch.randperm(n, generator=generator).to(draw.device)
    priority = core.long() * (4 * n) + draw.long() * (2 * n) + shuffle
    order = torch.argsort(priority, descending=True)
    budget = torch.round(draw.sum() * acquired.sum() / n).long()
    ranked = acquired[order]
    take = ranked & (torch.cumsum(ranked.long(), dim=0) <= budget)
    return torch.zeros_like(acquired).scatter(0, order, take)


def check_declared_lines(
    declared: int | dict[str, int] | str, acquired: Tensor, contrasts: list[str] | None
) -> dict[str, object]:
    """Compare ``data.acquisition.acquired_pe_lines`` with each sample's measured count.

    One host read. ``measured`` only records what was found; an int or a
    per-contrast mapping must match every sample, keyed by the contrast NAME the
    dataset publishes.

    Raises:
        ValueError: a count disagrees, or a per-contrast mapping meets a batch
            without contrast names or with a contrast it does not list.
    """
    counts = [int(c) for c in acquired.sum(-1).tolist()]
    if declared == "measured":
        return {"declared": "measured", "measured": sorted(set(counts))}
    if isinstance(declared, dict):
        if not contrasts:
            raise ValueError(
                "data.acquisition.acquired_pe_lines is declared per contrast, but the "
                "batch carries no 'contrast' names to key it by."
            )
        if len(counts) % len(contrasts):
            raise ValueError(
                f"{len(contrasts)} contrast names cannot label {len(counts)} batch rows."
            )
        names = [str(c) for c in contrasts for _ in range(len(counts) // len(contrasts))]
        unknown = sorted({n for n in names if n not in declared})
        if unknown:
            raise ValueError(
                f"data.acquisition.acquired_pe_lines lists {sorted(declared)} but the batch "
                f"holds contrast(s) {unknown}."
            )
        expected = [int(declared[n]) for n in names]
    else:
        names = [None] * len(counts)
        expected = [int(declared)] * len(counts)
    wrong = sorted({(n, e, c) for n, e, c in zip(names, expected, counts, strict=True) if e != c})
    if wrong:
        detail = "; ".join(f"{n or 'all'}: declared {e}, measured {c}" for n, e, c in wrong)
        raise ValueError(
            f"data.acquisition.acquired_pe_lines disagrees with the data ({detail}). "
            "Re-measure with scripts/physics/measure_phase_encode_axis.py --arm <yaml> (#2160)."
        )
    return {"declared": declared, "measured": sorted(set(counts))}


def lines_to_plane(lines: Tensor, shape: tuple[int, int], line_dim: int) -> Tensor:
    """``[B, N]`` kept lines as a ``[B, 1, H, W]`` float mask."""
    h, w = shape
    if line_dim == -1:
        plane = lines[:, None, None, :].expand(-1, 1, h, w)
    else:
        plane = lines[:, None, :, None].expand(-1, 1, h, w)
    return plane.float()
