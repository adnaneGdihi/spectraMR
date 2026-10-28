"""spin_history: through-plane motion carries magnetisation history between slices.

A spin that moves from slice j to slice k at the start of a shot was last excited by j's pulse
``dt'`` before its first excitation in k, counting only the excitations of j before the hop
(``intervals_at`` with ``cut``). Spoiled-GRE steady state before an excitation is
``M_ss = (1 - E1) / (1 - E1 cos a)``, ``E1 = exp(-TR/T1)`` (Bernstein, Handbook of MRI Pulse
Sequences, 2004, sec. 2.3). The moved spin has ``Mz+ = M_ss cos a`` after its last excitation, so at
its first new excitation (n = 0)

    M_0 = 1 - (1 - M_ss cos a) exp(-dt'/T1),

and it relaxes back with every further excitation, ``M_n = M_ss + (M_0 - M_ss)(E1 cos a)^n``
(closed form of the recursion ``M_{n+1} = 1 + (M_n cos a - 1) E1``). The signal gain over a static
spin is ``G_n = M_n / M_ss = 1 + a q^n`` with ``a = M_0/M_ss - 1``, ``q = E1 cos a``; ``n`` is the
number of excitations of the spin's slice since the hop. A source outside the stack or slab was
never excited: ``dt' -> inf``, ``M_0 = 1`` (inflow). A spin that stays in its slice (o = 0) keeps
the steady state (a = 0).

The displacement is one rigid through-plane shift per shot (8 shots of consecutive acquisition
time, the rigid-motion dz path), peak ``5 mm * theta``; each shot is treated as a hop from steady
state at its first line (consecutive hops are not chained). Multislice: slice k receives its two
neighbouring source slices, each with its own ``a``. 3-D: the slab is excited as a whole, so only
the voxels whose source lies outside the slab (a ``slab_coverage`` strip) have ``a != 0``, and ``n``
counts the shot's lines.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import (
    line_times,
    shot_index,
)
from spectramr.infrastructure.physics.degradations.blocks.time_states import assemble_from_states
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.degradations.catalogue.multislice_common import (
    intervals_at,
    positive,
    slab_coverage,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import check_volume
from spectramr.infrastructure.physics.degradations.volume.motion_schedules import MOTION_TR_S
from spectramr.infrastructure.physics.fft_ops import ifftnc
from spectramr.infrastructure.physics.per_sample_rng import per_sample_rand

__all__ = [
    "DEFAULT_FA_DEG",
    "DEFAULT_T1_MS",
    "DEFAULT_TR_MS",
    "N_SHOTS",
    "PEAK_DZ_MM",
    "excess_gain",
    "history_multislice",
    "history_volume",
    "lines_since_hop",
    "mz_after_n",
    "shot_times",
    "spin_history",
    "steady_state",
]

DEFAULT_FA_DEG = 30.0
DEFAULT_TR_MS = 50.0
DEFAULT_T1_MS = 1000.0
PEAK_DZ_MM = 5.0
N_SHOTS = 8  # the rigid-motion pose count


def steady_state(tr_s: float, t1_s: float, alpha: float) -> tuple[float, float]:
    """``(M_ss, q)``: the pre-excitation steady state and the per-excitation decay ``E1 cos a``."""
    e1 = math.exp(-tr_s / t1_s)
    return (1.0 - e1) / (1.0 - e1 * math.cos(alpha)), e1 * math.cos(alpha)


def mz_after_n(dt_s: Tensor, n: Tensor, tr_s: float, t1_s: float, fa_deg: float) -> Tensor:
    """``M_n = M_ss + (M_0 - M_ss) q^n`` for a hop with interval ``dt_s`` (``inf`` allowed)."""
    alpha = math.radians(fa_deg)
    m_ss, q = steady_state(tr_s, t1_s, alpha)
    m0 = 1.0 - (1.0 - m_ss * math.cos(alpha)) * torch.exp(-dt_s / t1_s)
    return m_ss + (m0 - m_ss) * q ** n.to(dt_s.dtype)


def excess_gain(dt_s: Tensor, tr_s: float, t1_s: float, fa_deg: float) -> Tensor:
    """``a = M_0 / M_ss - 1``: the first-excitation gain excess of a hop with interval ``dt_s``."""
    m_ss, _ = steady_state(tr_s, t1_s, math.radians(fa_deg))
    return mz_after_n(dt_s, torch.zeros(()), tr_s, t1_s, fa_deg) / m_ss - 1.0


def lines_since_hop(times: Tensor, shots: Tensor, per_slice: bool) -> Tensor:
    """Excitations ``n`` of each ``[D, H]`` line's spins since its shot began.

    Multislice counts a slice's own earlier lines of the same shot; 3-D counts every earlier line
    of the shot (the whole slab is excited each TR).
    """
    d, h = times.shape
    if per_slice:
        before = (times[:, None, :] < times[:, :, None]) & (shots[:, None, :] == shots[:, :, None])
        return before.sum(-1)
    order = torch.empty(d * h, dtype=torch.long)
    order[torch.argsort(times.flatten(), stable=True)] = torch.arange(d * h)
    flat = shots.flatten()
    start = torch.full((int(flat.max()) + 1,), d * h).scatter_reduce(0, flat, order, "amin")
    return (order - start[flat]).view(d, h)


def shot_times(times: Tensor, shots: Tensor) -> tuple[Tensor, Tensor]:
    """``(first, start)``: each slice's first line time per shot ``[D, S]`` (0 where it has none,
    never read) and each shot's first line time ``[S]`` (the instant the spin hops)."""
    hit = shots[:, None, :] == torch.arange(int(shots.max()) + 1)[None, :, None]
    per_slice = torch.where(hit, times[:, None, :], torch.full_like(times[:, None, :], math.inf))
    first = per_slice.amin(-1)
    return torch.where(torch.isfinite(first), first, torch.zeros_like(first)), first.amin(0)


def history_multislice(x: Tensor, dz: Tensor, table: Tensor, max_hop: int) -> Tensor:
    """The stack moved by ``dz`` slices (``[B]``, source = k - dz), source slice o weighted by table.

    ``table[o + max_hop, k]`` weights a spin whose source is slice ``k + o``; sources outside the
    stack are replicated from the edge slice.
    """
    b, _, d = x.shape[:3]
    m0 = torch.floor(dz)
    f = (dz - m0).view(b, 1, 1, 1, 1).to(x.real.dtype)
    m0 = m0.long()
    k = torch.arange(d, device=x.device)

    def part(o: Tensor) -> Tensor:
        src = (k[None] + o[:, None]).clamp(0, d - 1)
        xs = x.gather(2, src.view(b, 1, d, 1, 1).expand_as(x))
        gain = table[(o + max_hop)[:, None], k[None]].to(x.real.dtype)
        return xs * gain.view(b, 1, d, 1, 1)

    return (1.0 - f) * part(-m0) + f * part(-m0 - 1)


def history_volume(x: Tensor, dz: Tensor, excess: float | None) -> Tensor:
    """The slab content moved by ``dz`` voxels (``[B]``); ``excess`` weights the voxels whose source
    lies outside the slab (``None``: the plain shifted slab)."""
    b, _, d = x.shape[:3]
    shift = dz.view(b, 1, 1, 1, 1).to(x.real.dtype)
    disp = torch.zeros(b, 3, *x.shape[-3:], device=x.device, dtype=x.real.dtype)
    disp[:, 0] = -shift.view(b, 1, 1, 1)
    moved = warp_volume(x, disp)
    return moved if excess is None else moved * (excess * (1.0 - slab_coverage(shift, d)))


def spin_history(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    fa_deg: float = DEFAULT_FA_DEG,
    tr_ms: float = DEFAULT_TR_MS,
    t1_ms: float = DEFAULT_T1_MS,
) -> Tensor:
    """Per-shot through-plane displacement with the magnetisation history it carries."""
    check_volume(x, geometry, "spin_history")
    fa_deg = positive("spin_history", "fa_deg", fa_deg)
    if fa_deg > 180.0:
        raise ValueError(f"spin_history: fa_deg must be in (0, 180], got {fa_deg}")
    tr_s = positive("spin_history", "tr_ms", tr_ms) / 1000.0
    t1_s = positive("spin_history", "t1_ms", t1_ms) / 1000.0
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    b, _, d, h, _ = x.shape
    # Unknown voxel size: 1 mm voxels, as the other mm-declared motion axes assume.
    pitch = geometry.voxel_mm[0] if geometry.voxel_mm is not None else 1.0
    shots = shot_index(line_times(geometry, d, h, tr_s=MOTION_TR_S), min(N_SHOTS, d * h))
    n_shots = int(shots.max()) + 1
    u = per_sample_rand(b, n_shots, generator=_generator(x.device, seed), device=x.device)
    dz = (2.0 * u - 1.0) * (PEAK_DZ_MM * float(theta) / pitch)
    multislice = geometry.mode is AcquisitionMode.MULTISLICE_2D
    times = line_times(geometry, d, h, tr_s=tr_s).double()
    n_line = lines_since_hop(times, shots, multislice)
    _, q = steady_state(tr_s, t1_s, math.radians(fa_deg))
    qn = (q ** n_line.double()).to(device=x.device, dtype=x.real.dtype).view(1, 1, d, h, 1)
    if multislice:
        max_hop = math.ceil(PEAK_DZ_MM / pitch) + 1
        offsets = range(-max_hop, max_hop + 1)
        first, start = shot_times(times, shots)
        dt = intervals_at(geometry, d, h, tr_s, offsets, first, cut=start)
        excess = excess_gain(dt, tr_s, t1_s, fa_deg)
        excess[max_hop] = 0.0  # a spin that stays in its slice keeps the steady state
        excess = excess.to(x.device)
        ones = torch.ones_like(excess[..., 0])

        def render(p: int, weights: Tensor) -> Tensor:
            return history_multislice(x, dz[:, p], weights, max_hop)

        k_a = assemble_from_states(n_shots, shots, lambda p: render(p, ones), encoding="slice")
        k_b = assemble_from_states(
            n_shots, shots, lambda p: render(p, excess[..., p]), encoding="slice"
        )
        return ifftnc(k_a + qn * k_b, dims=(-2, -1))
    inflow = float(excess_gain(torch.tensor(math.inf, dtype=torch.float64), tr_s, t1_s, fa_deg))
    k_a = assemble_from_states(
        n_shots, shots, lambda p: history_volume(x, dz[:, p], None), encoding="volume"
    )
    k_b = assemble_from_states(
        n_shots, shots, lambda p: history_volume(x, dz[:, p], inflow), encoding="volume"
    )
    return ifftnc(k_a + qn * k_b, dims=(-3, -2, -1))
