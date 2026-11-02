r"""Schramm-Loewner Evolution (SLE) :math:`{}_\kappa` acquisition trajectories.

The chordal SLE :math:`{}_\kappa` curve in the upper half-plane
:math:`\mathbb{H}` is the random simple curve driven by the scaled Brownian
motion :math:`\sqrt{\kappa}B_t` through the Loewner equation

.. math::

    \partial_t g_t(z) = \frac{2}{g_t(z) - \sqrt{\kappa}B_t}.

By Schramm's theorem the law of the curve is conformally invariant; by
Beffara the Hausdorff dimension is :math:`\min(1+\kappa/8, 2)`. Sampling
k-space along an SLE :math:`{}_\kappa` curve yields fractal, conformally
invariant masks with a Talagrand-chaining RIP bound
:math:`\delta=O(s^{1-d/2}\sqrt{\log N/m})`.

This module implements the *discrete slit-mapping* construction
(Marshall-Rohde, Kennedy) which approximates the curve to first order in
:math:`\Delta t` by composing backward vertical slit maps
:math:`f_j(z)=U_j+\sqrt{(z-U_j)^2-\varepsilon^2}` with
:math:`\varepsilon=2\sqrt{\Delta t}` (half-plane capacity :math:`2\Delta t`,
the normalisation of :math:`\partial_t g_t = 2/(g_t-U_t)`) and driver
:math:`U_j=\sqrt{\kappa}B_{t_j}`. The tip at step :math:`k` is
:math:`f_0\circ\cdots\circ f_{k-1}(U_k+i\varepsilon)`, the square root taken
on the branch with positive imaginary part. The trace is then rasterised onto
the k-space grid as a binary acquisition mask.

Consistency check: in the deterministic limit :math:`\kappa\to 0` the
driver vanishes and the slit-map composition produces a straight vertical
segment — Cartesian-line acquisition.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "build_sle_budget_mask",
    "build_sle_kspace_mask",
    "kappa_to_dimension",
    "sample_sle_trace",
    "sample_sle_traces",
]

#: Traces drawn per batch while a point budget is being filled.
_BUDGET_BATCH = 16


def kappa_to_dimension(kappa: float) -> float:
    r"""Hausdorff dimension :math:`\dim_H(\gamma)=\min(1+\kappa/8, 2)` (Beffara 2008)."""
    if kappa < 0.0:
        raise ValueError(f"kappa must be non-negative; got {kappa}.")
    return min(1.0 + kappa / 8.0, 2.0)


def sample_sle_traces(
    kappa: float,
    n_steps: int,
    n_traces: int,
    *,
    T: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Sample ``n_traces`` independent SLE :math:`{}_\kappa` traces in :math:`\mathbb{H}`.

    Returns:
        ``(n_traces, n_steps + 1, 2)`` float tensor of ``(x, y)`` tip coordinates;
        row 0 is the origin and ``y >= 0`` throughout.
    """
    if kappa < 0.0 or kappa >= 8.0:
        raise ValueError(f"kappa must lie in [0, 8); got {kappa}.")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")
    if n_traces < 1:
        raise ValueError("n_traces must be >= 1.")

    dt = T / n_steps
    eps = 2.0 * math.sqrt(dt)
    increments = torch.randn(n_traces, n_steps, generator=generator, device=device)
    drivers = (math.sqrt(kappa * dt) * torch.cumsum(increments.double(), dim=1)).to(
        torch.complex128
    )

    # Every tip starts as its own slit tip and is pulled back through the earlier
    # maps; the recurrence is sequential in j but batched over traces and tips.
    z = drivers + 1j * eps
    for j in range(n_steps - 2, -1, -1):
        u = drivers[:, j : j + 1]
        w = torch.sqrt((z[:, j + 1 :] - u) ** 2 - eps * eps)
        z[:, j + 1 :] = u + torch.where(w.imag < 0, -w, w)

    trace = torch.zeros((n_traces, n_steps + 1, 2), device=z.device)
    trace[:, 1:, 0] = z.real.float()
    trace[:, 1:, 1] = z.imag.float()
    return trace


def sample_sle_trace(
    kappa: float,
    n_steps: int,
    *,
    T: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Sample one realisation of an SLE :math:`{}_\kappa` trace in :math:`\mathbb{H}`.

    Args:
        kappa: SLE parameter in :math:`[0,8)`; controls fractal dimension.
        n_steps: Number of discretisation steps in time.
        T: Final time. Curve length scales as :math:`\sqrt{T}`.
        generator: Optional :class:`torch.Generator` for reproducibility.
        device: Device on which to allocate.

    Returns:
        Tensor of shape ``(n_steps + 1, 2)`` giving ``(x, y)`` Cartesian
        coordinates of the trace tip at each time. ``y >= 0`` always.
    """
    return sample_sle_traces(kappa, n_steps, 1, T=T, generator=generator, device=device)[0]


def _raster_indices(
    traces: torch.Tensor, shape: tuple[int, int], flips: torch.Tensor | None = None
) -> torch.Tensor:
    """Linear grid index of every trace point, each trace scaled to fill the grid.

    ``flips`` is an optional ``(n_traces, 2)`` bool mirroring a trace's ``(x, y)``.
    """
    h, w = shape
    lo = traces.amin(dim=1, keepdim=True)
    span = (traces.amax(dim=1, keepdim=True) - lo).clamp_min(1e-6)
    unit = (traces - lo) / span
    if flips is not None:
        unit = torch.where(flips.to(unit.device).unsqueeze(1), 1.0 - unit, unit)
    x_idx = (unit[..., 0] * (w - 1)).long().clamp(0, w - 1)
    y_idx = (unit[..., 1] * (h - 1)).long().clamp(0, h - 1)
    return y_idx * w + x_idx


def build_sle_kspace_mask(
    kappa: float,
    shape: tuple[int, int],
    n_steps: int,
    *,
    T: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Rasterise an SLE :math:`{}_\kappa` trace into a binary k-space mask.

    The trace is centred and scaled into the image grid; lattice points
    visited by the discretised trace receive value 1, others 0. The mask
    always contains the DC sample at the grid centre.

    Args:
        kappa: SLE parameter in :math:`[0,8)`.
        shape: ``(H, W)`` of the k-space grid.
        n_steps: Number of trace points to draw (recommend :math:`O(\sqrt{HW})`).
        T: Loewner final time.
        generator: Optional :class:`torch.Generator`.
        device: Device on which to allocate the mask.

    Returns:
        Binary mask of dtype ``torch.bool`` and shape ``shape``.
    """
    h, w = shape
    trace = sample_sle_trace(kappa=kappa, n_steps=n_steps, T=T, generator=generator, device=device)
    mask = torch.zeros(h * w, dtype=torch.bool, device=trace.device)
    mask[_raster_indices(trace.unsqueeze(0), shape).reshape(-1)] = True
    mask = mask.view(h, w)
    mask[h // 2, w // 2] = True
    return mask


def build_sle_budget_mask(
    kappa: float,
    shape: tuple[int, int],
    n_points: int,
    *,
    allowed: torch.Tensor | None = None,
    n_steps: int | None = None,
    T: float = 1.0,
    generator: torch.Generator | None = None,
    max_traces: int = 4096,
) -> tuple[torch.Tensor, int]:
    r"""Exactly ``n_points`` grid bins visited by independent SLE :math:`{}_\kappa` shots.

    One trace of dimension :math:`1+\kappa/8` covers far fewer bins than a 4x
    budget, so shots are drawn until the union of their bins inside ``allowed``
    reaches ``n_points``; the last shot is truncated along its own curve order.
    Each shot is mirrored along each axis with probability 1/2, so the expected
    sample density is symmetric about the grid centre (DC).

    Args:
        kappa: SLE parameter in :math:`[0,8)`.
        shape: ``(H, W)`` of the k-space grid.
        n_points: Number of bins to acquire.
        allowed: Optional ``(H, W)`` bool; only these bins may be acquired.
        n_steps: Steps per shot; defaults to ``2 * max(H, W)``.
        T: Loewner final time.
        generator: Optional :class:`torch.Generator`.
        max_traces: Shot cap; reaching it raises instead of under-filling.

    Returns:
        ``(mask, n_traces)``: an ``(H, W)`` bool mask with exactly ``n_points``
        ones, and the number of shots it used.
    """
    h, w = shape
    eligible = torch.ones(h * w, dtype=torch.bool) if allowed is None else allowed.reshape(-1)
    if n_points > int(eligible.sum()):
        raise ValueError(f"n_points={n_points} exceeds the {int(eligible.sum())} allowed bins.")
    steps = 2 * max(h, w) if n_steps is None else n_steps
    mask = torch.zeros(h * w, dtype=torch.bool)
    used = 0
    while int(mask.sum()) < n_points:
        if used >= max_traces:
            raise RuntimeError(
                f"SLE budget mask reached {int(mask.sum())}/{n_points} bins after "
                f"{max_traces} shots; raise max_traces or lower the budget."
            )
        traces = sample_sle_traces(kappa, steps, _BUDGET_BATCH, T=T, generator=generator)
        # Every trace is rooted at y=0, so unmirrored shots pile density onto one edge of
        # the grid; a random reflection per shot and axis centres the density on DC.
        flips = torch.rand(_BUDGET_BATCH, 2, generator=generator) < 0.5
        lin = _raster_indices(traces, shape, flips).reshape(-1).cpu()
        shot = torch.arange(lin.numel()) // (steps + 1)
        keep = eligible[lin] & ~mask[lin]
        lin, shot = lin[keep], shot[keep]
        uniq, inverse = torch.unique(lin, return_inverse=True)
        first = torch.full((uniq.numel(),), lin.numel(), dtype=torch.long)
        first.scatter_reduce_(0, inverse, torch.arange(lin.numel()), reduce="amin")
        taken = first.sort().values[: n_points - int(mask.sum())]
        mask[lin[taken]] = True
        filled = int(mask.sum()) >= n_points and taken.numel() > 0
        used += int(shot[taken[-1]]) + 1 if filled else _BUDGET_BATCH
    return mask.view(h, w), used
