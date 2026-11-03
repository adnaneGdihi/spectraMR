r"""Monotone-remap invariance: how much a model's output moves under an intensity remap.

For a model :math:`G` and its input :math:`\mathbf x`, draw strictly increasing
intensity maps :math:`g_k` and report the mean relative change

.. math::

   s = \frac{1}{K}\sum_{k=1}^{K}
       \frac{\lVert G(g_k\circ\mathbf x) - G(\mathbf x)\rVert_2}
            {\lVert G(\mathbf x)\rVert_2},

averaged over the batch. A model fronted by an exact rank transform (MCGI) scores
exactly 0; a model on raw intensities does not. Lower is better.

Each :math:`g_k` is a piecewise-linear map of the per-image min-max normalised
intensity, slopes in ``[0.5, 2]``, quantised onto an integer grid of :math:`2^m`
levels and scaled by a power of two near the image's range. A map evaluated in
floating point can merge two adjacent intensities, which changes the ties a rank
transform sees; here each distinct value also gets its dense rank added to its grid
level, and the grid stays below :math:`2^{24}`, so the remapped intensities are
exact float32 integers times a power of two, strictly increasing over distinct
values and equal over tied ones. The maps are drawn from a fixed seed, so two arms
scored on the same input see the same remaps.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.registry import register_metric

__all__ = ["MonotoneRemapInvariance", "random_monotone_remap"]

_KNOTS = 8
_EXACT_INT_BITS = 24  # float32 holds every integer below 2**24 exactly


def random_monotone_remap(x: Tensor, generator: torch.Generator) -> Tensor:
    """A strictly increasing intensity map per ``(batch, channel)`` image of ``x``.

    Args:
        x: Real float32 or float64 tensor ``[B, C, *spatial]``.
        generator: CPU generator the map parameters are drawn from.

    Returns:
        Tensor of ``x``'s shape and dtype; ties in ``x`` stay ties and distinct
        values stay distinct and keep their order.
    """
    if x.is_complex():
        raise ValueError("A monotone intensity remap is undefined on a complex input.")
    if x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"The exact remap needs float32 or float64 input; got {x.dtype}.")
    b, c = x.shape[0], x.shape[1]
    flat = x.reshape(b, c, -1).to(torch.float64)
    n = flat.shape[-1]
    # Grid levels plus at most n dense-rank steps must stay below 2**24 (h(1) <= 2).
    level_bits = ((2**_EXACT_INT_BITS - n - 1) // 2).bit_length() - 1
    if level_bits < 8:
        raise ValueError(f"An image of {n} voxels is too large for the exact float32 remap.")
    srt = flat.sort(dim=-1).values
    lo, span = srt[..., :1], srt[..., -1:] - srt[..., :1]
    safe_span = torch.where(span > 0, span, torch.ones_like(span))
    u = (srt - lo) / safe_span
    slopes = 0.5 + 1.5 * torch.rand(b, c, _KNOTS, generator=generator, dtype=torch.float64)
    octave = torch.randint(0, 2, (b, c, 1), generator=generator).to(torch.float64)
    slopes, octave = slopes.to(flat.device), octave.to(flat.device)
    width = 1.0 / _KNOTS
    base = torch.cumsum(slopes * width, dim=-1) - slopes * width
    idx = (u * _KNOTS).floor().clamp(0, _KNOTS - 1).long()
    h = base.gather(-1, idx) + slopes.gather(-1, idx) * (u - idx.to(torch.float64) * width)
    # cummax absorbs float wobble at the knots, so levels never decrease along the sort.
    levels = torch.cummax(torch.round(h * 2.0**level_bits), dim=-1).values
    is_new = torch.cat([torch.zeros_like(srt[..., :1]), (srt.diff(dim=-1) > 0).double()], -1)
    grid_sorted = levels + is_new.cumsum(dim=-1)
    grid = grid_sorted.gather(-1, torch.searchsorted(srt, flat))
    exponent = torch.round(torch.log2(safe_span)) + octave - level_bits
    return (grid * torch.exp2(exponent)).to(x.dtype).reshape_as(x)


@register_metric(
    "monotone_remap_invariance",
    requires_reference=False,
    needs=("reconstructor", "model_input"),
    direction="lower",
)
class MonotoneRemapInvariance:
    """Relative output change under random strictly increasing intensity remaps.

    Args:
        num_remaps: ``K`` remaps per call (each costs one extra forward).
        seed: Seed of the remap draw.
    """

    name = "monotone_remap_invariance"
    higher_is_better = False

    def __init__(self, num_remaps: int = 2, seed: int = 0) -> None:
        if num_remaps < 1:
            raise ValueError(f"num_remaps must be >= 1; got {num_remaps}.")
        self.num_remaps = int(num_remaps)
        self.seed = int(seed)

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del prediction, target
        ctx = resolve_context(context, kwargs)
        ctx.require(self.name, "reconstructor", "model_input")
        forward, x = ctx.reconstructor, ctx.model_input
        assert forward is not None and x is not None
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        with torch.no_grad():
            base = forward(x)
            dims = tuple(range(1, base.ndim))
            norm = torch.linalg.vector_norm(base, dim=dims)
            norm = norm.clamp(min=torch.finfo(norm.dtype).tiny)
            total = torch.zeros_like(norm)
            for _ in range(self.num_remaps):
                moved = forward(random_monotone_remap(x, generator))
                total = total + torch.linalg.vector_norm(moved - base, dim=dims) / norm
        return (total / self.num_remaps).mean()
