"""slice_crosstalk: a neighbouring slice's RF pulse partly saturates the spins of this slice.

The real slice profile is a rect of width ``thk`` convolved with a Gaussian of std
``sigma = theta * thk / 2`` (finite-bandwidth pulse), so the tails reach the adjacent slab when
the gap is small and the effect grows continuously from zero. A spin of slice k in the overlap was
saturated by the neighbour's pulse ``dt`` earlier and has recovered by ``1 - exp(-dt/T1)`` of it
(Bloch T1 recovery; multislice crosstalk, Haacke et al., MRI: Physical Principles and
Sequence Design, 1999):

    signal_k = x_k * prod over the adjacent slices j of (1 - s * exp(-dt_{j->k} / T1)),

with ``s`` the saturation fraction: the mean over slab k of the neighbour's unit-height profile,
``s = (1/t) int_{-t/2}^{t/2} R(P - u) du`` (``P`` the centre distance, ``t`` the thickness,
``R`` :func:`profile_response`). In closed form, with ``a = sqrt2 sigma`` and
``F(x) = x erf(x) + exp(-x^2)/sqrt(pi)`` (the second antiderivative of erf):

    s = a / (2 t) * [F((P + t)/a) - 2 F(P/a) + F((P - t)/a)].

``dt`` comes from the slice order and TR (``hop_intervals``).
SEQUENTIAL is block-by-block here (see ``line_times``): the neighbours are a whole slice
acquisition apart and recover almost fully, so it saturates far less than INTERLEAVED, where
the neighbours fire a fraction of TR apart.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex
from spectramr.infrastructure.physics.degradations.catalogue.multislice_common import (
    SIGMA_PER_THICKNESS,
    hop_intervals,
    positive,
    thickness_over_pitch,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import check_volume

__all__ = ["DEFAULT_T1_MS", "DEFAULT_TR_MS", "neighbour_saturation", "slice_crosstalk"]

DEFAULT_TR_MS = 500.0
DEFAULT_T1_MS = 1000.0


def _f(x: float) -> float:
    return x * math.erf(x) + math.exp(-x * x) / math.sqrt(math.pi)


def neighbour_saturation(thickness: float, sigma: float) -> float:
    """Mean over a slab of ``thickness`` of the next slice's profile (centre distance 1, all in
    pitches; ``sigma`` > 0), by the closed form in the module docstring."""
    a = math.sqrt(2.0) * sigma
    p = 1.0
    return (
        a
        / (2.0 * thickness)
        * (_f((p + thickness) / a) - 2.0 * _f(p / a) + _f((p - thickness) / a))
    )


def slice_crosstalk(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    tr_ms: float = DEFAULT_TR_MS,
    t1_ms: float = DEFAULT_T1_MS,
) -> Tensor:
    """Scale each slice by the saturation its adjacent slices left in it."""
    check_volume(x, geometry, "slice_crosstalk")
    if geometry.mode is not AcquisitionMode.MULTISLICE_2D:
        raise ValueError(f"slice_crosstalk is a multislice axis; got {geometry.mode!r}")
    tr_ms = positive("slice_crosstalk", "tr_ms", tr_ms)
    t1_ms = positive("slice_crosstalk", "t1_ms", t1_ms)
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    thk = thickness_over_pitch(geometry, "slice_crosstalk")
    s = neighbour_saturation(thk, float(theta) * SIGMA_PER_THICKNESS * thk)
    d, h = x.shape[-3], x.shape[-2]
    dt = hop_intervals(geometry, d, h, tr_ms / 1000.0, (-1, 1))
    remaining = torch.exp(-dt / (t1_ms / 1000.0))
    gain = (1.0 - s * remaining).prod(dim=0).to(device=x.device, dtype=x.real.dtype)
    return x * gain.view(1, 1, d, 1, 1)
