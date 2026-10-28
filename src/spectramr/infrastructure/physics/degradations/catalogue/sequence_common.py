"""What the sequence-timing catalogue axes share: line order from acquisition timing, and a k-space line filter.

A 2-D slice and each slice of a multislice stack is one 2-D k-space, so its lines are timed as a
one-partition volume (``D = 1``); a 3-D volume uses its own geometry's line order.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    LineOrder,
)
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc

__all__ = ["ORDERS", "apply_line_weights", "check_positive", "ordered_line_times"]

#: The ``order`` kwarg's options.
ORDERS = {"linear": LineOrder.LINEAR, "centric": LineOrder.CENTRIC}
_DIMS = {
    AcquisitionMode.SLICE_2D: (-2, -1),
    AcquisitionMode.MULTISLICE_2D: (-2, -1),
    AcquisitionMode.VOLUME_3D: (-3, -2, -1),
}


def check_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return float(value)


def ordered_line_times(
    geometry: AcquisitionGeometry, d: int, h: int, order: str | None
) -> tuple[Tensor, int]:
    """``([D', H] line times in TR units, inner-loop length)`` of the acquisition.

    ``D' = D`` in a volume and 1 in a slice / multislice (each slice is its own 2-D acquisition).
    ``order`` overrides the geometry's ``line_order`` (a 2-D default is ``linear``). The inner loop
    is ``H`` (ky) except for ``kz_inner``, where it is ``D``.
    """
    if order is not None and order not in ORDERS:
        raise ValueError(f"order must be one of {sorted(ORDERS)}, got {order!r}")
    volume = geometry.mode is AcquisitionMode.VOLUME_3D
    if volume:
        geom = geometry if order is None else replace(geometry, line_order=ORDERS[order])
    else:
        geom = AcquisitionGeometry(
            AcquisitionMode.VOLUME_3D, None, line_order=ORDERS[order or "linear"]
        )
    d_eff = d if volume else 1
    inner = d_eff if geom.line_order is LineOrder.KZ_INNER else h
    return line_times(geom, d_eff, h, tr_s=1.0), inner


def apply_line_weights(x: Tensor, weights: Tensor, mode: AcquisitionMode) -> Tensor:
    """Multiply k-space line ``(kz, ky)`` of every readout by ``weights[kz, ky]``.

    A slice / multislice k-space is 2-D (``weights`` is ``[1, H]``), a volume's is 3-D.
    """
    dims = _DIMS[mode]
    k = fftnc(x, dims) * weights.to(device=x.device, dtype=torch.float32).unsqueeze(-1)
    return ifftnc(k, dims)
