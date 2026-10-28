"""The bins Algorithm F fits its decay over: the mask's lines, or shells of radial ``|k|``.

``reverse_decay`` fits a power law to the measured power of each bin and bounds
the unmeasured coefficients of each bin by it. On a Cartesian mask a bin is one
of the mask's lines -- a row at ``|ky|`` (dim -2) or a column at ``|kx|``
(dim -1), whichever axis the mask removes lines along (#2160) -- and is either
measured or not. A 2-D point mask (gaussian,
variable density, radial, spiral) measures some coefficients of every row,
mostly near ``kx = 0``, so the mean over a row's measured coefficients
overstates that row's energy and the row bound barely binds. A shell of
constant ``|k| = sqrt(kx^2 + ky^2)`` is sampled evenly by such a mask, so its
measured power is an unbiased estimate of the shell's.

The row axis reduces with the same tensor operations as the row-only fit it
generalizes, and the column axis is the same reduction on the other dimension,
so an arm whose mask removes rows reconstructs bit for bit as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


def row_index(height: int, device: torch.device) -> Tensor:
    """``|ky|`` of each row of a centred k-space plane, as float."""
    return (torch.arange(height, device=device) - height // 2).abs().float()


@dataclass(frozen=True)
class DecayAxis:
    """One binning of a centred ``H x W`` plane.

    ``coord`` is each bin's distance from the centre, ``[nbins]``. ``bins`` maps
    each coefficient to its bin, ``[H, W]``, and is ``None`` on a line axis,
    whose bins are the lines along ``line_dim`` themselves (-2 rows, -1 columns).
    """

    name: str
    coord: Tensor
    bins: Tensor | None
    line_dim: int = -2

    @property
    def _across(self) -> int:
        """The plane dimension a line axis sums over: the other one."""
        return -1 if self.line_dim == -2 else -2

    @property
    def num_bins(self) -> int:
        return int(self.coord.shape[0])

    def reduce_sum(self, t: Tensor) -> Tensor:
        """Sum ``t`` (``[B, C, H, W]``) over channels and over each bin: ``[B, nbins]``."""
        if self.bins is None:
            return t.sum(dim=(-3, self._across))
        flat = t.sum(dim=-3).flatten(1)
        index = self.bins.flatten().unsqueeze(0).expand_as(flat)
        return flat.new_zeros(flat.shape[0], self.num_bins).scatter_add_(1, index, flat)

    def reduce_max(self, t: Tensor) -> Tensor:
        """Maximum of a non-negative ``t`` over channels and each bin: ``[B, nbins]``."""
        if self.bins is None:
            return t.amax(dim=(-3, self._across))
        flat = t.amax(dim=-3).flatten(1)
        index = self.bins.flatten().unsqueeze(0).expand_as(flat)
        out = flat.new_zeros(flat.shape[0], self.num_bins)
        return out.scatter_reduce_(1, index, flat, reduce="amax", include_self=True)

    def spread(self, per_bin: Tensor) -> Tensor:
        """A ``[B, nbins]`` value on every coefficient of its bin, broadcastable to ``[B, 1, H, W]``."""
        if self.bins is None:
            if self.line_dim == -2:
                return per_bin[:, None, :, None]
            return per_bin[:, None, None, :]
        return per_bin[:, self.bins][:, None]

    def coord_map(self) -> Tensor:
        """Each coefficient's bin distance, broadcastable to ``[B, 1, H, W]``."""
        if self.bins is None:
            if self.line_dim == -2:
                return self.coord.view(1, 1, -1, 1)
            return self.coord.view(1, 1, 1, -1)
        return self.coord[self.bins].view(1, 1, *self.bins.shape)


def row_axis(height: int, width: int, device: torch.device) -> DecayAxis:
    """One bin per phase-encode row, at distance ``|ky|``."""
    del width
    return DecayAxis(name="row", coord=row_index(height, device), bins=None)


def column_axis(height: int, width: int, device: torch.device) -> DecayAxis:
    """One bin per column, at distance ``|kx|``: the line axis of a column mask."""
    del height
    return DecayAxis(name="column", coord=row_index(width, device), bins=None, line_dim=-1)


def radial_axis(height: int, width: int, device: torch.device) -> DecayAxis:
    """One bin per unit shell of ``|k|``: coefficient ``(ky, kx)`` falls in ``round(|k|)``."""
    ky = row_index(height, device).view(-1, 1)
    kx = row_index(width, device).view(1, -1)
    bins = torch.sqrt(ky * ky + kx * kx).round().long()
    coord = torch.arange(int(bins.max()) + 1, device=device, dtype=torch.float32)
    return DecayAxis(name="radial", coord=coord, bins=bins)


#: Fit axis name -> builder, the one place an axis is selected by name.
DECAY_AXES: dict[str, Callable[[int, int, torch.device], DecayAxis]] = {
    "row": row_axis,
    "column": column_axis,
    "radial": radial_axis,
}

#: Accelerator ``line_axis`` -> the decay axis whose bins are the mask's lines.
#: A mask with no line axis (a 2-D point pattern) keeps the row binning it has
#: always had; the ``_radial`` envelopes exist for those.
LINE_DECAY_AXES: dict[str | None, str] = {"y": "row", "x": "column", None: "row"}


def build_decay_axis(name: str, height: int, width: int, device: torch.device) -> DecayAxis:
    """The axis ``name`` over an ``height x width`` plane; an unknown name raises."""
    if name not in DECAY_AXES:
        raise ValueError(f"decay axis {name!r} is not one of {sorted(DECAY_AXES)}.")
    return DECAY_AXES[name](height, width, device)


__all__ = [
    "DECAY_AXES",
    "LINE_DECAY_AXES",
    "DecayAxis",
    "build_decay_axis",
    "column_axis",
    "radial_axis",
    "row_axis",
    "row_index",
]
