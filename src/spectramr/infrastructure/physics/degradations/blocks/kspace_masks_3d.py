"""(kz, ky) Cartesian and partial-Fourier sampling masks for 3-D encoding.

Masks are per call, not per sample; pass a ``torch.Generator`` for reproducibility.
"""

from __future__ import annotations

from typing import Literal

import torch

from spectramr.infrastructure.physics.digital_twin_simulator import outer_line_count
from spectramr.infrastructure.physics.fft_ops import centred_freqs

__all__ = ["cartesian_mask_3d", "partial_fourier_mask_3d"]


def _acs_block(n_kz: int, n_ky: int, acs: tuple[int, int]) -> torch.Tensor:
    a_kz, a_ky = acs
    if not (1 <= a_kz <= n_kz and 1 <= a_ky <= n_ky):
        raise ValueError(f"acs {acs} must lie in [1, ({n_kz}, {n_ky})]")
    block = torch.zeros(n_kz, n_ky, dtype=torch.bool)
    z0, y0 = n_kz // 2 - a_kz // 2, n_ky // 2 - a_ky // 2
    block[z0 : z0 + a_kz, y0 : y0 + a_ky] = True
    return block


def cartesian_mask_3d(
    n_kz: int,
    n_ky: int,
    R: float,
    *,
    acs: tuple[int, int] = (2, 8),
    density: Literal["uniform", "variable"] = "uniform",
    alpha: float = 3.0,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Boolean ``[n_kz, n_ky]`` mask acquiring ``max(acs_area, round(n_kz*n_ky/R))`` lines.

    fastMRI convention: the centred ACS block counts toward ``R``. The outer budget is
    ``outer_line_count`` applied to the flattened ``n_kz * n_ky`` grid, drawn without
    replacement from the ACS complement; ``variable`` weights lines by ``(1+|k|)^-alpha``
    with ``|k|`` the normalised (kz, ky) radius. ``generator`` must be a CPU generator.
    """
    if n_kz < 1 or n_ky < 1:
        raise ValueError(f"n_kz and n_ky must be >= 1, got ({n_kz}, {n_ky})")
    if not R >= 1.0:
        raise ValueError(f"R must be >= 1, got {R}")
    if density not in ("uniform", "variable"):
        raise ValueError(f"density must be 'uniform' or 'variable', got {density!r}")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    mask = _acs_block(n_kz, n_ky, acs)
    free = (~mask).flatten().nonzero().squeeze(-1)
    n_outer = outer_line_count(n_kz * n_ky, int(mask.sum()), R)
    if n_outer > 0:
        if density == "uniform":
            pick = torch.randperm(free.numel(), generator=generator)[:n_outer]
        else:
            kz = 2 * centred_freqs(n_kz)
            ky = 2 * centred_freqs(n_ky)
            radius = torch.sqrt(kz[:, None] ** 2 + ky[None, :] ** 2).flatten()[free]
            pick = torch.multinomial(
                (1.0 + radius) ** -alpha, n_outer, replacement=False, generator=generator
            )
        mask.view(-1)[free[pick]] = True
    return mask.to(device) if device is not None else mask


def partial_fourier_mask_3d(
    n_kz: int,
    n_ky: int,
    fraction: float,
    *,
    axis: Literal["ky", "kz", "ky+kz"],
) -> torch.Tensor:
    """Keep the low-index ``round(n * fraction)`` samples along ``axis`` (both for ``ky+kz``).

    For ``ky+kz`` the mask is the intersection (a rectangle), not a union.
    """
    if not 0.5 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0.5, 1], got {fraction}")
    if axis not in ("ky", "kz", "ky+kz"):
        raise ValueError(f"axis must be 'ky', 'kz' or 'ky+kz', got {axis!r}")
    keep_kz = round(n_kz * fraction) if "kz" in axis else n_kz
    keep_ky = round(n_ky * fraction) if "ky" in axis else n_ky
    mask = torch.zeros(n_kz, n_ky, dtype=torch.bool)
    mask[:keep_kz, :keep_ky] = True
    return mask
