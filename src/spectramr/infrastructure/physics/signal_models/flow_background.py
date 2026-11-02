"""Background-phase correction for phase-contrast velocity maps.

Eddy currents and concomitant fields add a smooth, spatially varying offset to
every encoded phase. It is estimated on static tissue, where the true velocity
is zero, as a low-order polynomial in the spatial coordinates, and subtracted
everywhere (the correction OSU's ``backgroundCorrection3D`` applies before
velocity quantification). The fit is weighted least squares solved through the
normal equations, batched over every leading axis, so a whole ``[E, T, ...]``
stack is one ``linalg.solve`` rather than a loop.
"""

from __future__ import annotations

import itertools

import torch

#: Ridge added to the normal equations, relative to their mean diagonal. Keeps a
#: sparse static mask finite; it is far below any well-posed fit's eigenvalues.
_RIDGE = 1e-6


def polynomial_basis(
    shape: tuple[int, ...], order: int, *, device=None, dtype=None
) -> torch.Tensor:
    """Monomials of total degree ``<= order`` over a grid, ``[prod(shape), n_terms]``.

    Coordinates are normalised to ``[-1, 1]`` per axis so the normal equations
    stay well conditioned at any matrix size.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    axes = [torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype) for n in shape]
    grids = [g.reshape(-1) for g in torch.meshgrid(*axes, indexing="ij")]
    terms = [
        exponents
        for exponents in itertools.product(range(order + 1), repeat=len(shape))
        if sum(exponents) <= order
    ]
    columns = [
        torch.stack([g**e for g, e in zip(grids, exps, strict=True)]).prod(dim=0) for exps in terms
    ]
    return torch.stack(columns, dim=-1)


def fit_background_phase(
    phase: torch.Tensor, static_mask: torch.Tensor, order: int = 1, spatial_ndim: int = 2
) -> torch.Tensor:
    """Polynomial background fitted on ``static_mask``, evaluated on the full grid.

    Args:
        phase: ``[..., *spatial]`` phase (radians), the last ``spatial_ndim`` axes spatial.
        static_mask: weights broadcastable to ``phase``; 1 on static tissue, 0 elsewhere.
        order: total polynomial degree.
        spatial_ndim: 2 or 3.

    Returns:
        The fitted background, same shape as ``phase``. Subtract it (and re-wrap)
        to correct. A mask with fewer voxels than monomials returns the ridge
        solution rather than raising: callers needing a guarantee check the
        voxel count once, outside the hot path.
    """
    if spatial_ndim not in (2, 3):
        raise ValueError(f"spatial_ndim must be 2 or 3, got {spatial_ndim}")
    spatial = tuple(phase.shape[-spatial_ndim:])
    basis = polynomial_basis(spatial, order, device=phase.device, dtype=phase.dtype)
    lead = phase.shape[:-spatial_ndim]
    values = phase.reshape(-1, basis.shape[0])
    weights = torch.broadcast_to(static_mask.to(phase.dtype), phase.shape).reshape(values.shape)

    gram = torch.einsum("np,pm,pk->nmk", weights, basis, basis)
    rhs = torch.einsum("np,pm,np->nm", weights, basis, values)
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1.0)
    eye = torch.eye(basis.shape[1], device=phase.device, dtype=phase.dtype)
    # solve_ex: the ridge keeps the system non-singular, so skip the error check's host sync.
    coeffs, _ = torch.linalg.solve_ex(gram + _RIDGE * scale[:, None, None] * eye, rhs)
    return (coeffs @ basis.T).reshape(*lead, *spatial)


__all__ = ["fit_background_phase", "polynomial_basis"]
