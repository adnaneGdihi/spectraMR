r"""Voxelwise Fisher-Rao geodesic loss under a Rician magnitude likelihood.

Each voxel of a magnitude image is modelled as a Rician observation
:math:`m\sim\mathrm{Rice}(\nu,\sigma)` of the true amplitude :math:`\nu`, with
the arm's declared noise level :math:`\sigma`. The prediction and the target
are two points :math:`\nu_p,\nu_t` on this one-parameter statistical manifold,
and the loss is the squared Fisher-Rao geodesic distance between them.

Fisher information. With :math:`z=m\nu/\sigma^2` and :math:`A=I_1/I_0`, the
score is :math:`\partial_\nu\log p=(m\,A(z)-\nu)/\sigma^2`, so

.. math::

    I(\nu)=\frac{\mathbb{E}[m^2A(z)^2]-\nu^2}{\sigma^4}=\frac{J(\nu/\sigma)}{\sigma^2},
    \qquad J(u)=\mathbb{E}_{r\sim\mathrm{Rice}(u,1)}\!\big[r^2A(ru)^2\big]-u^2 .

:math:`J(u)\to1` as :math:`u\to\infty` (the Gaussian limit) and
:math:`J(u)\approx u^2` as :math:`u\to0` (Rayleigh :math:`\mathbb{E}[r^4]=8`), so
the metric vanishes where the signal sinks into the noise floor.

Geodesic distance. On a one-dimensional manifold the geodesic is the
parameter segment itself, so with :math:`\Phi(u)=\int_0^u\sqrt{J(s)}\,ds`

.. math::

    d_{\mathrm{FR}}(\nu_p,\nu_t)=\Big|\Phi\!\big(\tfrac{\nu_t}{\sigma}\big)-\Phi\!\big(\tfrac{\nu_p}{\sigma}\big)\Big| .

The loss is :math:`\sigma^2\,\overline{d_{\mathrm{FR}}^2}`. In the high-SNR
limit :math:`\Phi(u)=u+\mathrm{const}`, so it is exactly the MSE; at low SNR it
discounts errors the Rician noise floor cannot distinguish. ``likelihood=
"gaussian"`` sets :math:`J\equiv1` (the MSE), which is the single-knob control.

:math:`J` and :math:`\Phi` are tabulated once in float64 by trapezoidal
quadrature over :math:`r`, with exponentially scaled Bessel functions, and
evaluated by linear interpolation with a unit-slope tail beyond the grid.
Because the Rician likelihood is even in :math:`\nu`, :math:`\Phi` is extended
as an odd function, so a negative prediction is still pulled towards the target
rather than mirrored onto it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

#: Grid over the dimensionless SNR u = nu / sigma; J(U_MAX) = 1 - O(1/U_MAX^2).
_U_MAX = 32.0
_N_U = 1025
_N_R = 2048
_R_PAD = 12.0


def rician_fisher_information(u: torch.Tensor, n_r: int = _N_R) -> torch.Tensor:
    r"""Dimensionless Rician Fisher information :math:`J(u)` (float64 quadrature)."""
    u = u.to(torch.float64).reshape(-1, 1)
    r_max = float(u.max()) + _R_PAD
    r = torch.linspace(0.0, r_max, n_r, dtype=torch.float64).reshape(1, -1)
    x = r * u
    # Rice(u, 1) density r exp(-(r-u)^2/2) i0e(ru): i0e = exp(-x) I0(x) keeps it finite.
    density = r * torch.exp(-0.5 * (r - u) ** 2) * torch.special.i0e(x)
    ratio = torch.special.i1e(x) / torch.special.i0e(x)
    second_moment = torch.trapezoid(density * (r * ratio) ** 2, r, dim=-1)
    mass = torch.trapezoid(density, r, dim=-1)
    return (second_moment / mass - u.reshape(-1) ** 2).clamp_min(0.0)


def _gaussian_j(u: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(u, dtype=torch.float64)


#: likelihood name -> dimensionless Fisher information J(u).
LIKELIHOODS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "rician": rician_fisher_information,
    "gaussian": _gaussian_j,
}


@lru_cache(maxsize=len(LIKELIHOODS))
def _arc_length_table(likelihood: str) -> torch.Tensor:
    """Tabulated Phi(u) = int_0^u sqrt(J) on the uniform u grid, float64."""
    u = torch.linspace(0.0, _U_MAX, _N_U, dtype=torch.float64)
    root_j = LIKELIHOODS[likelihood](u).sqrt()
    du = u[1] - u[0]
    increments = 0.5 * (root_j[1:] + root_j[:-1]) * du
    return torch.cat([torch.zeros(1, dtype=torch.float64), increments.cumsum(0)])


@register_loss(
    name="fisher_rao_geodesic",
    aliases=[
        "fisher_rao_flow",
        "fr_geodesic",
        "information_geometry_geodesic",
    ],
    domain="image",
)
class FisherRaoGeodesicLoss(nn.Module):
    r"""Squared voxelwise Fisher-Rao distance under a Rician (or Gaussian) likelihood.

    Args:
        noise_sigma: Noise standard deviation :math:`\sigma` of the magnitude
            likelihood, in the units of the normalised images. Required: the
            geometry is defined by it.
        likelihood: ``"rician"`` (the treatment) or ``"gaussian"`` (:math:`J\equiv1`,
            exactly the MSE; the control).
        reduction: ``"mean"`` or ``"sum"`` over voxels.
    """

    def __init__(
        self,
        noise_sigma: float | None = None,
        likelihood: str = "rician",
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if noise_sigma is None or not noise_sigma > 0.0:
            raise ValueError(
                "fisher_rao_geodesic needs kwargs.noise_sigma > 0 (the magnitude "
                f"likelihood's noise std in normalised image units); got {noise_sigma!r}."
            )
        if likelihood not in LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {sorted(LIKELIHOODS)}; got {likelihood!r}."
            )
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}.")
        self.noise_sigma = float(noise_sigma)
        self.likelihood = likelihood
        self.reduction = reduction
        self.register_buffer(
            "_phi", _arc_length_table(likelihood).to(torch.float32), persistent=False
        )
        self._du = _U_MAX / (_N_U - 1)

    def arc_length(self, u: torch.Tensor) -> torch.Tensor:
        r"""Odd extension of :math:`\Phi(u)`, linear per grid cell, unit slope past the grid."""
        phi = self._phi.to(device=u.device, dtype=u.dtype)
        a = u.abs()
        pos = (a / self._du).clamp(max=_N_U - 1 - 1e-4)
        idx = pos.floor().long()
        frac = pos - idx
        inside = phi[idx] + frac * (phi[idx + 1] - phi[idx])
        tail = phi[-1] + (a - _U_MAX)
        return torch.sign(u) * torch.where(a < _U_MAX, inside, tail)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"shape mismatch: prediction {tuple(prediction.shape)} vs "
                f"target {tuple(target.shape)}."
            )
        inv_sigma = 1.0 / self.noise_sigma
        dist = self.arc_length(target.float() * inv_sigma) - self.arc_length(
            prediction.float() * inv_sigma
        )
        sq = dist.square() * (self.noise_sigma**2)
        return sq.mean() if self.reduction == "mean" else sq.sum()
