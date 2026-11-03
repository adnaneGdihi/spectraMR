"""4D Beltrami solver for spatiotemporal SFCs (fMRI §1, MRF §1).

Generalises :class:`LinearBeltramiSolver` from the 2D anatomical
plan to four real dimensions. The 4D Beltrami operator acts on a
tensor field ``μ(x, t)`` and recovers a quasi-conformal
homeomorphism ``φ : ℝ³ × [0, T] → ℝ³ × [0, T]`` whose Jacobian
satisfies ``J(φ) = U Σ Vᵀ`` with ``Σ V^T - I = μ`` ([5, §3.2]).

For the separable case used by the MRF and fMRI proposals
(``μ = μ_spatial(x, t) ⊗ ν_temporal(x, t)``), the solver
decomposes into a planar 2D Beltrami solve plus a 1D time
reparameterisation — both already shipped under
``spectramr.infrastructure.physics.conformal_geometry`` and
``spectramr.infrastructure.physics.manifolds.time_reparameterisation``.

References:
    [4] L. V. Ahlfors, *Lectures on Quasiconformal Mappings*, AMS, 2006.
    [5] J. Heinonen, *Lectures on Analysis on Metric Spaces*, Springer, 2001.
    [6] Lui et al., "Teichmüller mapping (T-map)", *SIAM J. Imag. Sci.*, 2014.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spectramr.infrastructure.physics.conformal_geometry import LinearBeltramiSolver


@dataclass(frozen=True)
class Beltrami4DResult:
    phi_spatial: torch.Tensor  # complex [B, T, H, W]
    phi_temporal: torch.Tensor  # real    [B, T]
    K: float  # maximal dilatation
    iterations: int


class SeparableBeltrami4DSolver(torch.nn.Module):
    """Separable 4D Beltrami solve via planar μ + scalar ν decomposition.

    The full 4D Beltrami solve in :math:`\\mathbb{R}^3\\times[0,T]`
    is computationally heavy. For the proposals in this plan the
    spatial and temporal Beltrami fields are independent, so we
    perform a per-time-slice 2D Beltrami solve and a 1D time
    reparameterisation, then combine.

    The maximal dilatation of the product map is
    :math:`K = (1 + \\max(\\|\\mu\\|_\\infty, \\|\\nu\\|_\\infty))
    / (1 - \\max(\\|\\mu\\|_\\infty, \\|\\nu\\|_\\infty))`,
    which the solver reports for the Tier-1 audit check
    ``mrf_spiral_rotation_metadata_required``.
    """

    def __init__(self, *, planar_max_iter: int = 20, tol: float = 1e-3) -> None:
        super().__init__()
        self.planar = LinearBeltramiSolver(max_iter=planar_max_iter, tol=tol)

    @staticmethod
    def time_warp(nu: torch.Tensor, T: int) -> torch.Tensor:
        """Convert a per-step temporal dilatation ``ν[B, T]`` (with
        ``|ν| < 1``) to a cumulative reparameterisation
        ``φ_temporal[B, T] ∈ [0, T-1]``."""
        rates = 1.0 + nu
        cumulative = torch.cumsum(rates, dim=-1)
        denom = cumulative[..., -1:].clamp_min(1e-6)
        return cumulative / denom * float(T - 1)

    def forward(self, mu: torch.Tensor, nu: torch.Tensor) -> Beltrami4DResult:
        """Solve.

        Args:
            mu: Complex tensor ``[B, T, H, W]`` planar Beltrami coefficients.
            nu: Real tensor ``[B, T]`` with ``|ν| < 1``, the temporal
                dilatation field.

        Returns:
            :class:`Beltrami4DResult` with the per-time spatial map and
            the cumulative temporal warp.
        """
        if mu.dim() != 4 or not mu.is_complex():
            raise ValueError("expected complex mu of shape [B, T, H, W]")
        if nu.dim() != 2 or nu.shape[0] != mu.shape[0]:
            raise ValueError("expected real nu of shape [B, T]")
        B, T, H, W = mu.shape
        spatial = []
        total_iters = 0
        for t in range(T):
            r = self.planar(mu[:, t])
            spatial.append(r.phi)
            total_iters = max(total_iters, r.iterations)
        phi_spatial = torch.stack(spatial, dim=1)
        phi_temporal = self.time_warp(nu, T)
        k = max(mu.abs().amax().item(), nu.abs().amax().item())
        K = (1.0 + k) / max(1.0 - k, 1e-6)
        return Beltrami4DResult(
            phi_spatial=phi_spatial,
            phi_temporal=phi_temporal,
            K=K,
            iterations=total_iters,
        )


__all__ = ["Beltrami4DResult", "SeparableBeltrami4DSolver"]
