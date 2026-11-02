"""Beltrami-parameterised adaptive space-filling-curve block (§1).

Warps the framework's fixed 2-D Hilbert curve by a learnable
quasi-conformal homeomorphism. A small head predicts a Beltrami
coefficient ``μ_θ`` (constrained to ``|μ| ≤ k < 1`` via
:func:`enforce_beltrami_constraint`); :class:`LinearBeltramiSolver`
turns it into a fold-free warp ``φ = z + h`` with bounded dilatation
``K = (1+k)/(1-k)``. The adaptive scan then orders grid cells by the
**base-Hilbert rank of their warped position** ``φ(z)``, so the
traversal bends toward content while staying locality-preserving (a
bounded-distortion map cannot scramble neighbouring ranks). When
``μ_θ ≡ 0`` the warp is the identity and the ordering is *exactly* the
fixed Hilbert curve — so the strategy strictly generalises a
fixed-Hilbert HSSC scan (pinned by
``test_beltrami_sfc_block_zero_mu_is_hilbert``).

Gradient path (#2360). The hard permutation comes from an ``argsort`` and
carries no gradient, so the block also returns ``soft_perm``: a row-softmax
assignment of each scan slot to the cells whose continuous warped Hilbert
rank lies nearest that slot's rank. The rank is read off the Hilbert-rank
map by bilinear sampling at ``φ(z)``, so it is differentiable in ``μ_θ``.
A consumer gathers with the hard indices and adds the straight-through
term ``soft - soft.detach()`` (QuadMamba-style [9]), which leaves the
forward values unchanged and lets the reconstruction loss reach ``μ_θ``.

Earlier this block sorted cells by ``φ.angle()`` — a single global
scalar that collapses 2-D locality (max L1 step ≈ 13 vs the Hilbert
curve's 1) and did *not* reduce to Hilbert at ``μ=0``, contradicting
the docstring's own (untested) consistency claim.

References:
    [1] L. V. Ahlfors, *Lectures on Quasiconformal Mappings*,
        AMS, 2006.
    [8] L. M. Lui et al., "Teichmüller mapping (T-map)", *SIAM J.
        Imag. Sci.*, 7(1), 2014.
    [9] F. Xie et al., "QuadMamba: Learning quadtree-based selective
        scan for visual state space model", *NeurIPS*, 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.conformal_geometry import (
    LinearBeltramiSolver,
    enforce_beltrami_constraint,
)
from spectramr.models.blocks.topology_linearizer import _hilbert_2d_indices


class BeltramiSFCBlock(nn.Module):
    """Learnable Beltrami-adaptive SFC over a 2-D grid.

    Args:
        grid_size: Square grid side. Must be a power of two
            (constraint from the Hilbert generator).
        in_channels: Number of conditioning channels fed to the
            μ head (typically the partial reconstruction or a
            content prior).
        hidden: Hidden width of the μ head.
        k_max: Upper bound on ``|μ|``. Smaller values make the
            adaptive curve closer to Hilbert; ``k_max → 1`` allows
            maximal warping (and risks numerical instability).
        solver: Optional pre-built solver; defaults to a fresh
            :class:`LinearBeltramiSolver`.
        soft_temperature: Temperature, in Hilbert-rank units, of the
            soft assignment behind ``soft_perm``.
    """

    def __init__(
        self,
        *,
        grid_size: int,
        in_channels: int = 1,
        hidden: int = 32,
        k_max: float = 0.9,
        solver: LinearBeltramiSolver | None = None,
        soft_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if grid_size & (grid_size - 1):
            raise ValueError(f"grid_size must be a power of two, got {grid_size}")
        if not soft_temperature > 0.0:
            raise ValueError(f"soft_temperature must be > 0, got {soft_temperature}")
        self.soft_temperature = float(soft_temperature)
        self.grid_size = grid_size
        self.k_max = k_max
        self.solver = solver or LinearBeltramiSolver(max_iter=20, tol=1e-3)

        # Tiny ConvNet that maps a content feature map to a complex μ field.
        self.mu_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=3, padding=1),  # real + imag of μ
        )

        # Fixed Hilbert traversal indices on the grid, cached as a buffer.
        idx = torch.as_tensor(_hilbert_2d_indices(grid_size, grid_size), dtype=torch.long)
        self.register_buffer("hilbert_indices", idx, persistent=False)
        # Inverse permutation: hilbert_rank[g] = the step at which grid cell g is
        # visited along the Hilbert curve. Used to order cells by the rank of
        # their warped position (see forward).
        self.register_buffer("hilbert_rank", idx.argsort(), persistent=False)
        rank_map = self.hilbert_rank.to(torch.float32).view(1, 1, grid_size, grid_size)
        self.register_buffer("hilbert_rank_map", rank_map, persistent=False)

    def _mu_from_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the constrained Beltrami coefficient from features."""
        out = self.mu_head(x)
        mu = enforce_beltrami_constraint(torch.complex(out[:, 0], out[:, 1]), k_max=self.k_max)
        return mu

    def forward(self, features: torch.Tensor, *, soft: bool = True) -> dict[str, torch.Tensor]:
        """Return the adaptive SFC indices plus the underlying μ / φ fields.

        Args:
            features: ``[B, C, H, W]`` content tensor. ``H = W =
                grid_size`` is enforced.
            soft: Also build ``soft_perm``; an evaluation-only caller that
                needs no gradient passes False and the key is absent.

        Returns:
            Dict with ``{"indices": [B, H*W] int64, "soft_perm":
            [B, H*W, H*W], "mu": [B, H, W], "phi": [B, H, W]}``. ``indices[b, k]`` is the flat grid
            index of the cell visited k-th in the adaptive scan
            (consumed by downstream Mamba scans); it is
            non-differentiable (produced by ``argsort``). ``soft_perm[b, k]``
            is a differentiable relaxation of the one-hot row of
            ``indices[b, k]`` for the straight-through gradient. ``mu`` and
            ``phi`` carry gradient and are exposed for the Beltrami /
            trajectory regularisers and diagnostics.
        """
        if features.shape[-2:] != (self.grid_size, self.grid_size):
            raise ValueError(
                f"BeltramiSFCBlock got features of spatial shape "
                f"{tuple(features.shape[-2:])}; expected ({self.grid_size},)*2"
            )
        gs = self.grid_size
        n = gs * gs
        mu = self._mu_from_features(features)
        phi = self.solver(mu).phi  # φ = z + h on the canonical [-1, 1] grid

        # Warp each cell's position by φ and re-quantise to a grid cell.
        # φ.real is the warped x (column) coordinate, φ.imag the warped y (row).
        scale = 0.5 * (gs - 1)
        col = ((phi.real + 1.0) * scale).round().clamp_(0, gs - 1).long()
        row = ((phi.imag + 1.0) * scale).round().clamp_(0, gs - 1).long()
        warped_flat = (row * gs + col).view(features.shape[0], -1)  # [B, N]

        # Order cells by the base-Hilbert rank of their WARPED position, with a
        # tie-break on the cell's own Hilbert rank so the composite key is a
        # strict total order — hence ``indices`` is always a valid permutation,
        # and μ=0 (⇒ warped == identity) reproduces the fixed Hilbert order
        # exactly.
        rank = self.hilbert_rank.to(warped_flat.device)
        warped_rank = rank[warped_flat]  # [B, N]
        composite = warped_rank * n + rank.unsqueeze(0)
        indices = composite.argsort(dim=-1, stable=True)  # [B, N] int64
        out = {"indices": indices, "mu": mu, "phi": phi}
        if soft:
            out["soft_perm"] = self._soft_permutation(phi, indices)
        return out

    def _soft_permutation(self, phi: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Relax the hard order into a ``[B, N, N]`` row-stochastic assignment."""
        b = phi.shape[0]
        # grid_sample's (x, y) on [-1, 1] with align_corners=True is exactly the
        # φ.real / φ.imag cell-centre convention used for the hard order above.
        grid = torch.stack((phi.real, phi.imag), dim=-1).to(self.hilbert_rank_map.dtype)
        rank_map = self.hilbert_rank_map.expand(b, -1, -1, -1)
        score = F.grid_sample(
            rank_map, grid, mode="bilinear", padding_mode="border", align_corners=True
        ).view(b, -1)  # [B, N] continuous warped Hilbert rank
        slot_score = score.gather(1, indices)  # [B, N]: rank of the cell in slot k
        dist = (slot_score.unsqueeze(-1) - score.unsqueeze(1)).pow(2)
        return torch.softmax(-dist / self.soft_temperature, dim=-1)


__all__ = ["BeltramiSFCBlock"]
