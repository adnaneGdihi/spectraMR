"""CS-MNO single layer: spectral + SFC-Mamba + pointwise residual.

Forward law (per spec §3.1, ``TODO/Mamba_NO.md``):

.. math::

    v_{l+1}(x) = \\sigma\\Big(
        \\underbrace{\\mathcal{F}^{-1}(R_l\\,\\mathcal{F}(v_l))(x)}_{\\text{spectral, K modes}}
        \\;+\\;
        \\underbrace{\\big[\\pi^{-1} \\circ \\mathrm{Mamba}_l \\circ \\pi(v_l)\\big](x)}_{\\text{SFC scan with }\\Delta_t}
        \\;+\\;
        W_l\\,v_l(x)
    \\Big)

The spectral branch supplies the global low-frequency content; the
SFC-Mamba branch supplies high-frequency, anatomy-aligned correlations
along the Hilbert path; the 1×1 conv ``W`` is a pointwise residual
that lets the network bypass either branch when useful.

The layer is dimension-agnostic at the API level: ``d=2`` uses
``SpectralBranch2d`` + ``nn.Conv2d``, ``d=3`` uses
``SpectralBranch3d`` + ``nn.Conv3d``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .continuous_sfc_mamba import ContinuousSFCMamba
from .spectral_branch import SpectralBranch1d, SpectralBranch2d, SpectralBranch3d

__all__ = ["CSMNOLayer"]


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class CSMNOLayer(nn.Module):
    """One CS-MNO layer.

    Args:
        dim:               Channel dim of input/output (per-token width).
        modes:             Truncation modes for the spectral branch.
                           Ignored when ``disable_spectral=True``.
        d_state:           Hidden state dim of the SFC-Mamba scan.
        spatial_dims:      2 for image-like tensors ``[B, C, H, W]``,
                           3 for volumes ``[B, C, D, H, W]``.
        use_physical_arc:  If True, the SFC-Mamba scan uses the
                           per-token physical Δt (Theorem A); else
                           falls back to vanilla Mamba's learnt Δ.
        disable_spectral:  If True, the spectral branch is replaced
                           with a no-op ``nn.Identity()``. This turns
                           the layer into a pure SFC-Mamba layer (used
                           by HS-MNO and C-MNO from the spec).
        activation:        Pointwise nonlinearity name; one of
                           {gelu, relu, silu}. Unknown values raise
                           — no silent fallback.
    """

    def __init__(
        self,
        dim: int,
        modes: int,
        d_state: int = 16,
        spatial_dims: int = 2,
        use_physical_arc: bool = True,
        disable_spectral: bool = False,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if spatial_dims not in (1, 2, 3):
            raise ValueError(f"CSMNOLayer: spatial_dims must be 1, 2, or 3, got {spatial_dims}")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"CSMNOLayer: unknown activation {activation!r}. Available: {sorted(_ACTIVATIONS)}"
            )

        self.dim = dim
        self.spatial_dims = spatial_dims

        # Mamba is intrinsically 1-D — only the FNO spectral branch and
        # the residual W projection vary per spatial rank.
        if spatial_dims == 1:
            self.spectral = nn.Identity() if disable_spectral else SpectralBranch1d(dim, modes)
            self.W = nn.Conv1d(dim, dim, kernel_size=1)
        elif spatial_dims == 2:
            self.spectral = nn.Identity() if disable_spectral else SpectralBranch2d(dim, modes)
            self.W = nn.Conv2d(dim, dim, kernel_size=1)
        else:
            self.spectral = nn.Identity() if disable_spectral else SpectralBranch3d(dim, modes)
            self.W = nn.Conv3d(dim, dim, kernel_size=1)

        self.mamba = ContinuousSFCMamba(dim=dim, d_state=d_state, use_physical_arc=use_physical_arc)
        self.act = _ACTIVATIONS[activation]()

    def forward(
        self,
        x: torch.Tensor,
        forward_idx: torch.Tensor,
        inverse_idx: torch.Tensor,
        delta_phys: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:           ``[B, C, *spatial]`` input.
            forward_idx: ``[N]`` SFC permutation (token order).
            inverse_idx: ``[N]`` inverse permutation.
            delta_phys:  ``[N]`` per-token physical Δt, or None when
                         the SFC-Mamba block is in scalar-Δ mode.

        Returns:
            ``[B, C, *spatial]`` output (same shape as input).
        """
        B, C = x.shape[:2]
        spatial = x.shape[2:]
        # ``spatial`` is a torch.Size (Python ints), so math.prod is exact and
        # allocation-free; the former torch.tensor(...).prod().item() built a
        # throwaway CPU tensor on every forward. See non-negotiable 9.
        # N806: `N` is the token count in this file's `[B, C, N]` shape
        # vocabulary, matching `B, C` above -- a tensor-dimension symbol,
        # not a variable that wants snake_case.
        N = math.prod(spatial)

        # SFC branch: linearise → scan → de-linearise.
        x_flat = x.reshape(B, C, N)  # [B, C, N]
        x_seq = x_flat[:, :, forward_idx]  # [B, C, N]
        x_seq = x_seq.transpose(1, 2)  # [B, N, C]
        y_seq = self.mamba(x_seq, delta_phys)  # [B, N, C]
        y_flat = y_seq.transpose(1, 2)  # [B, C, N]
        # Inverse permutation via gather, then reshape back.
        y_unscat = y_flat.new_empty(y_flat.shape)
        y_unscat[:, :, forward_idx] = y_flat  # in-place scatter via index
        y_grid = y_unscat.reshape(B, C, *spatial)

        # Spectral branch + pointwise residual.
        return self.act(self.spectral(x) + y_grid + self.W(x))
