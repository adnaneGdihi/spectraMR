"""Conformal multi-scale GeoMamba-ULF generator (§3 of the SFC plan).

Pyramid generator for ULF→HF super-resolution whose bottleneck uses
a Mamba-style traversal along a space-filling curve. The scale
hierarchy is a standard convolutional pyramid; here we use a depthwise
1-D conv as an SSM stand-in to keep the registration smoke-testable
without pulling Mamba kernels.

The adaptive Beltrami ordering is supplied by ``AdaptiveSFCHSSCStrategy``
through ``sfc_indices`` — a single, regularised μ-head owned by the
strategy. The generator owns **no** μ-head of its own; standalone calls
(no ``sfc_indices``) fall back to a fixed Hilbert scan. A previous
version instantiated a second :class:`BeltramiSFCBlock` here, but its
parameters received no gradient whenever the strategy drove the scan —
a dead, never-trained code path (pitfall #16) diverging from the head
the regularisers actually constrained.

References:
    [4] Driscoll & Trefethen, *Schwarz-Christoffel Mapping*, CUP, 2002.
    [6] Marques, Simonis & Webb, "Low-field MRI: An MR physics
        perspective", *J. Magn. Reson. Imaging*, 49(6), 2019.
    [8] L. M. Lui et al., "Teichmüller mapping (T-map)", *SIAM J.
        Imag. Sci.*, 7(1), 2014.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.topology_linearizer import _hilbert_2d_indices
from spectramr.models.registry import register_model


def _gather_along_sfc(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a ``[B, C, H, W]`` tensor along a per-sample traversal."""
    B, C, H, W = x.shape
    flat = x.view(B, C, H * W)
    idx = indices.unsqueeze(1).expand(B, C, H * W)
    return flat.gather(2, idx)


def _scatter_along_sfc(seq: torch.Tensor, indices: torch.Tensor, shape) -> torch.Tensor:
    B, C, _ = seq.shape
    H, W = shape
    out = seq.new_zeros(B, C, H * W)
    idx = indices.unsqueeze(1).expand(B, C, H * W)
    out.scatter_(2, idx, seq)
    return out.view(B, C, H, W)


def _straight_through(hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
    # Forward value is ``hard``; only the soft permutation receives the extra gradient.
    return hard + (soft - soft.detach())


@register_model(
    "conformal_geomamba_ulf",
    training_mode="reconstruction",
    spatial_dims=(2,),
)
class ConformalGeoMambaULF(nn.Module):
    """Conformal multi-scale ULF→HF super-resolution generator.

    Args:
        in_channels / out_channels: I/O channel counts.
        base_width: Width at the finest scale; doubled at each level.
        scales: Number of pyramid scales (``S+1`` resolutions).
        grid_size: Spatial resolution at the bottleneck (Beltrami head
            requires a power-of-two square grid).
        k_max: Beltrami constraint.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_width: int = 16,
        scales: int = 2,
        grid_size: int = 32,
        k_max: float = 0.9,
    ) -> None:
        super().__init__()
        if grid_size & (grid_size - 1):
            raise ValueError(f"grid_size must be a power of two, got {grid_size}")
        self.scales = scales
        self.grid_size = grid_size
        del k_max  # no internal μ-head; the strategy owns the Beltrami head
        widths = [base_width * (2**s) for s in range(scales + 1)]

        self.stem = nn.Conv2d(in_channels, widths[0], 3, padding=1)
        self.down = nn.ModuleList(
            [nn.Conv2d(widths[s], widths[s + 1], 4, stride=2, padding=1) for s in range(scales)]
        )
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(widths[s + 1], widths[s], 4, stride=2, padding=1)
                for s in range(scales)
            ]
        )
        # Bottleneck 1-D scan (SSM stand-in). The traversal order is supplied by
        # the strategy via ``sfc_indices``; standalone calls fall back to this
        # fixed Hilbert curve (no internal, never-trained μ-head).
        hilb = torch.as_tensor(_hilbert_2d_indices(grid_size, grid_size), dtype=torch.long)
        self.register_buffer("_hilbert_indices", hilb, persistent=False)
        self.scan = nn.Conv1d(widths[-1], widths[-1], kernel_size=7, padding=3, groups=widths[-1])
        self.head = nn.Conv2d(widths[0], out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        *args,
        sfc_indices: torch.Tensor | None = None,
        sfc_soft_perm: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        feats = [self.stem(x)]
        for s in range(self.scales):
            feats.append(self.down[s](feats[-1]))
        bottleneck = feats[-1]
        # Resize bottleneck to the SFC grid if needed.
        Hb, Wb = bottleneck.shape[-2:]
        gs = self.grid_size
        b_resized = F.interpolate(bottleneck, size=(gs, gs), mode="bilinear", align_corners=False)
        # Prefer the adaptive SFC ordering computed AND regularised by the
        # strategy (AdaptiveSFCHSSCStrategy passes ``sfc_indices``), so the
        # curve that drives the reconstruction is the same head the Beltrami /
        # trajectory regularisers constrain and the optimiser steps. Standalone
        # calls (no strategy) fall back to the fixed Hilbert curve.
        if sfc_indices is not None:
            expected = (x.shape[0], gs * gs)
            if tuple(sfc_indices.shape) != expected:
                raise ValueError(
                    "sfc_indices must have shape (B, grid_size**2) = "
                    f"{expected} to match the {gs}x{gs} bottleneck grid; got "
                    f"{tuple(sfc_indices.shape)}"
                )
            indices = sfc_indices.to(device=b_resized.device, dtype=torch.long)
        else:
            indices = self._hilbert_indices.to(b_resized.device).unsqueeze(0).expand(x.shape[0], -1)
        seq = _gather_along_sfc(b_resized, indices)
        if sfc_soft_perm is not None:
            if sfc_indices is None:
                raise ValueError("sfc_soft_perm requires the sfc_indices it relaxes")
            if tuple(sfc_soft_perm.shape) != (x.shape[0], gs * gs, gs * gs):
                raise ValueError(
                    "sfc_soft_perm must have shape (B, grid_size**2, grid_size**2); "
                    f"got {tuple(sfc_soft_perm.shape)}"
                )
            perm = sfc_soft_perm.to(dtype=seq.dtype)
            flat = b_resized.detach().reshape(seq.shape)
            seq = _straight_through(seq, torch.bmm(flat, perm.transpose(1, 2)))
        seq_out = self.scan(seq) + seq
        b_back = _scatter_along_sfc(seq_out, indices, (gs, gs))
        if sfc_soft_perm is not None:
            soft_back = torch.bmm(seq_out.detach(), perm).view_as(b_back)
            b_back = _straight_through(b_back, soft_back)
        b_out = F.interpolate(b_back, size=(Hb, Wb), mode="bilinear", align_corners=False)
        h = b_out
        for s in range(self.scales - 1, -1, -1):
            h = self.up[s](h) + feats[s]
        return self.head(h)


__all__ = ["ConformalGeoMambaULF"]
