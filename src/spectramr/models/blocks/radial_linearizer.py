"""Morphological Radial Linearizer for Anatomy-Aware Vision SSMs.

Replaces Cartesian space-filling curves with **concentric shell routing**
that respects the physical geometry of the human head.  The sequence starts
at the FOV center (deep-brain tissue) and spirals out through discrete
Archimedean shells, or vice-versa.

**Why this matters (The Empty Air Paradox):**
Standard Hilbert/Morton curves begin at the top-left corner — pure background
air and Rician noise.  The Mamba ODE solver wastes its finite memory capacity
on noise before reaching anatomy.  A centrifugal spiral initializes the
hidden state on pristine central tissue.

Two operational levels are provided:

1. **Pixel-level permutations** (``radial_inside_out_indices``,
   ``radial_outside_in_indices``) — drop-in compatible with
   ``ImageTopologyLinearizer`` and ``MambaLayer2D.linearization_mode``.
2. **Patch-level ``MorphologicalRadialLinearizer``** — includes active
   latent segmentation (energy-based background masking) and ``einops``
   patch tokenization.  Use this for advanced dual-branch architectures.

Reference:
    Pipe, J.G. (1999). "Motion correction with PROPELLER MRI."
    Magnetic Resonance in Medicine, 42(5), 963–969.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pixel-level concentric permutation generators (for _MODE_GENERATORS)
# ---------------------------------------------------------------------------


def _concentric_sort_keys_2d(
    h: int,
    w: int,
    inside_out: bool = True,
) -> torch.Tensor:
    """Compute sort keys for concentric shell ordering on a 2D grid.

    Args:
        h: Grid height.
        w: Grid width.
        inside_out: If True, center→edge (centrifugal).
                    If False, edge→center (centripetal).

    Returns:
        Sort-key tensor of shape ``[h * w]``.
    """
    Y, X = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    Y_c, X_c = Y - cy, X - cx

    R = torch.sqrt(X_c**2 + Y_c**2)
    Theta = torch.atan2(Y_c, X_c)

    # Bin radius to discrete shells for stable ordering
    R_binned = torch.round(R)

    if inside_out:
        return R_binned.flatten() * 1000 + Theta.flatten()
    else:
        return -R_binned.flatten() * 1000 + Theta.flatten()


def _concentric_sort_keys_3d(
    d: int,
    h: int,
    w: int,
    inside_out: bool = True,
) -> torch.Tensor:
    """Compute sort keys for concentric shell ordering on a 3D grid."""
    Z, Y, X = torch.meshgrid(
        torch.arange(d, dtype=torch.float32),
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    cz, cy, cx = (d - 1) / 2.0, (h - 1) / 2.0, (w - 1) / 2.0
    Z_c, Y_c, X_c = Z - cz, Y - cy, X - cx

    R = torch.sqrt(X_c**2 + Y_c**2 + Z_c**2)
    Theta = torch.atan2(Y_c, X_c)
    Phi = torch.acos(Z_c / (R + 1e-6))

    R_binned = torch.round(R)

    if inside_out:
        return R_binned.flatten() * 1e6 + Phi.flatten() * 1000 + Theta.flatten()
    else:
        return -R_binned.flatten() * 1e6 + Phi.flatten() * 1000 + Theta.flatten()


def radial_inside_out_indices(h: int, w: int) -> torch.Tensor:
    """Centrifugal (center→edge) concentric permutation for a 2D grid.

    Returns:
        1D int64 permutation tensor of length ``h * w``.
    """
    keys = _concentric_sort_keys_2d(h, w, inside_out=True)
    return torch.argsort(keys).numpy()


def radial_outside_in_indices(h: int, w: int) -> torch.Tensor:
    """Centripetal (edge→center) concentric permutation for a 2D grid.

    Returns:
        1D int64 permutation tensor of length ``h * w``.
    """
    keys = _concentric_sort_keys_2d(h, w, inside_out=False)
    return torch.argsort(keys).numpy()


# ---------------------------------------------------------------------------
# Patch-level MorphologicalRadialLinearizer (advanced)
# ---------------------------------------------------------------------------


class MorphologicalRadialLinearizer(nn.Module):
    """Concentric shell patch linearizer with active segmentation masking.

    Routes patches from inside-out or outside-in through discrete
    Archimedean shells.  Generates an energy-based binary mask that
    gates background air patches, preventing the Mamba ODE from
    absorbing Rician noise.

    Args:
        shape: Full spatial dims ``(H, W)`` or ``(D, H, W)``.
        patch_size: Local patch edge length (e.g. 8 → 8×8 patches).
        mode: ``"inside_out"`` (centrifugal) or ``"outside_in"`` (centripetal).
        is_3d: Whether the input is volumetric.
        energy_quantile: Quantile threshold for background masking.
            Patches below this energy quantile are masked out.

    Example::

        lin = MorphologicalRadialLinearizer((256, 256), patch_size=8,
                                            mode="inside_out")
        seq, mask = lin(img)        # [B, N, token_dim], [B, N, 1]
        img_out = lin.reverse(seq)  # [B, C, H, W]
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        patch_size: int = 8,
        mode: str = "inside_out",
        is_3d: bool = False,
        energy_quantile: float = 0.05,
    ) -> None:
        super().__init__()
        self.P = patch_size
        self.mode = mode
        self.is_3d = is_3d
        self.energy_quantile = energy_quantile

        if not is_3d:
            self.H, self.W = shape[0], shape[1]
            self.gh, self.gw = self.H // self.P, self.W // self.P
            grid_shape = (self.gh, self.gw)
        else:
            self.D, self.H, self.W = shape[0], shape[1], shape[2]
            self.gd = self.D // self.P
            self.gh = self.H // self.P
            self.gw = self.W // self.P
            grid_shape = (self.gd, self.gh, self.gw)

        # Pre-compute concentric topology ONCE
        mac_fwd = self._generate_concentric_topology(grid_shape)
        # ``_generate_concentric_topology`` may already return a tensor;
        # ``torch.tensor(tensor, ...)`` triggers a deprecation warning that's
        # treated as an error under the project's filterwarnings policy.
        if torch.is_tensor(mac_fwd):
            self.register_buffer("mac_fwd", mac_fwd.detach().clone().to(torch.long))
        else:
            self.register_buffer("mac_fwd", torch.tensor(mac_fwd, dtype=torch.long))
        self.register_buffer("mac_inv", torch.argsort(self.mac_fwd))

        total_patches = math.prod(grid_shape)
        logger.info(
            "[MorphologicalRadialLinearizer] mode=%s, shape=%s, "
            "patch_size=%d, grid=%s, total_patches=%d",
            mode,
            shape,
            patch_size,
            grid_shape,
            total_patches,
        )

    def _generate_concentric_topology(self, grid_shape: tuple[int, ...]) -> torch.Tensor:
        """Generate concentric shell patch permutation."""
        inside_out = self.mode == "inside_out"

        if not self.is_3d:
            keys = _concentric_sort_keys_2d(grid_shape[0], grid_shape[1], inside_out=inside_out)
        else:
            keys = _concentric_sort_keys_3d(
                grid_shape[0],
                grid_shape[1],
                grid_shape[2],
                inside_out=inside_out,
            )
        return torch.argsort(keys)

    def forward(
        self,
        x: torch.Tensor,
        generate_mask: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Patchify, route concentrically, and optionally generate mask.

        Args:
            x: Input tensor ``[B, C, H, W]`` or ``[B, C, D, H, W]``.
            generate_mask: Whether to compute the energy-based tissue mask.

        Returns:
            Tuple of:
            - sequence: Routed patch tokens ``[B, N_patches, token_dim]``.
            - mask: Binary tissue mask ``[B, N_patches, 1]`` or ``None``.

        forward method for MorphologicalRadialLinearizer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            generate_mask (bool): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        from einops import rearrange

        if not self.is_3d:
            # [B, C, (gh*P), (gw*P)] → [B, C, gh*gw, P*P]
            patches = rearrange(
                x,
                "b c (gh ph) (gw pw) -> b c (gh gw) (ph pw)",
                gh=self.gh,
                gw=self.gw,
                ph=self.P,
                pw=self.P,
            )
        else:
            patches = rearrange(
                x,
                "b c (gd pd) (gh ph) (gw pw) -> b c (gd gh gw) (pd ph pw)",
                gd=self.gd,
                gh=self.gh,
                gw=self.gw,
                pd=self.P,
                ph=self.P,
                pw=self.P,
            )

        # Route patches through concentric shells
        routed = patches[:, :, self.mac_fwd, :]

        # Merge channel and patch dims → token sequence
        sequence = rearrange(routed, "b c n p -> b n (c p)")

        # Active segmentation: mask out background air patches
        mask = None
        if generate_mask:
            patch_energy = torch.norm(sequence, dim=-1, keepdim=True)
            threshold = torch.quantile(patch_energy, self.energy_quantile, dim=1, keepdim=True)
            mask = (patch_energy > threshold).float()

        return sequence, mask

    def reverse(self, sequence: torch.Tensor) -> torch.Tensor:
        """Restore Cartesian spatial layout from routed patch sequence.

        Args:
            sequence: Patch token sequence ``[B, N_patches, token_dim]``.

        Returns:
            Restored tensor ``[B, C, H, W]`` or ``[B, C, D, H, W]``.
        """
        from einops import rearrange

        if not self.is_3d:
            C = sequence.shape[-1] // (self.P * self.P)
            patches = rearrange(sequence, "b n (c p) -> b c n p", c=C)
            patches = patches[:, :, self.mac_inv, :]
            return rearrange(
                patches,
                "b c (gh gw) (ph pw) -> b c (gh ph) (gw pw)",
                gh=self.gh,
                gw=self.gw,
                ph=self.P,
                pw=self.P,
            )
        else:
            C = sequence.shape[-1] // (self.P**3)
            patches = rearrange(sequence, "b n (c p) -> b c n p", c=C)
            patches = patches[:, :, self.mac_inv, :]
            return rearrange(
                patches,
                "b c (gd gh gw) (pd ph pw) -> b c (gd pd) (gh ph) (gw pw)",
                gd=self.gd,
                gh=self.gh,
                gw=self.gw,
                pd=self.P,
                ph=self.P,
                pw=self.P,
            )

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, patch_size={self.P}, "
            f"grid=({self.gh}, {self.gw}), "
            f"energy_quantile={self.energy_quantile}"
        )
