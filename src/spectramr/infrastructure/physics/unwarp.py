"""Physical unwarp composer for ULF MRI.

Composes the inverse displacement field from up to three sources of
EPI geometric distortion:

1. **B0 inhomogeneity** — static field offset map (Hz).
2. **Gradient nonlinearity** — vendor-supplied displacement
   field (typically polynomial, see
   :class:`spectramr.infrastructure.physics.vf_corrections.GNLUnwarpingOperator`).
3. **Maxwell concomitant gradient** — quadratic field offset
   that becomes geometrically dominant at ULF (see
   :mod:`spectramr.infrastructure.physics.maxwell_concomitant`).

All three are reduced to a per-voxel **phase-encode-axis pixel
shift** and applied via ``F.grid_sample`` in the same single
warp. Composition order does not matter to first order because
all three are small perturbations of the identity at clinical
imaging.

Convention:

- Tensor layout is ``(B, C, D, H, W)`` per CLAUDE.md.
- The phase-encode axis is **H** by default (most common in axial
  EPI). Override with ``pe_axis="W"`` if needed.
- Field maps are in Hz and have shape ``(D, H, W)`` (broadcast
  across batch and channel).

This module never calls raw FFTs (CLAUDE.md invariant) — it works
entirely in the spatial domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalUnwarp(nn.Module):
    """Composer for the (B0, grad-NL, Maxwell) ULF unwarp.

    Args:
        echo_spacing_s: EPI echo-spacing in seconds. Determines the
            Hz → pixel conversion: shift_pixels = delta_f_hz *
            echo_spacing_s * n_pe_lines / fov_pe.
        n_pe_lines: number of phase-encode lines (typically the H
            dimension of the volume; pass explicitly when partial-
            Fourier or in-plane acceleration is in play).
        pe_axis: phase-encode direction, ``"H"`` or ``"W"``.
            Default ``"H"``.
        align_corners: passed straight through to ``F.grid_sample``.
            Defaults to ``True`` because the identity grid is built with
            ``torch.linspace(-1.0, 1.0, N)``, which only samples the
            integer pixel centers under the ``align_corners=True``
            convention; using ``False`` here would make an identity
            warp (no field, no GNL) produce a linearly-interpolated
            output instead of the input. See
            ``tests/unit/infrastructure/physics/test_unwarp.py::TestIdentityWarp``.
    """

    def __init__(
        self,
        echo_spacing_s: float | None,
        n_pe_lines: int | None = None,
        pe_axis: str = "H",
        align_corners: bool = True,
        shift_mode: str = "epi",
        pixel_bandwidth_hz: float | None = None,
    ) -> None:
        super().__init__()
        if pe_axis not in {"H", "W"}:
            raise ValueError(f"pe_axis must be 'H' or 'W', got {pe_axis!r}")
        if shift_mode == "epi":
            if echo_spacing_s is None or echo_spacing_s <= 0:
                raise ValueError(
                    "echo_spacing_s must be > 0 for shift_mode='epi', got " f"{echo_spacing_s!r}"
                )
        elif shift_mode == "readout":
            if pixel_bandwidth_hz is None or pixel_bandwidth_hz <= 0:
                raise ValueError(
                    "shift_mode='readout' needs pixel_bandwidth_hz > 0, got "
                    f"{pixel_bandwidth_hz!r}"
                )
        else:
            raise ValueError(f"shift_mode must be 'epi' or 'readout', got {shift_mode!r}")
        self.echo_spacing_s = echo_spacing_s
        self.n_pe_lines = n_pe_lines
        self.pe_axis = pe_axis
        self.align_corners = align_corners
        self.shift_mode = shift_mode
        self.pixel_bandwidth_hz = pixel_bandwidth_hz

    @property
    def shift_axis(self) -> str:
        """The axis the offset moves samples along: PE for 'epi', readout for 'readout'."""
        if self.shift_mode == "epi":
            return self.pe_axis
        return "W" if self.pe_axis == "H" else "H"

    def forward(
        self,
        volume: torch.Tensor,
        b0_hz: torch.Tensor | None = None,
        gnl_displacement: torch.Tensor | None = None,
        maxwell_hz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the composed inverse warp to ``volume``.

        Args:
            volume: ``(B, C, D, H, W)`` distorted input.
            b0_hz: optional ``(D, H, W)`` or ``(B, D, H, W)`` field
                offset map in Hz.
            gnl_displacement: optional ``(B, 3, D, H, W)`` dense
                displacement field in **normalised** coordinates
                ``[-1, 1]``.
            maxwell_hz: optional ``(D, H, W)`` or ``(B, D, H, W)``
                concomitant offset map in Hz (typically computed by
                :func:`maxwell_concomitant.maxwell_concomitant_field_hz`).

        Returns:
            ``(B, C, D, H, W)`` unwarped volume.
        """
        if volume.ndim != 5:
            raise ValueError(f"PhysicalUnwarp expects (B, C, D, H, W), got {tuple(volume.shape)}")
        B, C, D, H, W = volume.shape
        device = volume.device
        dtype = volume.dtype

        # Build the base sampling grid in normalised [-1, 1] coords.
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, D, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
            indexing="ij",
        )
        # grid_sample expects (x, y, z) ordering on the last dim.
        grid = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)

        # Hz offsets, summed, then converted to pixel shifts along the shift axis.
        delta_hz = self._zero_field(B, D, H, W, device, dtype)
        if b0_hz is not None:
            delta_hz = delta_hz + self._broadcast_field(b0_hz, B)
        if maxwell_hz is not None:
            delta_hz = delta_hz + self._broadcast_field(maxwell_hz, B)
        if self.shift_mode == "epi":
            # shift = f * ESP * Npe (the canonical EPI distortion formula).
            n_pe = self.n_pe_lines if self.n_pe_lines is not None else (H if self.pe_axis == "H" else W)
            delta_px = delta_hz * (float(self.echo_spacing_s) * n_pe)
        else:
            # Readout model: a df offset moves a sample by df / pixel bandwidth.
            delta_px = delta_hz / float(self.pixel_bandwidth_hz)
        n_axis = H if self.shift_axis == "H" else W
        shift_norm = 2.0 * delta_px / max(n_axis - 1, 1)
        grid = grid.clone()
        grid[..., 1 if self.shift_axis == "H" else 0] = (
            grid[..., 1 if self.shift_axis == "H" else 0] + shift_norm
        )

        # Apply the GNL displacement on top (already in normalised coords).
        if gnl_displacement is not None:
            if gnl_displacement.shape != (B, 3, D, H, W):
                raise ValueError(
                    "gnl_displacement must have shape (B, 3, D, H, W); got "
                    f"{tuple(gnl_displacement.shape)}"
                )
            # Permute (B, 3, D, H, W) → (B, D, H, W, 3) for grid_sample.
            gnl = gnl_displacement.permute(0, 2, 3, 4, 1)
            grid = grid + gnl

        # `mode="bilinear"` in 5D gives trilinear sampling.
        return F.grid_sample(
            volume,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_field(
        B: int, D: int, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros((B, D, H, W), device=device, dtype=dtype)

    @staticmethod
    def _broadcast_field(field: torch.Tensor, B: int) -> torch.Tensor:
        if field.ndim == 3:
            return field.unsqueeze(0).expand(B, -1, -1, -1)
        if field.ndim == 4:
            if field.shape[0] == B:
                return field
            if field.shape[0] == 1:
                return field.expand(B, -1, -1, -1)
        raise ValueError(f"field must have shape (D,H,W) or (B,D,H,W); got {tuple(field.shape)}")


__all__ = ["PhysicalUnwarp"]
