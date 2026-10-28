"""KAN-based adaptive data consistency.

Replaces the CNN-based ``AdaptiveDataConsistency`` trust map with a small
shared KAN that consumes physical k-space coordinates plus the log-magnitude
of the measured signal:

    lambda(k) = sigmoid(KAN(log|y(k)| + eps, ||k||/k_max, atan2(k_y, k_x)/pi))

The KAN is shared across coil channels (the trust weight depends on physical
k-space geometry, not coil identity) and across batch elements.

Output blending is identical to the CNN ADC:

    k_new(k) = (1 - lambda(k)) * k_pred(k) + lambda(k) * y(k),  k in Omega
             = k_pred(k),                                       otherwise.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from spectramr.models.blocks.kan_layer import KANLayer

from .dc_mask import align_dc_mask
from .fft_ops import _to_complex, fft2c, ifft2c
from .implementations.fft_operator import MultiCoilFFTOperator


class KANAdaptiveDataConsistency(nn.Module):
    """KAN-based adaptive DC layer.

    Args:
        kan_hidden: Hidden dim of the two-layer KAN.
        kan_grid_size: Grid resolution per univariate edge.
        kan_spline_order: B-spline order (3 = cubic).
        eps: Small constant for log-magnitude stability.
    """

    def __init__(
        self,
        kan_hidden: int = 16,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        # Three input edges (log|y|, r, theta) -> 1 output (logit).
        self.kan = nn.Sequential(
            KANLayer(3, kan_hidden, kan_grid_size, kan_spline_order),
            KANLayer(kan_hidden, 1, kan_grid_size, kan_spline_order),
        )
        self.eps = float(eps)
        self.mc_operator = MultiCoilFFTOperator(normalized=True, center=True)
        # Cache of (radius, angle) channels keyed by (H, W). They depend
        # only on grid shape; rebuild only on shape change.
        self._coord_cache_shape: tuple[int, int] | None = None
        self.register_buffer("_coord_radius", torch.empty(0), persistent=False)
        self.register_buffer("_coord_angle", torch.empty(0), persistent=False)
        # Telemetry: most-recent trust-map summary statistics, populated by
        # _compute_lambda after every forward. Useful for the validation
        # callback that compares low-frequency-trust vs high-frequency-trust
        # — see plan §5.1 ("trust map statistics").
        # Layout: [center_mean, periphery_mean, overall_mean, overall_std].
        self.register_buffer("_last_trust_stats", torch.zeros(4), persistent=False)

    @torch.no_grad()
    def _build_coords(self, H: int, W: int, device: torch.device) -> None:
        ky = torch.linspace(-(H // 2), H - 1 - H // 2, H, device=device)
        kx = torch.linspace(-(W // 2), W - 1 - W // 2, W, device=device)
        ky_g, kx_g = torch.meshgrid(ky, kx, indexing="ij")
        r = torch.sqrt(ky_g**2 + kx_g**2)
        r_max = float(r.max().item())
        if r_max <= 0:
            r_max = 1.0
        # Normalize radius to [-1, 1] so it lies inside the default KAN grid range.
        # Map [0, k_max] -> [-1, 1].
        self._coord_radius = (r / r_max) * 2.0 - 1.0
        # Angle in [-pi, pi] / pi -> [-1, 1]
        self._coord_angle = torch.atan2(ky_g, kx_g) / math.pi
        self._coord_cache_shape = (H, W)

    def _compute_lambda(self, measured_kspace: torch.Tensor) -> torch.Tensor:
        """Return lambda map ``[B, 1, H, W]`` in (0, 1)."""
        # measured_kspace is complex [B, C, H, W] (or [B, 1, H, W]).
        B, _, H, W = measured_kspace.shape
        if self._coord_cache_shape != (H, W):
            self._build_coords(H, W, measured_kspace.device)

        mag = measured_kspace.abs() + self.eps
        # Mean over coil channels so the KAN is shared across the coil axis.
        if mag.shape[1] > 1:
            mag = mag.mean(dim=1, keepdim=True)
        log_mag = torch.log(mag)
        # Standardize log_mag per-batch into a roughly [-1, 1] range so it
        # sits inside the KAN's grid support (otherwise the spline value is
        # truncated to zero outside the grid).
        log_mag_flat = log_mag.flatten(2)  # [B, 1, H*W]
        m = log_mag_flat.mean(dim=-1, keepdim=True)
        s = log_mag_flat.std(dim=-1, keepdim=True).clamp_min(1e-3)
        log_mag = ((log_mag_flat - m) / s).reshape(B, 1, H, W).clamp(-3.0, 3.0) / 3.0

        # Broadcast spatial-only coords to the batch dimension.
        r = self._coord_radius.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)
        theta = self._coord_angle.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)

        feats = torch.cat([log_mag, r, theta], dim=1)  # [B, 3, H, W]
        feats_seq = feats.permute(0, 2, 3, 1).reshape(B * H * W, 3)
        logits = self.kan(feats_seq).reshape(B, H, W, 1).permute(0, 3, 1, 2)
        lam = torch.sigmoid(logits)  # [B, 1, H, W]

        # Telemetry: trust at center 8% (DC region) vs periphery (everything
        # else). Higher center means the KAN learned to trust acquired
        # low-frequency samples more — the expected behavior for any
        # well-conditioned MRI reconstruction.
        with torch.no_grad():
            r_norm = self._coord_radius.abs() if self._coord_radius.numel() > 0 else None
            if r_norm is not None and r_norm.numel() > 0:
                # _coord_radius is in [-1, 1]; abs() gives [0, 1] from DC outward.
                center_mask = (r_norm < 0.08).float()
                periph_mask = 1.0 - center_mask
                # Per-batch mean over spatial dims, then average across batch.
                lam_center = (lam * center_mask).sum(dim=(-2, -1)) / (
                    center_mask.sum().clamp_min(1.0)
                )
                lam_periph = (lam * periph_mask).sum(dim=(-2, -1)) / (
                    periph_mask.sum().clamp_min(1.0)
                )
                self._last_trust_stats[0] = lam_center.mean().detach()
                self._last_trust_stats[1] = lam_periph.mean().detach()
                self._last_trust_stats[2] = lam.mean().detach()
                self._last_trust_stats[3] = lam.std().detach()

        return lam

    def forward(
        self,
        img_pred: torch.Tensor,
        measured_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
        is_kspace_domain: bool = False,
        acs_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mirror the AdaptiveDataConsistency interface.

        Args:
            img_pred: Predicted tensor (image or k-space, real-interleaved or
                complex). When in k-space domain, FFT is skipped.
            measured_kspace: Acquired k-space measurements.
            mask: Sampling mask ``[B, 1, H, W]``.
            sensitivity_maps: Optional coil sensitivity maps for SENSE.
            is_kspace_domain: Treat ``img_pred`` as k-space if True.
            acs_mask: Accepted for interface compatibility; unused.

        Returns:
            Data-consistent tensor in the same domain/layout as ``img_pred``.
        """
        del acs_mask  # unused
        if measured_kspace is None or mask is None:
            return img_pred

        # Auto-detect k-space domain when not explicitly set.
        if not is_kspace_domain:
            is_kspace_domain = (
                not torch.is_complex(img_pred)
                and img_pred.ndim == 4
                and img_pred.shape[1] % 2 == 0
                and not torch.is_complex(measured_kspace)
                and measured_kspace.ndim == 4
                and measured_kspace.shape[1] == img_pred.shape[1]
            )

        is_complex_input = torch.is_complex(img_pred)

        # Convert prediction to complex.
        if not is_complex_input and img_pred.shape[1] % 2 == 0:
            x = torch.complex(img_pred[:, 0::2, ...], img_pred[:, 1::2, ...])
        else:
            x = _to_complex(img_pred)

        # Get k-space prediction.
        if is_kspace_domain:
            k_pred = x
        elif sensitivity_maps is not None:
            k_pred = self.mc_operator.forward(x, smaps=sensitivity_maps, mask=None)
        else:
            k_pred = fft2c(x)

        # Convert measured k-space to complex.
        if not torch.is_complex(measured_kspace):
            measured_kspace = torch.complex(
                measured_kspace[:, 0::2, ...], measured_kspace[:, 1::2, ...]
            )

        # KAN trust map is independent of coil identity; broadcast over channels.
        lambda_map = self._compute_lambda(measured_kspace)  # [B, 1, H, W]

        k_new = (1.0 - lambda_map) * k_pred + lambda_map * measured_kspace

        k_out = torch.where(align_dc_mask(mask, k_pred.shape[-3]).bool(), k_new, k_pred)

        # Convert back to output domain.
        if is_kspace_domain:
            if is_complex_input:
                return k_out
            if img_pred.shape[1] % 2 == 0:
                return torch.stack([k_out.real, k_out.imag], dim=2).flatten(1, 2)
            return k_out.real

        if sensitivity_maps is not None:
            x_out = self.mc_operator.adjoint(k_out, smaps=sensitivity_maps, combine_method="sense")
        else:
            x_out = ifft2c(k_out)

        if is_complex_input:
            return x_out
        if img_pred.shape[1] % 2 == 0:
            return torch.stack([x_out.real, x_out.imag], dim=2).flatten(1, 2)
        return x_out.real
