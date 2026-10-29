"""GeoMamba-UNet generator for ULF→HF MRI super-resolution.

Composes the existing infrastructure into the GeoMamba-ULF forward
path (plan §3):

.. math::

    \\hat I_{HF}^{(c)} = R\\!\\left(
        \\pi^{-1} \\circ f_\\theta^{(c)} \\circ \\pi (I_{LF}^{(c)})
    \\right)\\Big|_F

- :math:`\\pi`: :class:`MetricSFCLinearizer` permutation (foreground-only,
  intensity-aware).
- :math:`f_\\theta^{(c)}`: stack of :class:`FiLMMambaBlock` blocks with
  per-volume contrast embedding.
- :math:`\\pi^{-1}`: linearizer reverse with the input volume as the
  background fill (== identity-on-background re-embedding R).

The unwarp step :math:`\\varphi^{-1}` runs in the *strategy* (not in
the model) since it depends on per-subject field maps.

Token plumbing:

- Spatial → tokens: Conv3d-1×1 lifts ``(B, in_ch, D, H, W)`` to
  ``(B, d_model, D, H, W)``, then the linearizer permutes to
  ``(B, d_model, N_F)``, transposed to ``(B, N_F, d_model)`` for
  Mamba.
- Tokens → spatial: ``(B, N_F, d_model)`` → ``(B, d_model, N_F)`` →
  reverse permute (background filled from input embedding) →
  Conv3d-1×1 to ``out_ch``.

The model is *registered* with ``@register_model`` so the
strategy can resolve it via ``model.model_type = "geo_mamba_unet"``
in v6.0 YAML configs.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.film_mamba_block import FiLMMambaBlock
from spectramr.models.blocks.metric_sfc import MetricSFCLinearizer
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


def _distributed_world_size() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        from spectramr.core.parallel_groups import dp_world_size

        return dp_world_size()
    return 1


def _broadcast_from_rank0(tensor: torch.Tensor) -> torch.Tensor:
    """Broadcast from the data-parallel root; the global rank 0 on a trivial mesh."""
    import torch.distributed as dist

    from spectramr.core.parallel_groups import dp_group, dp_src_rank

    tensor = tensor.clone()
    dist.broadcast(tensor, src=dp_src_rank(), group=dp_group())
    return tensor


@register_model(
    name="geo_mamba_unet",
    training_mode="geomamba_ulf",
    supports_contrast_conditioning=True,
    spatial_dims=(3,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class GeoMambaUNet(nn.Module):
    """Foreground-restricted contrast-conditioned Mamba SR generator.

    Args:
        in_channels: number of input channels (typically 1 for
            magnitude or 2 for real/imag).
        out_channels: number of output channels (typically equal
            to in_channels for reconstruction).
        d_model: token embedding dimension (Mamba inner width).
        n_blocks: number of stacked FiLMMambaBlock blocks.
        contrast_dim: contrast-embedding dimension; the
            ``n_contrasts`` learnable rows are looked up by
            ``contrast_id`` at forward time.
        n_contrasts: number of distinct contrasts in the cohort.
        d_state: Mamba state dimension.
        d_conv: Mamba depthwise-conv kernel.
        expand: Mamba inner expansion factor.
        film_hidden_mult: hidden-layer multiplier in the FiLM MLP.
        dropout: residual dropout in each FiLM-Mamba block.
        linearizer: optional pre-built :class:`MetricSFCLinearizer`.
            If ``None``, the model defers construction to the first
            ``forward()`` call (when intensity_volume + mask are
            available via kwargs).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 64,
        n_blocks: int = 4,
        contrast_dim: int = 8,
        n_contrasts: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        film_hidden_mult: int = 4,
        dropout: float = 0.0,
        linearizer: MetricSFCLinearizer | None = None,
        sfc_beta: float = 10.0,
        sfc_smoothing_sigma: float = 1.0,
        sfc_connectivity: int = 26,
        sfc_cache_dir: str | None = "cache/metric_sfc",
        displacement_head: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.contrast_dim = contrast_dim
        self.n_contrasts = n_contrasts

        # Metric-SFC linearizer parameters (config.training.geomamba_ulf.metric_sfc).
        # Defaults mirror MetricSFCLinearizer's own defaults so an unset config
        # block is behaviour-preserving; the GeoMambaULFStrategy bridges the
        # config values onto these attributes at setup. Previously the lazy
        # linearizer build hard-coded the defaults, so the entire MetricSFCConfig
        # block (beta / smoothing_sigma / connectivity / cache_dir) was inert —
        # the abl_no_metric_sfc arm and the beta-HPO sweep were silent no-ops.
        self.sfc_beta = float(sfc_beta)
        self.sfc_smoothing_sigma = float(sfc_smoothing_sigma)
        if sfc_connectivity not in (6, 18, 26):
            raise ValueError(f"sfc_connectivity must be 6, 18, or 26; got {sfc_connectivity}")
        self.sfc_connectivity = int(sfc_connectivity)
        self.sfc_cache_dir = sfc_cache_dir

        self.input_proj = nn.Conv3d(in_channels, d_model, kernel_size=1)
        self.output_proj = nn.Conv3d(d_model, out_channels, kernel_size=1)
        # Optional geometric branch: a zero-initialised 1x1x1 head predicts a
        # normalised (x, y, z) displacement, and the output is resampled through it.
        self.displacement_head_enabled = bool(displacement_head)
        self.displacement_head: nn.Conv3d | None = None
        if self.displacement_head_enabled:
            self.displacement_head = nn.Conv3d(d_model, 3, kernel_size=1)
            nn.init.zeros_(self.displacement_head.weight)
            nn.init.zeros_(self.displacement_head.bias)
        # Set by forward: the displacement in voxel units, (B, 3, D, H, W) in (x, y, z)
        # order, or None when the head is off.
        self.last_displacement: torch.Tensor | None = None

        self.contrast_embedding = nn.Embedding(n_contrasts, contrast_dim)

        self.blocks = nn.ModuleList(
            [
                FiLMMambaBlock(
                    d_model=d_model,
                    contrast_dim=contrast_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    film_hidden_mult=film_hidden_mult,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            ]
        )

        # Optional pre-built linearizer (template-space mode). Strategy
        # may build and inject this at training-start so we pay MST cost
        # exactly once.
        # When missing, it is built ONCE from the first forward batch and
        # assigned here (see _resolve_linearizer) so all later forwards hit the
        # early-return — no per-step cache-key GPU sync.
        self._linearizer: MetricSFCLinearizer | None = linearizer
        # Restores a lazily built curve from a checkpoint before the strict load.
        self.register_load_state_dict_pre_hook(self._restore_linearizer_from_state)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        volume: torch.Tensor,
        mask: torch.Tensor | None = None,
        contrast_id: torch.Tensor | None = None,
        intensity_template: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through GeoMamba-UNet.

        Args:
            volume: ``(B, in_channels, D, H, W)`` distorted/unwarped
                input volume.
            mask: ``(B, 1, D, H, W)`` or ``(D, H, W)`` foreground
                mask. Required if no linearizer was injected at init.
            contrast_id: ``(B,)`` long tensor of contrast indices in
                ``[0, n_contrasts)``. Defaults to all-zeros.
            intensity_template: ``(D, H, W)`` reference intensity
                volume for the metric SFC. Used only when no
                linearizer is available; falls back to the
                first-batch mean if not provided.

        Returns:
            ``(B, out_channels, D, H, W)`` predicted volume with
            background voxels copied from ``volume`` unchanged.
        """
        if volume.ndim != 5:
            raise ValueError(f"Expected (B, C, D, H, W), got {tuple(volume.shape)}")
        B, C, D, H, W = volume.shape
        if self.in_channels != C:
            raise ValueError(f"Input has {C} channels; model was built for {self.in_channels}")

        linearizer = self._resolve_linearizer(volume, mask, intensity_template)

        if contrast_id is None:
            contrast_id = torch.zeros(B, dtype=torch.long, device=volume.device)
        contrast_emb = self.contrast_embedding(contrast_id)  # (B, contrast_dim)

        # Lift to token embedding.
        embedded = self.input_proj(volume)  # (B, d_model, D, H, W)

        # Permute to SFC tokens: (B, d_model, N_F) → (B, N_F, d_model).
        tokens = linearizer(embedded).transpose(1, 2)

        for block in self.blocks:
            tokens = tokens + block(tokens, contrast_emb)  # residual

        # Reverse permute back to (B, d_model, D, H, W); use the
        # input embedding as the background to keep non-foreground
        # information unchanged through the network.
        spatial_tokens = tokens.transpose(1, 2)  # (B, d_model, N_F)
        restored = linearizer.reverse(spatial_tokens, background=embedded)

        # Project to output channels.
        out = self.output_proj(restored)
        self.last_displacement = None
        if self.displacement_head is not None:
            out, self.last_displacement = self._warp_by_displacement(
                out, self.displacement_head(restored), (D, H, W)
            )

        # Final R: paste the original input back into background voxels
        # of the predicted volume, ensuring the network's reconstruction
        # is restricted to F.
        if mask is not None:
            mask_b = self._broadcast_mask(mask, B, D, H, W).to(out.dtype)
            out = mask_b * out + (1.0 - mask_b) * volume

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _warp_by_displacement(
        out: torch.Tensor, disp_norm: torch.Tensor, shape: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resample ``out`` through ``identity + disp``; return the warp and the displacement in voxels."""
        B = out.shape[0]
        D, H, W = shape
        zz, yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, D, device=out.device, dtype=out.dtype),
            torch.linspace(-1.0, 1.0, H, device=out.device, dtype=out.dtype),
            torch.linspace(-1.0, 1.0, W, device=out.device, dtype=out.dtype),
            indexing="ij",
        )
        identity = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)
        disp_grid = disp_norm.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3), (x, y, z)
        warped = F.grid_sample(
            out,
            identity + disp_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        # A normalised step of 2 spans (n - 1) voxels on each axis.
        scale = torch.tensor(
            [(W - 1) / 2.0, (H - 1) / 2.0, (D - 1) / 2.0], device=out.device, dtype=out.dtype
        ).view(1, 3, 1, 1, 1)
        return warped, disp_norm * scale

    def _resolve_linearizer(
        self,
        volume: torch.Tensor,
        mask: torch.Tensor | None,
        intensity_template: torch.Tensor | None,
    ) -> MetricSFCLinearizer:
        if self._linearizer is not None:
            return self._linearizer
        if mask is None:
            # Auto-generate a foreground mask from the input volume when no
            # explicit mask is supplied. The SFC linearizer needs a mask to
            # define which voxels participate in the traversal; for typical
            # MRI inputs background is ≈ 0 and any small positive threshold
            # cleanly separates anatomy from air. The earlier hard ``raise``
            # broke every GeoMamba-ULF training arm
            # (``geomamba_ulf_v0`` / ``v0_synthetic`` / ``v0_multicontrast`` /
            # ``v1_p2`` / ``v1_p3`` / 5 ablations / 3 specialists in the
            # 2026-05-05 cluster smoke) because the dataloader did not stash
            # a separate ``mask`` field. The cached linearizer below makes
            # this O(once-per-shape) — same cost as if a mask were provided.
            logger.info(
                "[GeoMambaUNet] no mask provided — auto-generating "
                "foreground mask from |volume| > 1e-3 (channel-mean)."
            )
            # ``volume`` is (B, C, D, H, W); reduce channels then threshold.
            vol_mag = volume.abs().mean(dim=1, keepdim=True)
            mask = (vol_mag > 1e-3).to(vol_mag.dtype)
        # Use the first sample of the batch as the template if no
        # explicit one was provided. Keeps the v0 path simple.
        template: torch.Tensor = (
            intensity_template if intensity_template is not None else volume[0, 0]
        )
        mask_3d = mask[0, 0] if mask.ndim == 5 else mask

        # Build ONCE from the first batch and keep it: the curve is a fixed
        # template-bound permutation, persisted with the weights (see
        # _install_linearizer) so a fresh model reloads the same order.
        if _distributed_world_size() > 1:
            # Every replica must share rank 0's curve.
            template = _broadcast_from_rank0(template.detach())
            mask_3d = _broadcast_from_rank0(mask_3d.detach())
        self._install_linearizer(template.detach(), mask_3d.detach(), device=volume.device)
        assert self._linearizer is not None
        logger.info(
            "[GeoMambaUNet] built lazy MetricSFCLinearizer (shape=%s, fg=%d)",
            self._linearizer.spatial_shape,
            self._linearizer.n_foreground,
        )
        return self._linearizer

    def _install_linearizer(
        self, template: torch.Tensor, mask_3d: torch.Tensor, *, device: torch.device
    ) -> None:
        """Build the curve, move its index buffers to ``device`` once, and persist its inputs."""
        linearizer = MetricSFCLinearizer(
            intensity_volume=template,
            mask=mask_3d,
            beta=self.sfc_beta,
            smoothing_sigma=self.sfc_smoothing_sigma,
            connectivity=cast(Literal[6, 18, 26], self.sfc_connectivity),
            cache_dir=self.sfc_cache_dir,
        ).to(device)
        # Persistent so state_dict carries the inputs needed to rebuild the curve.
        self.register_buffer("_linearizer_template", template.to(device).clone())
        self.register_buffer("_linearizer_mask", mask_3d.to(device).clone())
        self._linearizer = linearizer

    def _restore_linearizer_from_state(
        self, module: nn.Module, state_dict: dict[str, Any], prefix: str, *args: Any
    ) -> None:
        """Rebuild the stored curve before the strict load so its buffers match."""
        t_key = prefix + "_linearizer_template"
        if t_key not in state_dict:
            if self._linearizer is None and any(
                k.startswith(prefix + "_linearizer.") for k in state_dict
            ):
                raise ValueError(
                    "checkpoint carries MetricSFC buffers but no '_linearizer_template'; "
                    "it predates the persisted curve. Re-train or re-export it."
                )
            return
        # A checkpoint's curve wins over any injected linearizer.
        device = next(self.parameters()).device
        self._install_linearizer(
            state_dict[t_key].detach().to(device),
            state_dict[prefix + "_linearizer_mask"].detach().to(device),
            device=device,
        )

    @staticmethod
    def _broadcast_mask(mask: torch.Tensor, B: int, D: int, H: int, W: int) -> torch.Tensor:
        if mask.ndim == 3:
            return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, D, H, W)
        if mask.ndim == 4:  # (B, D, H, W)
            return mask.unsqueeze(1).expand(B, 1, D, H, W)
        if mask.ndim == 5:  # (B, 1, D, H, W) already
            return mask
        raise ValueError(
            f"mask must be 3D/4D/5D (D,H,W)|(B,D,H,W)|(B,1,D,H,W); got {tuple(mask.shape)}"
        )


__all__ = ["GeoMambaUNet"]
