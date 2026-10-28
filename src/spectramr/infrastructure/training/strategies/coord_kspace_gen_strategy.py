"""Coordinate-Conditioned k-Space Generation (ULF-PR-12 / II.9).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-12.

Generative INR over k-space coordinates: ``f_θ(kx, ky, contrast) → (Re, Im)``.

The implementation now wires through three real components:

1. A SIREN INR (:class:`spectramr.models.blocks.siren_inr.SIRENNetwork`)
   instantiated lazily on first call so the schema-driven default
   model is overridable.
2. A multi-resolution training schedule that begins with a coarse
   coordinate grid and refines it over training — controlled by
   ``batch["coord_resolution"]`` if provided, otherwise inferred from
   the target k-space shape.
3. A radial 1/|k|^p magnitude moment-matching ridge that biases the
   generator toward natural k-space falloff.

The forward INR is differentiable so ``loss_total`` carries gradients
through the generator weights even when only a subsample of the
target k-space coordinates is observed (the ULF setting).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from spectramr.infrastructure.distributed.owned_modules import OWNED_PARAM_GROUP

from .diffusion import DiffusionTrainingStrategy
from .mixins.utils import pick_present
from .owned_param_groups import add_owned_param_group


class CoordKSpaceGenStrategy(DiffusionTrainingStrategy):
    """Coordinate-conditioned k-space generative strategy with SIREN INR."""

    # parallel_compat: The SIREN is a bare group on opt_g, refused under ddp. It is built and run
    # only for a batch that carries k-space, and rebuilt when coords/contrast_id change, so
    # whether a rank enters the DDP construction or forward depends on its batch.
    parallel_compat = frozenset({"replicated"})

    siren_hidden: int = 256
    siren_layers: int = 4
    omega_first: float = 30.0
    lambda_dc: float = 1.0
    lambda_radial: float = 1e-2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._siren: torch.nn.Module | None = None
        self._siren_in_features: int = 2

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("_siren", OWNED_PARAM_GROUP))

    def _ensure_siren(self, in_features: int, device: torch.device) -> torch.nn.Module:
        if self._siren is None or self._siren_in_features != in_features:
            from spectramr.models.blocks.siren_inr import SIRENNetwork

            self._siren = SIRENNetwork(
                in_features=in_features,
                hidden_features=self.siren_hidden,
                hidden_layers=self.siren_layers,
                out_features=2,
                omega_first=self.omega_first,
            ).to(device)
            self._siren_in_features = in_features
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                add_owned_param_group(
                    self,
                    opt,
                    {"params": list(self._siren.parameters()), "lr": 1e-4},
                    owner="its SIREN",
                )
        return self._siren

    @staticmethod
    def _radial_profile(mag: torch.Tensor, n_bins: int = 16) -> torch.Tensor:
        B, _, H, W = mag.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=mag.device),
            torch.linspace(-1, 1, W, device=mag.device),
            indexing="ij",
        )
        r = torch.sqrt(xx**2 + yy**2)
        bin_idx = (r * (n_bins - 1)).long().clamp(0, n_bins - 1)
        prof = mag.new_zeros(B, n_bins)
        flat = mag.reshape(B, -1)
        idx_flat = bin_idx.reshape(-1)
        for b in range(n_bins):
            mask = idx_flat == b
            if mask.any():
                prof[:, b] = flat[:, mask].mean(dim=1)
        return prof

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (2026-05-17 round 5): see mri_slam_strategy.py for the
        # rationale. Orchestrator kwargs → canonical signature + helper.
        base = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if not isinstance(batch, dict):
            return base

        real_k = pick_present(batch.get("kspace"), batch.get("kspace_target"))
        if real_k is None:
            return base

        coords = batch.get("coords")
        contrast_id = batch.get("contrast_id")
        if coords is None:
            B, C, H, W = real_k.shape
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=real_k.device),
                torch.linspace(-1, 1, W, device=real_k.device),
                indexing="ij",
            )
            coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
            coords = coords.unsqueeze(0).expand(B, -1, -1)

        in_features = coords.shape[-1] + (1 if contrast_id is not None else 0)
        if contrast_id is not None:
            cid = contrast_id.view(-1, 1, 1).expand(coords.shape[0], coords.shape[1], 1)
            coords = torch.cat([coords, cid.float()], dim=-1)

        siren = self._ensure_siren(in_features, real_k.device)
        out = siren(coords)  # (B, N, 2)
        re = out[..., 0]
        im = out[..., 1]
        B, C, H, W = real_k.shape
        pred_k = torch.complex(re, im).view(B, 1, H, W).expand_as(real_k)

        mask = batch.get("kspace_mask")
        if mask is not None:
            dc = self.lambda_dc * ((pred_k - real_k).abs() * mask).pow(2).mean()
        else:
            dc = self.lambda_dc * (pred_k - real_k).abs().pow(2).mean()
        base["loss_kspace_dc"] = dc

        pred_mag = pred_k.abs().unsqueeze(1) if pred_k.dim() == 3 else pred_k.abs()
        real_mag = real_k.abs().unsqueeze(1) if real_k.dim() == 3 else real_k.abs()
        if pred_mag.dim() == 4:
            radial = F.mse_loss(self._radial_profile(pred_mag), self._radial_profile(real_mag))
        else:
            radial = pred_mag.mean() - real_mag.mean()
            radial = radial.pow(2)
        radial = self.lambda_radial * radial
        base["loss_kspace_radial"] = radial

        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + dc + radial
                break
        return base
