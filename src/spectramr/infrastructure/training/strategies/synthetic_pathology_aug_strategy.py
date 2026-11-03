"""Synthetic Pathology Augmentation Strategy (Integration plan idea 7).

Plan: TODO/integration_plan_ulf_cheap_fast_mri.md §7.

On-the-fly synthetic lesion injection plus a region-weighted loss.  We
no longer rely on an external transform — the lesion sampler is in
this strategy:

- :class:`LesionSampler` produces (B, 1, H, W) Gaussian-blob masks
  with random center / radius / intensity per sample.
- The strategy draws ONE blob per sample and composites it onto the input
  *and* the target, with a per-sample random toggle so a fraction of
  samples remain clean.
- Loss = parent recon fidelity + lesion-region-weighted L1 + an auxiliary
  lesion-presence classifier on the prediction, built by the strategy when
  ``lambda_lesion_cls > 0``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixins.utils import pick_present
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class LesionSampler(nn.Module):
    """Random Gaussian-blob lesion sampler — differentiable."""

    def __init__(
        self, max_radius: float = 0.15, intensity_range: tuple[float, float] = (0.3, 1.2)
    ) -> None:
        super().__init__()
        self.max_radius = max_radius
        self.intensity_range = intensity_range

    def draw(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One blob per sample: ``(intensity * soft_mask, binary_mask)``, each (B, 1, H, W)."""
        B, _, H, W = image.shape
        device = image.device
        cx = torch.rand(B, device=device) * 0.6 - 0.3
        cy = torch.rand(B, device=device) * 0.6 - 0.3
        radius = torch.rand(B, device=device) * (self.max_radius - 0.03) + 0.03
        intensity = (
            torch.rand(B, device=device) * (self.intensity_range[1] - self.intensity_range[0])
            + self.intensity_range[0]
        )

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        cx_b = cx.view(-1, 1, 1, 1)
        cy_b = cy.view(-1, 1, 1, 1)
        r_b = radius.view(-1, 1, 1, 1)
        d2 = (xx - cx_b) ** 2 + (yy - cy_b) ** 2
        mask = torch.exp(-d2 / r_b.pow(2).clamp_min(1e-4))
        return intensity.view(-1, 1, 1, 1) * mask, (mask > 0.4).float()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blob, mask = self.draw(image)
        return image + blob, mask


def build_lesion_presence_head(in_channels: int) -> nn.Module:
    """Small strided-conv classifier: (B, C, H, W) image -> (B, 1) lesion-presence logit."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 1),
    )


_LESION_HEAD = "lesion_cls_head"


class SyntheticPathologyAugStrategy(ReconstructionTrainingStrategy):
    """Recon with on-the-fly synthetic lesion augmentation."""

    p_augment: float = 0.5
    lambda_lesion_l1: float = 0.5
    lambda_lesion_cls: float = 0.1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sampler: LesionSampler | None = None
        self.lesion_cls_head: nn.Module | None = None
        # Typed sub-block (pitfall #15: read + validate + stamp). Previously the
        # augmentation probability + lesion-loss weights were hardcoded class
        # attributes that no YAML could override.
        cfg = getattr(self.config.training, "synthetic_pathology_aug", None)
        if cfg is not None:
            self.p_augment = float(cfg.p_augment)
            self.lambda_lesion_l1 = float(cfg.lambda_lesion_l1)
            self.lambda_lesion_cls = float(cfg.lambda_lesion_cls)
        logger.info(
            "SyntheticPathologyAugStrategy (source=%s): p_augment=%.3g, "
            "lambda_lesion_l1=%.4g, lambda_lesion_cls=%.4g",
            "config" if cfg is not None else "defaults",
            self.p_augment,
            self.lambda_lesion_l1,
            self.lambda_lesion_cls,
        )
        if self.lambda_lesion_cls > 0:
            self._build_lesion_head()

    def _build_lesion_head(self) -> None:
        """Build the lesion-presence head on the generator output and train it with opt_g."""
        opt = getattr(self.env, "opt_g", None)
        if opt is None:
            raise RuntimeError(
                "synthetic_pathology_aug: lambda_lesion_cls > 0 needs env.opt_g to train the "
                "lesion-presence head; set lambda_lesion_cls: 0 to disable the classifier."
            )
        head = build_lesion_presence_head(int(self.config.model.out_channels)).to(self.device)
        self.lesion_cls_head = self.register_owned_module(_LESION_HEAD, head)
        existing = {p for group in opt.param_groups for p in group["params"]}
        new = [p for p in head.parameters() if p not in existing]
        if new:
            lr = float(opt.param_groups[0]["lr"])
            opt.add_param_group({"params": new, "lr": lr})
            self._sync_schedulers_after_param_group_addition(lr)

    def _ensure_sampler(self, device: torch.device) -> LesionSampler:
        if self._sampler is None:
            self._sampler = LesionSampler().to(device)
        return self._sampler

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # See TODO/audit/smoke_audit_20260516.md §F6.
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if isinstance(batch, dict) and "input" in batch and "target" in batch:
            inp, tgt = batch["input"], batch["target"]
            if inp.shape[0] != tgt.shape[0] or inp.shape[-2:] != tgt.shape[-2:]:
                raise ValueError(
                    "synthetic_pathology_aug: input and target must share batch and spatial "
                    f"shape to carry the same lesion; got {tuple(inp.shape)} vs {tuple(tgt.shape)}."
                )
            sampler = self._ensure_sampler(inp.device)
            apply = torch.rand(inp.shape[0], device=inp.device) < self.p_augment
            # One draw composited onto both sides: a target-only lesion is not inferable.
            blob, lesion_mask = sampler.draw(inp)
            gate = apply.view(-1, 1, 1, 1).to(inp.dtype)
            gated_blob = gate * blob
            patched = dict(batch)
            patched["input"] = inp + gated_blob
            patched["target"] = tgt + gated_blob
            patched["lesion_mask"] = lesion_mask * gate
            patched["lesion_present"] = apply.to(inp.dtype)
            # The composited tensors are the model input/target (the parent builds lr/hr from these).
            base = super()._compute_losses_impl(
                input_batch=patched["input"],
                target_batch=patched["target"],
                epoch=epoch,
                **{**kwargs, "batch": patched},
            )
        else:
            base = super()._compute_losses_impl(
                input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
            )
            if not isinstance(batch, dict):
                return base
            patched = batch

        lesion_mask = patched.get("lesion_mask")
        # The model prediction is never written into the batch dict — read it from
        # the parent's prediction seam (set in
        # ReconstructionTrainingStrategy._compute_losses_impl). pred carries grad to
        # the generator (NOT detached), so the region-weighted term backprops.
        pred = getattr(self, "_last_prediction", None)
        # Prefer the parent's loss target (matches what pred was scored against);
        # fall back to the composited batch target. pick_present is the SSOT for
        # first-non-None when operands may be tensors.
        target = pick_present(getattr(self, "_last_target", None), patched.get("target"))
        if lesion_mask is not None and pred is not None and target is not None:
            weighted = ((pred - target).abs() * (1.0 + 4.0 * lesion_mask)).mean()
            base["loss_lesion_weighted"] = self.lambda_lesion_l1 * weighted

            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base:
                    base[total_key] = base[total_key] + base["loss_lesion_weighted"]
                    break
            else:
                raise KeyError(
                    "synthetic_pathology: no total-loss key (loss_total / "
                    "g_total_loss) present to fold in loss_lesion_weighted — the "
                    "lesion term would be silently dropped (audit 2026-06)."
                )

        label = patched.get("lesion_present")
        if self.lambda_lesion_cls > 0 and label is not None and pred is not None:
            if self.lesion_cls_head is None:
                raise RuntimeError(
                    "synthetic_pathology_aug: lambda_lesion_cls > 0 but the lesion-presence "
                    "head was never built; the classifier term would be silently dropped."
                )
            logits = self.call_owned(_LESION_HEAD, pred)
            cls = F.binary_cross_entropy_with_logits(logits.flatten(), label.flatten())
            base["loss_lesion_cls"] = self.lambda_lesion_cls * cls
            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base:
                    base[total_key] = base[total_key] + base["loss_lesion_cls"]
                    break
            else:
                raise KeyError(
                    "synthetic_pathology: no total-loss key (loss_total / "
                    "g_total_loss) present to fold in loss_lesion_cls — the "
                    "lesion-classifier term would be silently dropped."
                )
        return base
