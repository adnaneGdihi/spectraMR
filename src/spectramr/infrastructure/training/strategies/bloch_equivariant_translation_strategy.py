"""Bloch-equivariant cross-field translation — Idea 3.

Trains a translator ``B_θ`` (any registered ULF→HF generator) under
three branches:

1. **Supervised** — paired ULF/HF data with standard pixel-fidelity (L1).
2. **Unpaired Bloch-equivariant** — sample tissue parameters q from a
   parametric atlas, simulate ``F^B_ULF(q)`` and ``F^B_HF(q)`` via the
   differentiable Bloch operator, and minimise the residual
   ``‖B_θ(F^B_ULF(q)) − F^B_HF(q)‖²``.
3. **Cycle** — kept but downweighted by 10× (Eq. (1) is strictly
   stronger; see plan §Phase 3 Idea 3).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from spectramr.infrastructure.physics.differentiable_bloch import DifferentiableBlochLayer
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class _TissueAtlas:
    """Parametric mixture over white / grey matter, CSF, fat, muscle."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        # (M0, T1_ms, T2_ms) prototypes at 3 T (Stanisz et al. 2005-ish).
        self.prototypes = torch.tensor(
            [
                [0.85, 1100.0, 80.0],  # WM
                [0.95, 1500.0, 90.0],  # GM
                [1.00, 4000.0, 1800.0],  # CSF
                [0.70, 350.0, 70.0],  # fat
                [0.80, 900.0, 50.0],  # muscle
            ],
            dtype=torch.float32,
        )

    def sample(self, n: int, sigma: float = 0.05) -> torch.Tensor:
        """Sample n points from the mixture; returns [n, 3] (M0, T1, T2)."""
        idx = torch.randint(0, self.prototypes.shape[0], (n,))
        protos = self.prototypes[idx]
        noise = torch.randn_like(protos) * sigma * protos.abs()
        samples = (protos + noise).clamp(min=1e-3)
        # Enforce T2 ≤ T1.
        samples[..., 2] = torch.minimum(samples[..., 2], samples[..., 1])
        return samples.to(self.device)


class BlochEquivariantTranslationStrategy(BaseTrainingStrategy):
    """Cross-field translator with a Bloch-equivariance unpaired branch."""

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        bet = getattr(self.config.training, "bloch_equivariant_translation", None)
        # Field-strength dependent T1/T2 scalings — first-order approximation
        # to the Marques et al. (2019) literature.
        self.t1_scale_ulf = float(getattr(bet, "t1_scale_ulf", 0.30)) if bet else 0.30
        self.t2_scale_ulf = float(getattr(bet, "t2_scale_ulf", 1.05)) if bet else 1.05
        self.tr_ulf = float(getattr(bet, "tr_ulf", 8.0)) if bet else 8.0
        self.tr_hf = float(getattr(bet, "tr_hf", 8.0)) if bet else 8.0
        self.te_ulf = float(getattr(bet, "te_ulf", 4.0)) if bet else 4.0
        self.te_hf = float(getattr(bet, "te_hf", 4.0)) if bet else 4.0
        alpha = float(getattr(bet, "flip_angle_deg", 15.0)) if bet else 15.0
        self.flip_angle_rad = alpha * 3.14159265 / 180.0
        self.lambda_eq = float(getattr(bet, "lambda_bloch_eq", 1.0)) if bet else 1.0
        self.lambda_cycle = float(getattr(bet, "lambda_cycle", 0.1)) if bet else 0.1
        self.unpaired_patch = int(getattr(bet, "unpaired_patch", 32)) if bet else 32
        # audit_plan_novel.md Phase 3 follow-up: gradient checkpointing
        # through the Bloch forward operator. The differentiable Bloch
        # layer is optimised for forward-only use; for long pulse
        # trains backprop materialises an O(T) graph that can OOM. The
        # opt-in flag wraps each Bloch call in ``torch.utils.checkpoint``
        # to recompute activations during backward at ~2× compute cost.
        self.use_bloch_checkpoint = (
            bool(getattr(bet, "enable_bloch_checkpoint", False)) if bet else False
        )
        self.bloch = DifferentiableBlochLayer().to(self.device)
        self.atlas = _TissueAtlas(device=self.device)

    def _simulate_pair(self, n_voxels: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Simulate (ULF_image, HF_image) of shape [B=1, 1, H, W] from atlas."""
        H = self.unpaired_patch
        n = H * H
        q = self.atlas.sample(n).reshape(1, 1, H, H, 3)
        M0 = q[..., 0]
        T1 = q[..., 1]
        T2 = q[..., 2]

        def _ulf_forward(t1: torch.Tensor, t2: torch.Tensor, pd: torch.Tensor) -> torch.Tensor:
            return self.bloch(
                t1_map=(t1 * self.t1_scale_ulf).contiguous(),
                t2_map=(t2 * self.t2_scale_ulf).contiguous(),
                pd_map=pd,
                tr=self.tr_ulf,
                te=self.te_ulf,
                flip_angle_rad=torch.tensor(self.flip_angle_rad, device=self.device),
                b0_map=None,
                b1_map=None,
            )

        def _hf_forward(t1: torch.Tensor, t2: torch.Tensor, pd: torch.Tensor) -> torch.Tensor:
            return self.bloch(
                t1_map=t1,
                t2_map=t2,
                pd_map=pd,
                tr=self.tr_hf,
                te=self.te_hf,
                flip_angle_rad=torch.tensor(self.flip_angle_rad, device=self.device),
                b0_map=None,
                b1_map=None,
            )

        if self.use_bloch_checkpoint and torch.is_grad_enabled():
            ulf_img = torch_ckpt.checkpoint(_ulf_forward, T1, T2, M0, use_reentrant=False)
            hf_img = torch_ckpt.checkpoint(_hf_forward, T1, T2, M0, use_reentrant=False)
        else:
            ulf_img = _ulf_forward(T1, T2, M0)
            hf_img = _hf_forward(T1, T2, M0)
        return ulf_img.real if torch.is_complex(ulf_img) else ulf_img, (
            hf_img.real if torch.is_complex(hf_img) else hf_img
        )

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        # Supervised paired branch.
        hf_pred = self.generator_model(input_batch)
        supervised = F.l1_loss(hf_pred, target_batch)
        # Unpaired Bloch-equivariant branch.
        with torch.no_grad():
            ulf_sim, hf_sim = self._simulate_pair(self.unpaired_patch**2)
        hf_predicted_from_sim = self.generator_model(ulf_sim.to(input_batch.device))
        bloch_eq = F.mse_loss(hf_predicted_from_sim, hf_sim.to(input_batch.device))
        # Downweighted cycle branch (placeholder: identity through the same model).
        cycle = F.l1_loss(self.generator_model(target_batch), target_batch)
        total = supervised + self.lambda_eq * bloch_eq + self.lambda_cycle * cycle
        return {
            "g_total_loss": total,
            "bloch_eq_supervised": supervised.detach(),
            "bloch_eq_residual": bloch_eq.detach(),
            "bloch_eq_cycle": cycle.detach(),
        }


__all__ = ["BlochEquivariantTranslationStrategy"]
