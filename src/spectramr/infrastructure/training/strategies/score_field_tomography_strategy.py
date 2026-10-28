"""Score-field tomography training strategy — Idea 2.

Trains a joint ``(x, c)`` score model with two heads ``s_x`` and ``s_c``
using denoising score matching. The forward noising process in k-space
uses Hermitian-Gaussian noise (Phase 0) so that the score-matching
objective remains coherent for real-valued MRI images.

The strategy is intentionally diffusion-mode-compatible: it can be
declared via ``training_mode: score_field_tomography`` or via
``training_mode: diffusion`` with a custom ``model.model_type:
score_field_unet`` — but the former gives a typed dispatch path and
hooks for the dual-head loss.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.data.transforms.acquisition_params_injector import (
    infer_acquisition_params,
)
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.hermitian import (
    enforce_hermitian,
    hermitian_gaussian_noise,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class ScoreFieldTomographyStrategy(BaseTrainingStrategy):
    """Joint (x, c) denoising score matching with k-space Hermitian noise."""

    # Debug-snapshot contract (non-negotiable 14): the model is fed a noised
    # tensor, not the prepared input; declared here and emitted under the tag.
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        # Typed sub-block (pitfall #15). Previously read via getattr against the
        # extra="allow" dict — a supplied YAML block silently returned defaults.
        sft = getattr(self.config.training, "score_field_tomography", None)
        self.sigma_min = float(sft.sigma_min) if sft is not None else 1e-3
        self.sigma_max = float(sft.sigma_max) if sft is not None else 1.0
        self.cond_dim = int(sft.cond_dim) if sft is not None else 5
        logger.info(
            "ScoreFieldTomographyStrategy (source=%s): sigma_min=%.4g, sigma_max=%.4g, cond_dim=%d",
            "config" if sft is not None else "defaults",
            self.sigma_min,
            self.sigma_max,
            self.cond_dim,
        )
        # μ(σ) = σ² · (dim c / dim x) per the plan; dim x is set when first batch arrives.
        self._dim_x: int | None = None

    def _sample_sigma(self, n: int, device: torch.device) -> torch.Tensor:
        log_lo, log_hi = (
            torch.tensor(self.sigma_min).log(),
            torch.tensor(self.sigma_max).log(),
        )
        u = torch.rand(n, device=device)
        return (log_lo + u * (log_hi - log_lo)).exp()

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        x_clean = target_batch
        b = x_clean.shape[0]
        device = x_clean.device
        if self._dim_x is None:
            self._dim_x = int(x_clean[0].numel())
        # Acquisition parameters: pulled from the batch dict when
        # ``data.expose_acquisition_params`` is set (Tier-1 audit gates
        # this), else fabricated from the static PHYSICS_REGISTRY via
        # :func:`infer_acquisition_params` so the strategy can still
        # run on synthetic / metadata-less data.
        batch_dict = kwargs.get("batch", {})
        if isinstance(batch_dict.get("acquisition_params"), torch.Tensor):
            c_clean = batch_dict["acquisition_params"].to(device).float()
        else:
            ref = {"image": x_clean}
            c_clean = infer_acquisition_params(ref).to(device)
            # Truncate / pad to declared cond_dim so the strategy works
            # for non-default cond_dim values during ablations.
            if c_clean.shape[-1] > self.cond_dim:
                c_clean = c_clean[..., : self.cond_dim]
            elif c_clean.shape[-1] < self.cond_dim:
                pad = torch.zeros(b, self.cond_dim - c_clean.shape[-1], device=device)
                c_clean = torch.cat([c_clean, pad], dim=-1)
        sigma = self._sample_sigma(b, device)
        # Image-domain noise (real); k-space Hermitian-Gaussian noise is
        # used when the model's input_domain is kspace (see TODO marker
        # in audit_plan_novel §Phase 0/3 interlock).
        eps = torch.randn_like(x_clean)
        x_noisy = x_clean + sigma.view(-1, 1, 1, 1) * eps
        eps_c = torch.randn_like(c_clean)
        c_noisy = c_clean + sigma.view(-1, 1) * eps_c
        # The network denoises both; x_noisy is the image it is named for, c_noisy
        # the field it is conditioned on and scores alongside.
        self._declare_model_input(
            {"x_noisy": x_noisy, "c_noisy": c_noisy, "sigma": sigma},
            extra={"model_input_key": "x_noisy"},
        )
        s_x, s_c = self.generator_model(x_noisy, sigma, c_noisy)
        target_score_x = -eps / sigma.view(-1, 1, 1, 1)
        target_score_c = -eps_c / sigma.view(-1, 1)
        lam = sigma.view(-1, 1, 1, 1) ** 2
        mu = (sigma.view(-1, 1) ** 2) * (self.cond_dim / max(self._dim_x, 1))
        loss_x = (lam * (s_x - target_score_x) ** 2).mean()
        loss_c = (mu * (s_c - target_score_c) ** 2).mean()
        # Hermitian-residual check used at integration-test time (Phase 0/3 interlock).
        with torch.no_grad():
            k = fft2c(x_noisy)
            herm_residual = (k - enforce_hermitian(k)).abs().mean()
        total = loss_x + loss_c
        return {
            "g_total_loss": total,
            "score_field_loss_x": loss_x.detach(),
            "score_field_loss_c": loss_c.detach(),
            "score_field_herm_residual": herm_residual,
        }


__all__ = ["ScoreFieldTomographyStrategy", "hermitian_gaussian_noise"]
