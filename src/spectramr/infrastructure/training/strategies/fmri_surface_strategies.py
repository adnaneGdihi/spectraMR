"""fMRI surface / connectivity / HRF strategies — §§3, 4, 5.

* :class:`CorticalConformalFMRIReconStrategy`
  (``cortical_conformal_fmri_recon``) — fMRI §3. Reconstructs BOLD in
  conformally-flattened cortical coordinates. Pre-computed flattening
  grid is supplied via the batch; absent it, the strategy degenerates
  to a vanilla L1 reconstruction with a Laplacian-difference regulariser.
* :class:`RiemannianDFCDiffusionStrategy`
  (``riemannian_dfc_diffusion``) — fMRI §4. Riemannian DSM on the SPD
  manifold of FC matrices. Score is provided by ``tangent_spd_diffusion``.
* :class:`HRFManifoldDiffusionStrategy`
  (``hrf_manifold_diffusion``) — fMRI §5. 1-D denoising-score-matching
  diffusion on HRF time courses. (A Beltrami time-reparameterisation
  anchor term was removed: it was driven by a constant-zero input and so
  was identically zero with no gradient — see the audit 2026-06 note.)

References:
    [14] Gu et al., "Genus zero surface conformal mapping",
         *IEEE Trans. Med. Imag.*, 23(8), 2004.
    [18] Pennec et al., *Int. J. Comput. Vis.*, 66(1), 2006.
    [26] Friston et al., "Nonlinear responses in fMRI: Balloon model",
         *NeuroImage*, 12(4), 2000.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.infrastructure.physics.manifolds.spd_manifold import (
    SPDManifold,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class CorticalConformalFMRIReconStrategy(BaseTrainingStrategy):
    """Surface-based BOLD reconstruction in conformal cortical coordinates."""

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        # Typed sub-block (pitfall #15); the mount coerces a YAML mapping to the
        # validated block, so this is now a real typed read (was a silent
        # getattr-against-dict default).
        cfg = getattr(self.config.training, "cortical_conformal_fmri_recon", None)
        self.lambda_curv = float(cfg.lambda_curvature) if cfg is not None else 0.1
        logger.info(
            "CorticalConformalFMRIReconStrategy (source=%s): lambda_curvature=%.4g",
            "config" if cfg is not None else "defaults",
            self.lambda_curv,
        )

    @staticmethod
    def _disk_laplacian(x: torch.Tensor) -> torch.Tensor:
        kernel = x.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        kernel = kernel.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        batch = kwargs.get("batch", {})
        flatten_grid = batch.get("cortex_flatten_grid")
        if isinstance(flatten_grid, torch.Tensor):
            flatten_grid = flatten_grid.to(input_batch.device)
            x_disk = F.grid_sample(
                input_batch,
                flatten_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            y_disk = F.grid_sample(
                target_batch,
                flatten_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        else:
            x_disk, y_disk = input_batch, target_batch
        recon = self.generator_model(x_disk)
        recon_loss = F.l1_loss(recon, y_disk)
        u = batch.get("conformal_factor")
        if isinstance(u, torch.Tensor):
            u = u.to(input_batch.device)
            curv = (
                (self._disk_laplacian(recon) - torch.exp(-2.0 * u) * self._disk_laplacian(y_disk))
                .pow(2)
                .mean()
            )
        else:
            curv = (self._disk_laplacian(recon) - self._disk_laplacian(y_disk)).pow(2).mean()
        total = recon_loss + self.lambda_curv * curv
        return {
            "g_total_loss": total,
            "cortical_fmri_recon": recon_loss.detach(),
            "cortical_fmri_conformal_curvature": curv.detach(),
        }


class RiemannianDFCDiffusionStrategy(BaseTrainingStrategy):
    """Riemannian DSM on Sym⁺(n) for dynamic-FC generation.

    Caller supplies a batch of clean FC matrices via ``input_batch``
    (shape ``[B, n, n]`` SPD). The forward noise perturbation is
    performed in the affine-invariant exp map; the loss is the
    Frobenius residual between the predicted tangent score and the
    chart-space target tangent.
    """

    # Computes its own objective and reads no `losses:` block, so an empty builder stack
    # is the design (LossBuilder.validate) and any declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    # Debug-snapshot contract (non-negotiable 14): the model is fed a noised
    # tensor, not the prepared input; declared here and emitted under the tag.
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "riemannian_dfc_diffusion", None)
        self.sigma_min = float(cfg.sigma_min) if cfg is not None else 1e-2
        self.sigma_max = float(cfg.sigma_max) if cfg is not None else 1.0
        logger.info(
            "RiemannianDFCDiffusionStrategy (source=%s): sigma_min=%.4g, sigma_max=%.4g",
            "config" if cfg is not None else "defaults",
            self.sigma_min,
            self.sigma_max,
        )

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        if input_batch.dim() < 3:
            raise ValueError("RiemannianDFCDiffusionStrategy expects SPD [B, n, n] input")
        C_clean = input_batch
        n = C_clean.shape[-1]
        device = C_clean.device
        manifold = SPDManifold(n=n)
        b = C_clean.shape[0]
        log_lo, log_hi = (
            torch.tensor(self.sigma_min, device=device).log(),
            torch.tensor(self.sigma_max, device=device).log(),
        )
        sigma = (log_lo + torch.rand(b, device=device) * (log_hi - log_lo)).exp()
        # Sample a symmetric noise tangent, exp-map it onto the manifold.
        noise = torch.randn_like(C_clean)
        noise = 0.5 * (noise + noise.transpose(-1, -2)) * sigma.view(-1, 1, 1)
        C_noisy = manifold.exp_map(C_clean, noise)
        # (Removed a dead ``use_cayley`` knob that gated a discarded
        # ``cayley_transform(C_noisy)`` whose result never entered the score or
        # loss — flipping it changed nothing, pitfall #15.)
        # Generator returns a symmetric tangent score.
        self._declare_model_input({"C_noisy": C_noisy, "sigma": sigma}, extra={"model_input_key": "C_noisy"})
        score = self.generator_model(C_noisy, sigma)
        target_score = -noise / sigma.view(-1, 1, 1).clamp_min(1e-6)
        loss = F.mse_loss(score, target_score)
        return {
            "g_total_loss": loss,
            "riemannian_dfc_dsm": loss.detach(),
        }


class HRFManifoldDiffusionStrategy(BaseTrainingStrategy):
    """1-D diffusion over the HRF function manifold."""

    # Computes its own objective and reads no `losses:` block, so an empty builder stack
    # is the design (LossBuilder.validate) and any declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    # Debug-snapshot contract (non-negotiable 14): the model is fed a noised
    # tensor, not the prepared input; declared here and emitted under the tag.
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "hrf_manifold_diffusion", None)
        self.sigma_min = float(cfg.sigma_min) if cfg is not None else 1e-2
        self.sigma_max = float(cfg.sigma_max) if cfg is not None else 1.0
        logger.info(
            "HRFManifoldDiffusionStrategy (source=%s): sigma_min=%.4g, sigma_max=%.4g",
            "config" if cfg is not None else "defaults",
            self.sigma_min,
            self.sigma_max,
        )

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        # input_batch[B, T] is a noisy HRF observation, target a clean canonical HRF.
        device = input_batch.device
        b = input_batch.shape[0]
        log_lo, log_hi = (
            torch.tensor(self.sigma_min, device=device).log(),
            torch.tensor(self.sigma_max, device=device).log(),
        )
        sigma = (log_lo + torch.rand(b, device=device) * (log_hi - log_lo)).exp()
        noise = torch.randn_like(target_batch)
        h_noisy = target_batch + sigma.view(-1, 1) * noise
        self._declare_model_input({"h_noisy": h_noisy, "sigma": sigma}, extra={"model_input_key": "h_noisy"})
        score = self.generator_model(h_noisy, sigma)
        if score.dim() == 3:
            score = score.squeeze(1)
        target_score = -noise / sigma.view(-1, 1).clamp_min(1e-6)
        dsm = F.mse_loss(score, target_score)
        # (Removed a dead Beltrami ``tau_reg`` term: it built ``raw =
        # torch.zeros(...)`` so the constraint output was a non-learnable
        # constant and ``tau_reg`` was identically 0 with no gradient path —
        # ``1e-3 * 0`` contributed nothing. The ``tau_eps`` knob that fed it was
        # likewise a no-op. pitfall #15 / dead loss term.)
        return {
            "g_total_loss": dsm,
            "hrf_diffusion_dsm": dsm.detach(),
        }


__all__ = [
    "CorticalConformalFMRIReconStrategy",
    "HRFManifoldDiffusionStrategy",
    "RiemannianDFCDiffusionStrategy",
]
