"""Marker-Anchored Virtual Fiducial Reconstruction Strategy.

Implements the Marker-Anchored Digital Twin Pipeline as a training strategy:
the generator acts as a learned proximal regulariser and a marker-prior
projection anchors the fiducial voxels in a single forward pass. This is a
proximal/prior-projection step, not a multi-iteration ADMM unroll (no data
consistency loop, no μ/ρ dual variables exist in this strategy).

Pipeline
--------
1. **Probe**: Digital Twin embeds marker + applies physics degradation
2. **Extract**: B0/B1/PSF estimated from marker observations
3. **Construct**: Build operator chain from extracted parameters
4. **Regularise**: Generator (learned proximal) + marker-prior projection
   anchoring the fiducial voxels

All simulator parameters come from ``config.physics.digital_twin`` (SSOT);
loss weights come from ``config.losses.reconstruction`` (SSOT).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.physics.vf_operators import MarkerPriorProjection
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.ood_acceleration_readout import (
    ood_acceleration_readout,
    ood_accelerations,
)
from spectramr.infrastructure.training.strategies.simulator_builder import (
    build_simulator_from_config,
    undersampling_mask_kwargs,
)
from spectramr.models.blocks.vf_modules import NoiseVarianceEstimator
from spectramr.models.losses.registry import create_loss

logger = logging.getLogger(__name__)


class ConcreteVFADMMStrategy(BaseTrainingStrategy):
    """Single-pass reconstruction with marker-anchored prior projection.

    Per training step the loss path runs:
    - Step 1: Neural regularisation (generator as learned proximal map)
    - Step 2: Prior projection P(z) anchors marker voxels

    There is no data-consistency loop or multi-iteration ADMM unroll here;
    the marker prior is enforced via projection plus the marker/prior losses.

    Attributes:
        simulator: Digital Twin forward physics model (config-driven).
        noise_estimator: Marker-based noise variance estimator (monitoring).
    """

    #: The twin undersamples at ``physics.digital_twin.acceleration`` when its
    #: ``enable_undersampling`` is true; the top-level ``undersampling:`` block
    #: reaches nothing here (the k-space mixin's generator is never called), so
    #: this strategy does NOT claim ``applies_undersampling`` (VF review
    #: 2026-09-03). ``physics.digital_twin.ood_acceleration_range`` is read by
    #: ``validation_step`` (``ood_acceleration_readout``).
    reads_ood_acceleration_range = True

    def __init__(
        self,
        env: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize Digital Twin from config, operator chain, and losses."""
        # ── Build simulator from config.physics.digital_twin (SSOT) ──
        self.simulator = build_simulator_from_config(self.config, self.device)

        # ── Setup Unified Reconstruction Loss Computer (V6.0 Compliance) ──
        from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
            UnifiedReconstructionLossComputer,
        )

        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config,
            device=self.device,
        )

        # ── Loss weights from config (SSOT) ──
        recon = self.config.losses.reconstruction
        self._lambda_prior = recon.lambda_prior
        self._lambda_marker = recon.lambda_marker

        # ── VF-specific losses (SSOT via registry; weights come from config) ──
        self.loss_marker_prior = create_loss("marker_prior", lambda_prior=self._lambda_prior).to(
            self.device
        )
        self.loss_marker = create_loss("marker_corruption").to(self.device)

        # Noise variance estimator
        self.noise_estimator = NoiseVarianceEstimator()

        logger.info(
            "[VF-ADMM] λ_marker=%.2f, λ_prior=%.2f",
            self._lambda_marker,
            self._lambda_prior,
        )

    @staticmethod
    def _normalize_to_range(
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize complex tensor magnitude for [-1, 1] bounded variation."""
        if not tensor.is_complex():
            # Fallback for real tensors
            t_min = tensor.min()
            t_max = tensor.max()
            t_range = t_max - t_min + 1e-8
            return 2.0 * (tensor - t_min) / t_range - 1.0, t_min, t_max

        # For native complex tensors, divide by the max magnitude
        t_max_mag = tensor.abs().max().clamp(min=1e-8)
        zero = torch.zeros(1, device=tensor.device)
        return tensor / t_max_mag, zero, t_max_mag

    def _to_complex(self, batch: torch.Tensor) -> torch.Tensor:
        """Convert real-stacked tensor to complex."""
        if torch.is_complex(batch):
            return batch
        C = batch.shape[1]
        if C >= 2 and C % 2 == 0:
            return torch.complex(batch[:, 0::2], batch[:, 1::2])
        return batch.to(torch.complex64)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute ADMM-based reconstruction losses.

        Args:
            input_batch: Unused (we corrupt target internally).
            target_batch: Clean ground truth [B, C, H, W].
            epoch: Current epoch.

        Returns:
            Loss dict with 'g_total_loss' and components.
        """
        if target_batch.ndim == 5:
            mid = target_batch.shape[-1] // 2
            target_batch = target_batch[..., mid]

        target_complex = self._to_complex(target_batch)

        # [DOMAIN CORRECTION] Convert k-space to image domain if needed
        # The Digital Twin simulator expects image-domain inputs for motion/marker simulation
        if hasattr(self.config.data, "dataset_type") and self.config.data.dataset_type == "kspace":
            from spectramr.infrastructure.physics.fft_ops import FFTTransformer

            fft = FFTTransformer(device=self.device)
            target_complex = fft.ifft2c(target_complex)

        # 1. Simulate degradation via Digital Twin
        corrupted_image, marker_prior, _joint = self.simulator(target_complex)

        # 1.b Normalize!
        clean_norm, t_min, t_max = self._normalize_to_range(target_complex)
        if target_complex.is_complex():
            corrupted_norm = corrupted_image / t_max
            marker_prior_norm = marker_prior / t_max
        else:
            corrupted_norm = 2.0 * (corrupted_image - t_min) / (t_max - t_min + 1e-8) - 1.0
            marker_prior_norm = marker_prior

        # 2. Get marker mask
        marker_mask = self.simulator.marker_mask.to(self.device)

        # Convert complex marker_prior to real-stacked
        if torch.is_complex(marker_prior_norm):
            marker_prior_rs = torch.cat([marker_prior_norm.real, marker_prior_norm.imag], dim=1)
        else:
            marker_prior_rs = marker_prior_norm

        # Expand mask for MarkerPriorProjection
        proj_mask = marker_mask
        if marker_prior_rs.shape[1] != proj_mask.shape[1]:
            proj_mask = proj_mask.expand_as(marker_prior_rs)

        # 3. Build prior projection for this batch
        prior_proj = MarkerPriorProjection(
            marker_mask=proj_mask,
            marker_prior=marker_prior_rs,
        ).to(self.device)

        # 4. Prepare input
        corrupted_input = torch.cat([corrupted_norm.real, corrupted_norm.imag], dim=1)
        clean_target = torch.cat([clean_norm.real, clean_norm.imag], dim=1)

        # 5. Forward through generator (learned regulariser)
        gen = self.generator_model
        # Coerce marker to real-stacked [B, 2, H, W] to match corrupted_input
        marker_prior_real = (
            torch.cat([marker_prior_norm.real, marker_prior_norm.imag], dim=1)
            if torch.is_complex(marker_prior_norm)
            else marker_prior_norm
        )
        # When the twin undersampled this batch, hand the sampling mask to the
        # generator so its data-consistency cascades enforce consistency
        # against the measured k-space (no-op kwarg for non-undersampling arms).
        pred = gen(
            corrupted_input,
            marker_prior=marker_prior_real,
            **undersampling_mask_kwargs(self.simulator),
        )

        # 6. Apply prior projection (anchor marker voxels)
        pred_anchored = prior_proj(pred)

        # 7. Compute standard reconstruction losses
        env_losses: dict[str, Any] = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        # Live iteration (loop_state seam). Frozen at 0 the warm-up gate never
        # opened, so ``l1`` — undeclared by these arms and resolved from the
        # ``lambda_l1`` schema default of 10.0 — was absent from ``components``
        # for the whole run (pitfall #16, #1937).
        loss_output = self.loss_computer.compute(
            pred=pred_anchored,
            target=clean_target,
            epoch=epoch,
            iteration=resolve_loop_iteration(self),
            losses_dict=env_losses,
        )

        standard_recon_loss = loss_output.total
        if standard_recon_loss is None:
            standard_recon_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        loss_prior = self.loss_marker_prior(
            pred,
            clean_target,
            marker_mask=marker_mask,
            marker_prior=marker_prior_norm,
        )
        loss_marker = self.loss_marker(
            pred,
            clean_target,
            marker_mask=marker_mask,
            ideal_marker=marker_prior_norm,
        )

        # Noise variance from marker (monitoring)
        noise_var = self.noise_estimator(corrupted_input, marker_mask)

        g_total_loss = (
            standard_recon_loss
            + self._lambda_prior * loss_prior
            + self._lambda_marker * loss_marker
        )

        # Ensure loss is real before backward
        if torch.is_complex(g_total_loss):
            g_total_loss = g_total_loss.real

        result = {
            "g_total_loss": g_total_loss,
            "loss_prior": loss_prior.detach(),
            "loss_marker": loss_marker.detach(),
            "noise_var": noise_var.mean().detach(),
        }
        for k, v in loss_output.components.items():
            result[k] = v.detach() if hasattr(v, "detach") else v

        return result

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate with deterministic degradation and image quality metrics."""
        val_batch = (input_batch, target_batch)
        batch_idx = kwargs.get("batch_idx", 0)
        if isinstance(val_batch, (list, tuple)):
            target = val_batch[1]
        elif isinstance(val_batch, dict):
            target = val_batch.get("target", val_batch.get("hr"))
        elif hasattr(val_batch, "target"):
            target = val_batch.target
        else:
            target = val_batch

        target = self._to_device(target)

        if target.ndim == 5:
            mid = target.shape[-1] // 2
            target = target[..., mid]

        with torch.no_grad():
            # #1190: calling ``_compute_losses_impl`` directly bypasses the
            # ``_compute_losses`` wrapper that emits the ``model_output``
            # snapshot, so arm it here. ``snapshot_source("val")`` is the OUTER
            # manager so the phase is still "val" when the emit runs on exit --
            # without it the record would claim the training data chain.
            with (
                self.snapshot_source("val"),
                self._capture_model_output(
                    module=self.generator_model,
                    input_batch=target,
                    target_batch=target,
                    step=batch_idx,
                ),
            ):
                losses = self._compute_losses_impl(target, target, epoch=0)

            target_complex = self._to_complex(target)

            # [DOMAIN CORRECTION] Convert k-space to image domain if needed
            if (
                hasattr(self.config.data, "dataset_type")
                and self.config.data.dataset_type == "kspace"
            ):
                from spectramr.infrastructure.physics.fft_ops import FFTTransformer

                fft = FFTTransformer(device=self.device)
                target_complex = fft.ifft2c(target_complex)

            scores = self._score_at_current_twin(target_complex, cache_visuals=True)

        result = {k: float(v) for k, v in losses.items() if isinstance(v, torch.Tensor)}
        result.update(scores)

        # Out-of-distribution rungs (physics.digital_twin.ood_acceleration_range):
        # the same corrupt-reconstruct-score pass with the twin held at each rung;
        # ``val_ood_accelerations`` is written on every step (0 when undeclared).
        with torch.no_grad():
            result.update(
                ood_acceleration_readout(
                    self.simulator,
                    ood_accelerations(self.config),
                    lambda: self._score_at_current_twin(target_complex, cache_visuals=False),
                )
            )
        return result

    def _score_at_current_twin(
        self, target_complex: torch.Tensor, *, cache_visuals: bool
    ) -> dict[str, float]:
        """Corrupt ``target_complex`` with the twin as it is set now, reconstruct, score.

        One pass for the in-distribution rung and for every OOD rung
        (``at_acceleration`` on the twin is the only thing that changes).
        Returns ``val_psnr`` plus the configured validation metrics under their
        own keys; ``cache_visuals`` stores the magnitude pair for the
        pipeline's image capture (the in-distribution pass only).
        """
        corrupted_image, marker_prior, _ = self.simulator(target_complex)

        clean_norm, t_min, t_max = self._normalize_to_range(target_complex)
        if target_complex.is_complex():
            corrupted_norm = corrupted_image / t_max
            marker_prior_norm = marker_prior / t_max
        else:
            corrupted_norm = 2.0 * (corrupted_image - t_min) / (t_max - t_min + 1e-8) - 1.0
            marker_prior_norm = marker_prior

        corrupted_input = torch.cat([corrupted_norm.real, corrupted_norm.imag], dim=1)
        marker_prior_rs = (
            torch.cat([marker_prior_norm.real, marker_prior_norm.imag], dim=1)
            if torch.is_complex(marker_prior_norm)
            else marker_prior_norm
        )
        pred = self.generator_model(
            corrupted_input,
            marker_prior=marker_prior_rs,
            **undersampling_mask_kwargs(self.simulator),
        )

        # ADMM outputs real-stacked
        pred_mag = torch.complex(pred[:, 0::2], pred[:, 1::2]).abs()
        target_mag = clean_norm.abs()
        pm = pred_mag.unsqueeze(1) if pred_mag.ndim == 3 else pred_mag
        tm = target_mag.unsqueeze(1) if target_mag.ndim == 3 else target_mag

        if cache_visuals:
            # Cache magnitude images for pipeline image capture (train.py:1561).
            # Without this, the pipeline feeds raw k-space through the generator
            # (which expects Digital Twin-processed input) and produces garbage.
            self._last_visual_pred = pm
            self._last_visual_target = tm

        mse = torch.nn.functional.mse_loss(pred_mag, target_mag)
        data_range = target_mag.max() - target_mag.min() + 1e-8
        psnr = 10.0 * torch.log10(data_range**2 / (mse + 1e-10))
        scores: dict[str, float] = {"val_psnr": float(psnr)}

        # Emit YAML-configured ``validation.metrics`` so early-stopping /
        # best-metric monitors (val_hfen, val_robust_mri_psnr, ...) resolve;
        # without this only val_psnr existed and monitors never fired
        # (dispatch 6944227, 2026-05-25). compute() guards each metric, so a
        # single failure is skipped, not fatal -- surfaced as a warning, never
        # silently swallowed (CLAUDE.md #9/#10).
        computer = getattr(self, "validation_metrics_computer", None)
        if computer is not None:
            try:
                scores.update({k: float(v) for k, v in computer.compute(pm, tm).items()})
            except Exception as exc:
                logger.warning(
                    "[VFADMMStrategy] validation_metrics_computer failed; "
                    "only val_psnr available this step: %s",
                    exc,
                )
        return scores


__all__ = ["ConcreteVFADMMStrategy"]
