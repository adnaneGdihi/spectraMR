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

from spectramr.data.batch_types import read_batch_field
from spectramr.data.transforms.acquisition_params_injector import (
    infer_acquisition_params,
)
from spectramr.infrastructure.builders.leaf.adapter_builders import apply_pre_model
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.hermitian import (
    enforce_hermitian,
    hermitian_gaussian_noise,
)
from spectramr.infrastructure.physics.line_axis import line_axis_for
from spectramr.infrastructure.physics.sampling import (
    ACCELERATOR_TO_MASK_TYPE,
    MaskGenerator,
    MaskType,
)
from spectramr.infrastructure.physics.sampling_registry import SamplingPatternRegistry
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.diffusion.samplers import SamplerRegistry
from spectramr.models.diffusion.samplers.annealed_langevin_dc import (
    complex_to_interleaved,
    interleaved_to_complex,
)

logger = logging.getLogger(__name__)


def _as_interleaved(x: torch.Tensor) -> torch.Tensor:
    """The model's real ``[B, 2C, H, W]`` view of a complex image (``ifft_kspace_to_image`` output)."""
    return complex_to_interleaved(x) if torch.is_complex(x) else x


class ScoreFieldTomographyStrategy(BaseTrainingStrategy):
    """Joint (x, c) denoising score matching with k-space Hermitian noise."""

    # Debug-snapshot contract (non-negotiable 14): the model is fed a noised
    # tensor, not the prepared input; declared here and emitted under the tag.
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"
    # Validation undersamples the target with the arm's ``undersampling:`` block
    # (``_validation_mask``); training is unconditional DSM on the target.
    applies_undersampling = True
    #: The one reconstruction sampler this strategy validates with.
    validation_sampler: str = "annealed_langevin_dc"

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        # Typed sub-block (pitfall #15). Previously read via getattr against the
        # extra="allow" dict — a supplied YAML block silently returned defaults.
        sft = getattr(self.config.training, "score_field_tomography", None)
        self.sigma_min = float(sft.sigma_min) if sft is not None else 1e-3
        self.sigma_max = float(sft.sigma_max) if sft is not None else 1.0
        self.cond_dim = int(sft.cond_dim) if sft is not None else 5
        self.val_num_levels = int(sft.val_num_levels) if sft is not None else 10
        self.val_steps_per_level = int(sft.val_steps_per_level) if sft is not None else 5
        self.val_step_size = float(sft.val_step_size) if sft is not None else 0.2
        logger.info(
            "ScoreFieldTomographyStrategy (source=%s): sigma_min=%.4g, sigma_max=%.4g, "
            "cond_dim=%d, val sampler=%s (levels=%d, steps/level=%d, eps=%.3g)",
            "config" if sft is not None else "defaults",
            self.sigma_min,
            self.sigma_max,
            self.cond_dim,
            self.validation_sampler,
            self.val_num_levels,
            self.val_steps_per_level,
            self.val_step_size,
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
        x_clean = _as_interleaved(target_batch)
        b = x_clean.shape[0]
        device = x_clean.device
        if self._dim_x is None:
            self._dim_x = int(x_clean[0].numel())
        c_clean = self._resolve_condition(kwargs.get("batch"), x_clean)
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

    def _resolve_condition(self, batch: Any, x: torch.Tensor) -> torch.Tensor:
        """The acquisition vector ``c`` for ``x``: the batch's, else inferred.

        Read from the batch when ``data.expose.acquisition_params`` publishes it,
        else built from the static PHYSICS_REGISTRY by
        :func:`infer_acquisition_params` and truncated / zero-padded to
        ``cond_dim``. Training and validation share this so ``c*`` at sampling
        time is the vector the score was trained on.
        """
        device = x.device
        published = read_batch_field(batch, "acquisition_params") if batch is not None else None
        if isinstance(published, torch.Tensor):
            return published.to(device).float()
        c = infer_acquisition_params({"image": x}).to(device)
        if c.shape[-1] > self.cond_dim:
            return c[..., : self.cond_dim]
        if c.shape[-1] < self.cond_dim:
            pad = torch.zeros(x.shape[0], self.cond_dim - c.shape[-1], device=device)
            c = torch.cat([c, pad], dim=-1)
        return c

    def _undersampling_record(self) -> dict[str, Any]:
        """The declared undersampling the validation mask is drawn from (raises on unsupported)."""
        us = getattr(self.config, "undersampling", None)
        if us is None:
            raise ValueError(
                "ScoreFieldTomographyStrategy validation reconstructs from undersampled "
                "k-space and needs an `undersampling:` block (base_acceleration, "
                "center_fraction)."
            )
        direction = getattr(us, "mask_direction", None)
        acq = getattr(getattr(self.config, "data", None), "acquisition", None)
        # Undeclared direction samples rows ('y', dim -2), MaskGenerator's native axis.
        line_axis = (
            line_axis_for(direction, getattr(acq, "phase_encode_axis", None))
            if direction is not None
            else "y"
        )
        raw = str(us.acceleration_type).strip().lower()
        try:
            mask_type = MaskType(raw)
        except ValueError:
            canonical = SamplingPatternRegistry.resolve(raw)  # raises on an unknown name
            mask_type = ACCELERATOR_TO_MASK_TYPE.get(canonical)
            if mask_type is None:
                raise ValueError(
                    f"undersampling.acceleration_type={raw!r} is accelerator-only and has no "
                    f"static mask; supported: {sorted(ACCELERATOR_TO_MASK_TYPE)}"
                ) from None
        seed = us.mask_seed if us.mask_seed is not None else int(self.config.run.seed)
        return {
            "mask_type": mask_type.value,
            "acceleration": float(us.base_acceleration),
            "center_fraction": float(us.center_fraction),
            "density_power": float(us.density_power),
            "seed": int(seed),
            "line_axis": line_axis,
        }

    def _validation_mask(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Fixed-seed ``[1, 1, H, W]`` mask, cached per shape so ``val_psnr`` is comparable."""
        cache = self.__dict__.setdefault("_val_mask_cache", {})
        key = (height, width, str(device))
        if key not in cache:
            rec = self._undersampling_record()
            columns = rec["line_axis"] == "x"
            mask = MaskGenerator(seed=rec["seed"]).generate_mask(
                rec["mask_type"],
                (width, height) if columns else (height, width),
                acceleration_factor=rec["acceleration"],
                center_fraction=rec["center_fraction"],
                density_power=rec["density_power"],
            )
            mask = mask.T if columns else mask
            cache[key] = mask.to(device=device, dtype=torch.float32)[None, None]
        return cache[key]

    def _build_validation_sampler(self) -> Any:
        sampler = self.__dict__.get("_val_sampler")
        if sampler is None:
            sampler = SamplerRegistry.get(
                self.validation_sampler,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                num_levels=self.val_num_levels,
                steps_per_level=self.val_steps_per_level,
                step_size=self.val_step_size,
                seed=int(self.config.run.seed),
            )
            self._val_sampler = sampler
        return sampler

    def _first_steps_extra(self, epoch: int, batch: Any = None) -> dict[str, Any]:
        """Stamp the resolved score-field and validation-sampler knobs (pitfall 15)."""
        extra = super()._first_steps_extra(epoch, batch=batch)
        try:
            undersampling: dict[str, Any] | str = self._undersampling_record()
        except ValueError as exc:  # validation raises on it; the stamp records why
            undersampling = f"unresolvable: {exc}"
        extra["score_field_tomography"] = {
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "cond_dim": self.cond_dim,
            "validation_sampler": self.validation_sampler,
            "val_num_levels": self.val_num_levels,
            "val_steps_per_level": self.val_steps_per_level,
            "val_step_size": self.val_step_size,
            "validation_undersampling": undersampling,
        }
        return extra

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Reconstruct by annealed Langevin on ``s_x`` at fixed ``c*`` and score it.

        The target image (after the same ``pre_model`` chain as training) is
        undersampled with the arm's declared mask, ``y = M * fft2c(x)``; the chain
        starts from the zero-filled image and is projected onto ``y`` after every
        step. ``c*`` is the target's acquisition vector, resolved as in training.
        PSNR/SSIM are computed on magnitude images, the convention of the cohort.
        """
        del kwargs
        input_batch, target_batch = apply_pre_model(
            self.adapter_chains, input_batch, target_batch
        )
        target = _as_interleaved(target_batch.to(self.device, non_blocking=True)).float()
        z_target = interleaved_to_complex(target)
        mask = self._validation_mask(target.shape[-2], target.shape[-1], target.device)
        measured = fft2c(z_target) * mask
        x_init = complex_to_interleaved(ifft2c(measured))
        c_star = self._resolve_condition(batch_data, target)

        model = self.generator_model
        was_training = model.training
        model.eval()
        try:

            def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
                return model(x, sigma, c_star)[0]

            recon = self._build_validation_sampler().sample(score_fn, x_init, measured, mask)
        finally:
            model.train(was_training)
        mag_recon = interleaved_to_complex(recon).abs()
        return self._score_validation_output(mag_recon, z_target.abs())


__all__ = ["ScoreFieldTomographyStrategy", "hermitian_gaussian_noise"]
