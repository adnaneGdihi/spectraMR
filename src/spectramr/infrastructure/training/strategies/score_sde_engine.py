"""VE-SDE training and predictor-corrector validation for a score generator.

``DiffusionTrainingStrategy`` hands its training and validation steps to this engine when the
generator declares ``sde_type``. It is composed rather than subclassed so the strategy tower
stays at its baselined depth.

Training is denoising score matching on the VE-SDE ``x_t = x_0 + sigma(t) z``, ``t ~ U(eps, 1)``
(Song et al., ICLR 2021). The network is a denoiser ``D(x_t, t)`` fed the VP-equivalent input
``x_t / sqrt(1 + sigma^2)``; the score is ``(D - x_t) / sigma^2`` by Tweedie's formula, and the
declared image losses on ``(D, x_0)`` are the objective. Validation reconstructs from a
retrospective measurement ``y = m * fft2c(input)`` with the registered sampler named by
``physics.data_consistency.method`` (Chung and Ye, MedIA 2022).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from spectramr.core.module_utils import unwrap_model
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.sampling import MaskGenerator, MaskType
from spectramr.infrastructure.training.strategies.score_sde_config import (
    line_axis,
    validate_score_sde_config,
)
from spectramr.models.diffusion.samplers import get_sampler
from spectramr.models.diffusion.samplers.pc_sampler_consistency import (
    DEFAULT_CORRECTOR_STEPS,
    DEFAULT_SNR,
    SAMPLING_EPS,
)
from spectramr.models.diffusion.score_sde.sde import VESDE


def _flatten_depth(x: Tensor) -> Tensor:
    """``[B, C, H, W, D]`` -> ``[B*D, C, H, W]``; 4-D passes through."""
    if x.dim() == 5:
        b, c, h, w, d = x.shape
        return x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
    if x.dim() != 4:
        raise ValueError(f"score-SDE path expects 4-D or 5-D image tensors, got {x.dim()}-D.")
    return x


class ScoreSDEEngine:
    """The VE-SDE objective and its conditional sampler, bound to one run's config."""

    def __init__(
        self,
        sde: VESDE,
        num_timesteps: int,
        dc_method: str,
        dc_weight: float,
        sampling_steps: int,
        mask_type: str,
        acceleration: float,
        center_fraction: float,
        seed: int,
        line_axis: int = -2,
    ) -> None:
        self.sde = sde
        self.num_timesteps = int(num_timesteps)
        self.dc_method = dc_method
        self.dc_weight = float(dc_weight)
        self.sampling_steps = int(sampling_steps)
        self.mask_type = MaskType(mask_type)
        self.acceleration = float(acceleration)
        self.center_fraction = float(center_fraction)
        self.seed = int(seed)
        self.line_axis = int(line_axis)
        self._masks: dict[tuple[int, int, str], Tensor] = {}

    def describe(self) -> dict[str, Any]:
        """Declared-and-applied knobs, stamped into the snapshot provenance."""
        return {
            "sde_type": "vesde",
            "sigma_min": self.sde.sigma_min,
            "sigma_max": self.sde.sigma_max,
            "num_timesteps": self.num_timesteps,
            "time_conditioning": "t * (num_timesteps - 1)",
            "parameterization": "denoiser, score = (D - x_t) / sigma^2",
            "dc_method": self.dc_method,
            "dc_weight": self.dc_weight,
            "sampling_steps": self.sampling_steps,
            "snr": DEFAULT_SNR,
            "corrector_steps": DEFAULT_CORRECTOR_STEPS,
            "sampling_eps": SAMPLING_EPS,
            "measurement": f"{self.mask_type.value} R={self.acceleration} cf={self.center_fraction}",
            "mask_line_axis": self.line_axis,
        }

    def denoise(self, gen: Any, x_t: Tensor, t: Tensor) -> Tensor:
        sigma = self.sde.sigma(t).view(-1, *([1] * (x_t.dim() - 1)))
        out = gen(x_t / torch.sqrt(1.0 + sigma**2), t * (self.num_timesteps - 1))
        return out[0] if isinstance(out, tuple) else out

    def score(self, gen: Any, x: Tensor, t: Tensor) -> Tensor:
        sigma = self.sde.sigma(t).view(-1, *([1] * (x.dim() - 1)))
        return (self.denoise(gen, x, t) - x) / sigma**2

    def losses(self, strategy: Any, target_batch: Tensor) -> dict[str, Tensor]:
        clean = _flatten_depth(target_batch)
        t = torch.rand(clean.shape[0], device=clean.device) * (1.0 - SAMPLING_EPS) + SAMPLING_EPS
        sigma = self.sde.sigma(t).view(-1, *([1] * (clean.dim() - 1)))
        x_t = clean + sigma * torch.randn_like(clean)
        strategy._declare_model_input(
            {"model_input": x_t / torch.sqrt(1.0 + sigma**2), "noised": x_t, "target": clean},
            in_kspace_keys=set(),
            extra={"model_input_key": "model_input", "score_sde": self.describe()},
        )
        denoised = self.denoise(strategy.generator_model, x_t, t)
        components: dict[str, Tensor] = {}
        total = strategy._apply_builder_image_losses(denoised, clean, components)
        if total is None:
            raise ConfigurationError(
                "The score-SDE objective is the declared image losses on the Tweedie denoiser, "
                "and this arm declares none under losses.image_losses."
            )
        return {"loss_total": total, **components}

    def _mask(self, h: int, w: int, device: torch.device) -> Tensor:
        key = (h, w, str(device))
        mask = self._masks.get(key)
        if mask is None:
            # MaskGenerator selects rows (lines indexed on dim -2); transpose when they index dim -1.
            rows, cols = (h, w) if self.line_axis == -2 else (w, h)
            mask = MaskGenerator().generate_mask(
                self.mask_type,
                (rows, cols),
                acceleration_factor=self.acceleration,
                center_fraction=self.center_fraction,
                seed=self.seed,
            )
            if self.line_axis == -1:
                mask = mask.transpose(-2, -1)
            mask = mask.to(device=device, dtype=torch.float32).reshape(1, 1, h, w)
            self._masks[key] = mask
        return mask

    def predict(self, strategy: Any, input_batch: Tensor) -> Tensor:
        """Reconstruct from the measurement ``m * fft2c(input)`` with the declared sampler."""
        x = _flatten_depth(input_batch)
        mask = self._mask(x.shape[-2], x.shape[-1], x.device)
        gen = strategy.generator_model
        sampler = get_sampler(
            self.dc_method,
            score_fn=lambda z, t: self.score(gen, z, t),
            sde=self.sde,
            n_steps=self.sampling_steps,
            dc_weight=self.dc_weight,
        )
        return sampler.sample(fft2c(x) * mask, mask, shape=x.shape)

    def validation_step(
        self,
        strategy: Any,
        input_batch: Tensor,
        target_batch: Tensor,
        batch_data: Any,
        kwargs: dict,
    ) -> dict[str, float]:
        prep = strategy._prepare_validation_data(
            (input_batch, target_batch), input_batch, target_batch, batch_data
        )
        scale: Any = 1.0
        if prep is not None:
            input_batch, target_batch, scale = prep
        depth = input_batch.shape[-1] if input_batch.dim() == 5 else 1
        if isinstance(scale, Tensor) and scale.dim() == 4 and depth > 1:
            scale = scale.repeat_interleave(depth, dim=0)
        input_batch, target_batch = _flatten_depth(input_batch), _flatten_depth(target_batch)
        strategy.generator_model.eval()
        with torch.no_grad():
            pred = self.predict(strategy, input_batch)
            t0 = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
            return strategy._compute_validation_metrics(
                pred,
                target_batch,
                input_batch,
                t0,
                batch_data,
                scale,
                batch_idx=kwargs.get("batch_idx", 0),
            )


#: The DiffusionTrainingStrategy methods that dispatch to the engine.
_HOOKED = ("_compute_losses_impl", "validation_step", "_generate_validation_prediction")


def build_score_sde_engine(strategy: Any) -> ScoreSDEEngine | None:
    """The engine for a generator that declares ``sde_type``, else ``None``; raise on any conflict."""
    gen = unwrap_model(strategy.generator_model)
    if not isinstance(getattr(gen, "sde_type", None), str):
        return None
    overridden = [
        name
        for name in _HOOKED
        if getattr(type(strategy), name).__qualname__ != f"DiffusionTrainingStrategy.{name}"
    ]
    if overridden:
        raise ConfigurationError(
            f"{type(strategy).__name__} overrides {overridden}, where DiffusionTrainingStrategy "
            "hands a VE-SDE generator to the score-SDE engine; its sde_type would be ignored."
        )
    config = strategy.config
    declared = strategy._declared_prediction_type()
    if declared not in (None, "sample"):
        raise ConfigurationError(
            f"training.diffusion.prediction_type={declared!r} is not what the score-SDE path "
            "trains: the network is a denoiser (sample prediction)."
        )
    if strategy._conditions_on_input():
        raise ConfigurationError(
            "The score-SDE path trains an unconditional prior and conditions through data "
            "consistency at sampling; remove training.diffusion.condition_on_input."
        )
    validate_score_sde_config(config)
    dc = config.physics.data_consistency
    us = config.undersampling
    steps = strategy._resolve_validation_sampling_steps() or strategy.num_timesteps
    return ScoreSDEEngine(
        sde=VESDE(gen.sigma_min, gen.sigma_max, N=strategy.num_timesteps),
        num_timesteps=strategy.num_timesteps,
        dc_method=str(dc.method).lower(),
        dc_weight=dc.weight,
        sampling_steps=steps,
        mask_type=us.mask_types[0],
        acceleration=us.base_acceleration,
        center_fraction=us.center_fraction,
        seed=config.run.seed,
        line_axis=line_axis(config),
    )
