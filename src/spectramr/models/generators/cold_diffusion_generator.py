"""Cold Diffusion Generator (Refactored)
======================================

Cold Diffusion generator implementing the IGenerator interface.
Based on Cold Diffusion (https://arxiv.org/abs/2208.09392).

Refactored to use DiffusionMixin for shared diffusion logic.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from spectramr.config.schemas.enums import SchedulerTypes
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

from .mixins import DiffusionMixin

logger = logging.getLogger(__name__)


@dataclass
class ColdDiffusionGeneratorConfig:
    """Configuration for Cold Diffusion Generator parameters."""

    timesteps: int = 1000
    beta_schedule: str = "linear"
    device: str | None = None
    degradation_type: str = "noise"
    cold_schedule: str = "linear"
    restoration_steps: int = 1


@register_model(name="cold_diffusion", training_mode="diffusion")
class ColdDiffusionGenerator(DiffusionMixin, IGenerator, nn.Module):
    """Cold Diffusion generator using DiffusionMixin for shared logic.

    Instead of predicting noise, the model predicts the original image x_0.
    The sampling process then involves a degradation step.

    Uses DiffusionMixin for:
    - q_sample (forward diffusion)
    - p_sample (reverse diffusion step)
    - p_sample_loop (full sampling loop)
    - _extract (tensor indexing)
    - Schedule initialization
    """

    def __init__(
        self,
        denoising_model: nn.Module | None = None,
        config: ColdDiffusionGeneratorConfig | None = None,
        timesteps: int | None = None,
        beta_schedule: str | None = None,
        device: str | None = None,
        degradation_type: str | None = None,
        cold_schedule: str | None = None,
        restoration_steps: int | None = None,
    ) -> None:
        """Initialize Cold Diffusion Generator.

        Args:
            denoising_model: Neural network that predicts x_0 from x_t.
            config: Configuration object. If provided, overrides other params.
            timesteps: Number of diffusion timesteps.
            beta_schedule: Schedule type ('linear', 'cosine').
            device: Target device.
            degradation_type: Degradation type ('noise').
            cold_schedule: Cold schedule type ('linear', 'cosine').
            restoration_steps: Number of restoration steps.
        """
        super().__init__()

        # Build config
        if config is not None:
            self.config = config
        else:
            self.config = ColdDiffusionGeneratorConfig(
                timesteps=timesteps if timesteps is not None else 1000,
                beta_schedule=beta_schedule if beta_schedule is not None else "linear",
                device=device,
                degradation_type=degradation_type or "noise",
                cold_schedule=cold_schedule or "linear",
                restoration_steps=restoration_steps or 1,
            )

        # Set instance attributes
        self.degradation_type = self.config.degradation_type
        self.restoration_steps = self.config.restoration_steps

        # Create denoising model if not provided
        if denoising_model is None:
            from .unet_generator import UNetGenerator

            denoising_model = UNetGenerator(
                in_channels=1,
                out_channels=1,
                output_activation=None,
            )
        self.denoising_model = denoising_model
        self._denoiser_takes_time, self._denoiser_takes_cond = _denoiser_capabilities(
            denoising_model
        )
        if not self._denoiser_takes_time:
            logger.warning(
                "ColdDiffusionGenerator: denoiser %s ignores the timestep, so every noise "
                "level is denoised by one time-blind function.",
                type(denoising_model).__name__,
            )

        # Set device
        # `self.config` is a ColdDiffusionGeneratorConfig, NOT TrainingSettings:
        # it carries `device` and has no `run:` block. Phase 4b's
        # `device -> run.device` rename was applied here by name rather than by
        # receiver, turning a working read into a hard AttributeError.
        self._device = self.config.device or "cpu"

        # Initialize diffusion buffers via DiffusionMixin
        self._initialize_diffusion_buffers(
            timesteps=self.config.timesteps,
            beta_schedule=self.config.beta_schedule,
        )

        # Initialize cold schedule
        self._initialize_cold_schedule()

        # Move to device
        self.to(self._device)

        logger.debug(
            "Initialized ColdDiffusionGenerator: timesteps=%d, schedule=%s",
            self.timesteps,
            self.config.beta_schedule,
        )

    def _initialize_cold_schedule(self) -> None:
        """Initialize cold-specific schedule buffers."""
        if self.config.cold_schedule == SchedulerTypes.LINEAR.value:
            cold_schedule = torch.linspace(1.0, 0.0, self.timesteps)
        elif self.config.cold_schedule == SchedulerTypes.COSINE.value:
            t = torch.linspace(0, self.timesteps, self.timesteps + 1)
            cold_schedule = torch.cos(0.5 * torch.pi * t / self.timesteps)[:-1]
        else:
            msg = f"Unknown cold schedule: {self.config.cold_schedule}"
            raise ValueError(msg)

        self.register_buffer("cold_schedule", cold_schedule)

    @property
    def name(self) -> str:
        return "ColdDiffusionGenerator"

    @property
    def device(self) -> torch.device:
        return self.betas.device

    # -------------------------------------------------------------------------
    # DiffusionMixin Required Methods
    # -------------------------------------------------------------------------
    def _predict_start_from_noise(
        self,
        x_t: Tensor,
        t: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Predict x_0 from x_t and predicted noise.

        For Cold Diffusion, this is not used directly since we predict x_0.
        """
        # Cold diffusion predicts x_0 directly, not noise
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_t.shape,
        )
        return (x_t - sqrt_one_minus_alpha_cumprod * noise) / sqrt_alpha_cumprod

    def _denoise(self, x: Tensor, t: Tensor, cond: Tensor | None = None) -> Tensor:
        """Call the denoiser with the timestep (and ``cond``) its signature accepts."""
        if cond is not None and not self._denoiser_takes_cond:
            raise ValueError(
                f"denoiser {type(self.denoising_model).__name__} ignores `cond`, so a "
                "conditional prediction would equal the unconditional one."
            )
        if not self._denoiser_takes_time:
            return self.denoising_model(x)
        if cond is None:
            return self.denoising_model(x, t)
        return self.denoising_model(x, t, cond)

    def _model_forward(self, x: Tensor, t: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass of the denoising model.

        For Cold Diffusion, we predict x_0 directly (not noise).
        """
        return self._denoise(x, t, kwargs.get("cond"))

    # -------------------------------------------------------------------------
    # Cold Diffusion Specific Methods
    # -------------------------------------------------------------------------
    def _degrade(self, x_0_pred: Tensor, t: Tensor) -> Tensor:
        """Degrade predicted x_0 to x_t using the chosen degradation."""
        if self.degradation_type == "noise":
            return self.q_sample(x_start=x_0_pred, t=t)
        msg = f"Unknown degradation type: {self.degradation_type}"
        raise ValueError(msg)

    @torch.no_grad()
    def p_sample(
        self,
        x: Tensor,
        t: Tensor,
        t_index: int,
        **kwargs: Any,
    ) -> Tensor:
        """Cold diffusion sampling step.

        Overrides DiffusionMixin.p_sample with cold diffusion logic.
        """
        cond = kwargs.get("cond")
        guidance_scale = kwargs.get("guidance_scale", 1.0)

        # Predict x_0 directly
        if guidance_scale > 1.0 and cond is not None:
            uncond_pred = self._denoise(x, t)
            cond_pred = self._denoise(x, t, cond)
            pred_x_start = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        else:
            pred_x_start = self._denoise(x, t, cond)

        if t_index == 0:
            return pred_x_start

        # Degrade to t-1
        prev_t = torch.full_like(t, t_index - 1)
        return self._degrade(pred_x_start, prev_t)

    # -------------------------------------------------------------------------
    # IGenerator Interface
    # -------------------------------------------------------------------------
    def forward(self, x: Tensor, t: Tensor | None = None, cond: Tensor | None = None) -> Tensor:
        """Predict x_0, noising ``x`` first unless the caller supplies its own timestep.

        Args:
            x: Clean image (B, C, H, W) when ``t`` is None; otherwise the caller's
                already-corrupted x_t (a bridge or interpolant strategy forms its own).
            t: Optional per-sample timestep of ``x``; passed to the denoiser as given.
            cond: Optional conditioning passed to a denoiser that accepts it.

        Returns:
            Predicted clean image x_0 of shape (B, C, H, W).
        """
        # Shape guards (unit-test.md Directive 1.1)
        if x.dim() != 4:
            msg = f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D"
            raise ValueError(msg)

        if t is not None:
            return self._denoise(x, t, cond)

        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)
        x_t = self.q_sample(x, t)
        return self._denoise(x_t, t, cond)

    def generate(self, z: Tensor, **kwargs: Any) -> Tensor:
        """Sample with the shape of ``z``; ``kwargs`` carry ``cond`` and ``guidance_scale``."""
        return self.p_sample_loop(z.shape, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        return input_shape

    def get_parameter_count(self) -> int:
        """Count total parameters in the denoising model."""
        return sum(p.numel() for p in self.denoising_model.parameters() if p.requires_grad)


def _denoiser_capabilities(model: nn.Module) -> tuple[bool, bool]:
    """Whether ``model`` uses a timestep and a ``cond``.

    A declared ``consumes_time`` / ``consumes_cond`` wins over the signature, because a
    forward can accept an argument it discards (UNet takes ``cond`` and drops it).
    """
    params = list(inspect.signature(model.forward).parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    takes_time = bool(getattr(model, "consumes_time", len(positional) >= 2))
    takes_cond = bool(getattr(model, "consumes_cond", len(positional) >= 3))
    return takes_time, takes_cond
