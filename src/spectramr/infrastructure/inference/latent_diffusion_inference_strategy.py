"""Latent Diffusion Inference Strategy

Predict path for latent diffusion generators. ``LatentDiffusionGenerator.sample``
runs the reverse chain in latent space and decodes to image space, so this
strategy calls it once with the same contract training's validation uses.

The reverse schedule is bound to the training scheduler built from the arm's
``training.diffusion`` block, so predict inverts the forward process training ran (#204).
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.utils.diffusion_mixin import (
    bind_latent_reverse_schedule,
    build_training_scheduler,
)

from .base_inference_strategy import BaseInferenceStrategy

logger = logging.getLogger(__name__)


class LatentDiffusionInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for Latent Diffusion Models.

    Runs ``model.sample(shape, device, contrast_idx, condition_image, num_inference_steps)``
    and returns its decoded image. ``num_inference_steps`` is forwarded only when
    declared; otherwise the generator runs its full training-length chain.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize latent diffusion inference strategy.

        Args:
            model: A ``LatentDiffusionGenerator`` (or wrapper exposing ``sample`` and ``diffusion``).
            device: Device to run inference on.
            config: Full config dict (``model_dump()``), read for ``training.diffusion``.
        """
        super().__init__(model, device, config)

        self.ldm_config = self.config.get("latent_diffusion", {})
        self.num_inference_steps = self.ldm_config.get("num_inference_steps")
        self._schedule_bound = False

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.LATENT_DIFFUSION

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Move the conditioning image to the inference device.

        Args:
            input_tensor: Conditioning image ``[B, C, H, W]``.
            **kwargs: Additional parameters.

        Returns:
            Preprocessed tensor.
        """
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)
        return input_tensor.detach()

    def _bind_reverse_schedule(self) -> None:
        """Bind the generator's reverse schedule to the training SSOT, once (#204)."""
        if self._schedule_bound:
            return
        diffusion = (self.config.get("training") or {}).get("diffusion")
        if not diffusion:
            raise ConfigurationError(
                "Latent diffusion predict needs training.diffusion in the config to bind the "
                "reverse schedule to the training forward process; none was supplied."
            )
        scheduler = build_training_scheduler(
            num_timesteps=int(diffusion["timesteps"]),
            beta_schedule=str(diffusion["noise_schedule"]),
            device=self.device,
            beta_start=float(diffusion["beta_start"]),
            beta_end=float(diffusion["beta_end"]),
        )
        gen = self.model.module if hasattr(self.model, "module") else self.model
        if not bind_latent_reverse_schedule(gen, scheduler):
            raise ConfigurationError(
                f"{type(gen).__name__} owns no reverse schedule to bind; predict cannot "
                "verify it inverts the training forward process."
            )
        self._schedule_bound = True

    def run_inference(self, input_tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        """Sample the decoded image conditioned on ``input_tensor``.

        Args:
            input_tensor: Conditioning image ``[B, C, H, W]``; its shape is the output shape.
            **kwargs:
                - contrast_idx: Optional ``[B]`` long tensor of contrast labels.
                - num_inference_steps: Optional step count (None = full training chain).

        Returns:
            ``(image, metadata)``; the image is in the generator's decoded output space.
        """
        if kwargs.get("decode", True) is False:
            raise ValueError(
                "LatentDiffusionGenerator.sample() decodes to image space; decode=False has "
                "no latent output on this path."
            )
        if not hasattr(self.model, "sample"):
            raise ConfigurationError(
                f"{type(self.model).__name__} exposes no sample(); latent predict cannot run."
            )
        self._bind_reverse_schedule()
        steps = kwargs.get("num_inference_steps", self.num_inference_steps)
        with torch.no_grad():
            image = self.model.sample(
                tuple(input_tensor.shape),
                device=str(self.device),
                contrast_idx=kwargs.get("contrast_idx"),
                condition_image=input_tensor,
                num_inference_steps=steps,
            )
        return image, {}

    def postprocess_output(
        self,
        output_tensor: torch.Tensor | tuple[torch.Tensor, dict[str, Any]],
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Move the sampled image to CPU and map it to [0, 1].

        Args:
            output_tensor: Raw output, ``(image, metadata)`` or a bare image.
            **kwargs: Additional parameters.

        Returns:
            ``(image, metadata)`` with the image clamped to [0, 1].
        """
        if isinstance(output_tensor, tuple):
            output, metadata = output_tensor
        else:
            output = output_tensor
            metadata = {}

        output = output.detach().cpu()

        if output.ndim == 4:
            if output.min() < 0:
                output = (output + 1.0) / 2.0
            output = torch.clamp(output, 0.0, 1.0)

        return output, metadata
