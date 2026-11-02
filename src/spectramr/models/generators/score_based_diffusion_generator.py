"""Score-Based Diffusion Generator (Refactored)
==============================================

Score-based diffusion generator implementing the IGenerator interface.
Refactored to use DiffusionMixin for shared diffusion logic.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn

from spectramr.models.diffusion.score_sde.score_network import ScoreNetwork
from spectramr.models.diffusion.score_sde.sde_spec import refuse_discrete_path, resolve_sde
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

from .mixins import DiffusionMixin

logger = logging.getLogger(__name__)


@register_model(
    name="score_based_diffusion",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class ScoreBasedDiffusionGenerator(DiffusionMixin, IGenerator, nn.Module):
    """Score-based diffusion generator using DiffusionMixin.

    Uses score-based generative modeling for high-quality image synthesis.
    Inherits shared diffusion logic from DiffusionMixin:
    - q_sample (forward diffusion)
    - p_sample (reverse diffusion step)
    - p_sample_loop (full sampling loop)
    - Schedule initialization
    """

    # The bare forward self-noises a clean input, so the Tier-2 ``input_invariant``
    # probe cannot judge measurement dependence on it. The measurement enters through
    # the strategy instead: ``training.diffusion.condition_on_input`` concatenates it
    # onto x_t at training and in the validation sampler, or a posterior sampler
    # (``sample()``, e.g. dds) takes it at inference.
    synthetic_forward_probe_skip = frozenset({"input_invariant"})

    def __init__(
        self,
        denoising_model: nn.Module | None = None,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        device: str | None = None,
        gradient_lambda: float = 1.0,
        entropy_lambda: float = 0.1,
        ssim_lambda: float = 0.5,
        sde_type: str | None = None,
        sigma_min: float | None = None,
        sigma_max: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Score-Based Diffusion Generator.

        Args:
            denoising_model: Neural network that predicts noise.
            timesteps: Number of diffusion timesteps.
            beta_schedule: Schedule type ('linear', 'cosine').
            device: Target device.
            gradient_lambda: Weight for gradient error loss.
            entropy_lambda: Weight for gradient entropy loss.
            ssim_lambda: Weight for SSIM loss.
            sde_type: ``None`` (discrete DDPM) or ``"vesde"`` with ``sigma_min``/``sigma_max``.
            **kwargs: Extra parameters for ScoreNetwork if denoising_model is None.
        """
        super().__init__()
        self.sde_type, self.sigma_min, self.sigma_max = resolve_sde(sde_type, sigma_min, sigma_max)

        if denoising_model is None:
            # Auto-build ScoreNetwork from kwargs; num_layers > 4 reads as total depth, so 2 per level.
            features = kwargs.get("features", [32, 64, 128, 256])
            nf = features[0]
            ch_mult = [f // nf for f in features[1:]]
            if not ch_mult:
                ch_mult = [1, 2, 2]  # Default fallback
            num_res_blocks = kwargs.get("num_layers", 2)
            if num_res_blocks > 4:
                num_res_blocks = 2
            in_ch = kwargs.get("in_channels", 2)

            self.denoising_model = ScoreNetwork(
                in_channels=in_ch,
                nf=nf,
                ch_mult=ch_mult,
                num_res_blocks=num_res_blocks,
            )
            logger.info(f"Auto-built ScoreNetwork with nf={nf}, ch_mult={ch_mult}")
        else:
            self.denoising_model = denoising_model

        # Set device
        if device is None or (device == "cuda" and not torch.cuda.is_available()):
            device = "cpu"
        self._device_str = device

        # Score-based specific parameters
        self.gradient_lambda = gradient_lambda
        self.entropy_lambda = entropy_lambda
        self.ssim_lambda = ssim_lambda

        # Reconstruction-inference sampler (XC.1). The score model can run a
        # CONDITIONED posterior reconstruction through the SamplerRegistry,
        # mirroring the cold-diffusion generator, so a YAML ``inference_sampler``
        # (e.g. ``dds``) is honoured end-to-end rather than silently bypassed.
        # Validated against ``SamplerRegistry`` at sample-call time via
        # ``get_sampler`` (and at config time by ``check_inference_sampler_registered``).
        self._sampler_name = str(kwargs.get("inference_sampler") or kwargs.get("sampler") or "ddim")
        self._sampling_steps = int(kwargs.get("sampling_steps", 50))
        # Sampler-specific knobs threaded from model_kwargs to get_sampler at
        # inference (e.g. DynamicDPS guidance_mode/guidance_scale/sigma; DDS
        # cg_iters/cg_lambda). Only forwarded if present — the sampler validates.
        self._sampler_kwargs = {
            k: kwargs[k]
            for k in (
                "guidance_mode",
                "guidance_scale",
                "sigma",
                "n_coils",
                "cg_iters",
                "cg_lambda",
                "eta",
            )
            if k in kwargs
        }

        # Initialize diffusion buffers via DiffusionMixin
        self._initialize_diffusion_buffers(timesteps, beta_schedule)

        # Move model to device
        self.to(device)

        logger.debug(
            "Initialized ScoreBasedDiffusionGenerator: timesteps=%d, schedule=%s",
            timesteps,
            beta_schedule,
        )

    @property
    def name(self) -> str:
        """Return the model name."""
        return "ScoreBasedDiffusionGenerator"

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
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

        Uses DDPM formula: x_0 = (x_t - sqrt(1-α̅_t) * noise) / sqrt(α̅_t)
        """
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_t.shape,
        )
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)

        return (x_t - sqrt_one_minus_alpha_cumprod * noise) / sqrt_alpha_cumprod

    def _model_forward(self, x: Tensor, t: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass of the denoising model.

        Handles models that accept timestep (diffusion models) and
        those that don't (standard UNets).
        """
        try:
            return self.denoising_model(x, t)
        except TypeError:
            # Model doesn't accept timestep
            return self.denoising_model(x)

    # -------------------------------------------------------------------------
    # IGenerator Interface
    # -------------------------------------------------------------------------
    def forward(self, x: Tensor, timesteps: Tensor | None = None) -> Tensor:
        """Predict noise; self-noise a clean input only when no timestep is given.

        Args:
            x: The noised sample x_t (plus any channel-concatenated condition) when
                ``timesteps`` is given, else a clean image of shape (B, C, H, W).
            timesteps: Diffusion step per sample. The strategy owns the forward
                process and passes its own t, so x must not be noised again here.

        Returns:
            Predicted noise.
        """
        # Shape guard
        if x.dim() != 4:
            msg = f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D"
            raise ValueError(msg)
        if self.sde_type is not None and timesteps is None:
            refuse_discrete_path(self, "the self-noising DDPM forward (no timesteps)")

        if timesteps is not None:
            return self._model_forward(x, timesteps)

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device)

        # Forward diffusion via DiffusionMixin
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t, noise)

        # Predict noise using model wrapper
        return self._model_forward(x_t, t)

    def generate(self, z: Tensor, **kwargs: Any) -> Tensor:
        """Generate samples using score-based diffusion.

        Args:
            z: Input tensor (used to determine output shape).
            **kwargs: Additional arguments.

        Returns:
            Generated tensor.
        """
        if self.sde_type is not None:
            refuse_discrete_path(self, "generate (discrete p_sample_loop)")
        return self.p_sample_loop(z.shape, **kwargs)

    def sample(
        self,
        measurement: Tensor,
        mask: Tensor | None = None,
        inference_timesteps: int | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Posterior reconstruction from a measurement via the SamplerRegistry (XC.1).

        Routes through ``get_sampler(self._sampler_name, model=self, ...)`` — the
        same World-A convention the cold-diffusion generator uses — so a YAML
        ``inference_sampler`` such as ``dds`` (Decomposed Diffusion Sampling) is
        consumed end-to-end. The configured sampler must be a *posterior /
        reconstruction* sampler accepting ``(measurement, mask)`` (e.g. ``dds``,
        ``dps_mri``, ``pi_gdm``); an unconditional sampler fails loud (no silent
        fallback, pitfall #9). The ``DiffusionTrainingStrategy`` multistep
        validation path invokes this when ``validation.multistep_cold_sampling``
        is set and the generator exposes ``sample``.

        Args:
            measurement: undersampled k-space ``y`` (complex, ``[B, C, H, W]``).
            mask: acquisition mask; ``None`` lets the sampler infer it.
            inference_timesteps: override the reverse-step count.
            device: target device (defaults to the measurement's device).

        Returns:
            The reconstructed image.
        """
        del device, kwargs  # device follows the measurement; no extra kwargs needed.
        if self.sde_type is not None:
            refuse_discrete_path(self, f"the {self._sampler_name!r} posterior sampler")
        from spectramr.models.diffusion.samplers import get_sampler

        steps = int(inference_timesteps or self._sampling_steps)
        self.eval()
        with torch.no_grad():
            sampler = get_sampler(
                self._sampler_name,
                model=self,
                num_timesteps=self.timesteps,
                sampling_steps=steps,
                **self._sampler_kwargs,
            )
            return sampler.sample(measurement, mask)

    def get_parameter_count(self) -> int:
        """Count total parameters in the denoising model."""
        return sum(p.numel() for p in self.denoising_model.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        # Simple identity for same-resolution generation
        return input_shape
