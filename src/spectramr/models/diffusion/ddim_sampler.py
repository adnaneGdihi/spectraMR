"""DDIM Sampler Implementation
===========================

Denoising Diffusion Implicit Models (DDIM) sampler for faster and higher
quality sampling compared to DDPM. Based on the paper "Denoising Diffusion
Implicit Models" by Song et al. (https://arxiv.org/abs/2010.02502)

This implementation provides:
- Faster sampling with fewer steps
- Higher quality samples
- Deterministic sampling (when eta=0)
- Configurable eta parameter for stochasticity control
"""

from collections.abc import Callable

import torch

from spectramr.infrastructure.physics.data_consistency import SimpleDataConsistency
from spectramr.models.diffusion.samplers import register_sampler

from .base_diffusion import Diffusion


@register_sampler(name="ddim", aliases=["DDIM"])
class DDIMSampler:
    """DDIM Sampler for generating samples from diffusion models.

    DDIM allows for faster sampling by taking larger steps in the reverse process
    while maintaining sample quality. The eta parameter controls the amount of
    stochasticity in the sampling process.

    Args:
        diffusion: Base diffusion process (DDPM-style)
        eta: Controls stochasticity (0.0 = deterministic, 1.0 = DDPM-like)
        use_clipped_model_output: Whether to clip model output for stability

    """

    def __init__(
        self,
        diffusion: Diffusion,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        dc_weight: float = 0.0,
    ):
        """__init__.

        Args:
            diffusion (Diffusion): Description.
            eta (float): Description.
            use_clipped_model_output (bool): Description.
            dc_weight (float): Description.
        """
        self.diffusion = diffusion
        self.eta = eta
        self.use_clipped_model_output = use_clipped_model_output
        self.dc_weight = dc_weight

        # Initialize data consistency module if weight > 0
        self.data_consistency = SimpleDataConsistency(weight=dc_weight) if dc_weight > 0 else None

        # Pre-compute DDIM parameters
        self._precompute_ddim_parameters()

    def _precompute_ddim_parameters(self):
        """Pre-compute parameters needed for DDIM sampling."""
        # DDIM parameters (equations from the paper)
        device = self.diffusion.device
        self.alphas_cumprod = self.diffusion.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.diffusion.alphas_cumprod_prev.to(device)

        # DDIM-specific coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For DDIM sampling
        self.ddim_alphas = self.alphas_cumprod / self.alphas_cumprod_prev
        self.ddim_sigmas = (
            self.eta
            * torch.sqrt((1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
            * torch.sqrt(1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )

    def _get_ddim_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Get timesteps for DDIM sampling.

        Args:
            num_inference_steps: Number of sampling steps

        Returns:
            Timesteps tensor

        """
        step_ratio = self.diffusion.timesteps // num_inference_steps
        timesteps = (
            torch.arange(
                0,
                num_inference_steps,
                device=self.diffusion.device,
                dtype=torch.long,
            )
            * step_ratio
        )
        return timesteps.flip(0)  # Reverse order for sampling

    def _get_prev_timestep(
        self,
        timestep: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Get the previous timestep for DDIM sampling."""
        step_ratio = self.diffusion.timesteps // num_inference_steps
        prev_timestep = timestep - step_ratio
        return torch.clamp(prev_timestep, min=0)

    @torch.no_grad()
    def sample(
        self,
        model: Callable,
        shape: torch.Size,
        num_inference_steps: int = 50,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
        return_intermediates: bool = False,
        measured_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple:
        """Generate samples using DDIM sampling.

        Args:
            model: Denoising model (should accept x, t, cond parameters)
            shape: Shape of the output tensor
                   (batch_size, channels, height, width)
            num_inference_steps: Number of sampling steps (fewer = faster)
            cond: Conditional input for guided generation
            guidance_scale: Scale for classifier-free guidance
            generator: Random number generator for reproducibility
            return_intermediates: Whether to return intermediate steps
            measured_kspace: Measured k-space data for data consistency (optional)
            mask: Sampling mask for data consistency (optional)

        Returns:
            Generated samples, optionally with intermediate steps

        """
        # Prefer the conditioner's device when available — the diffusion
        # process may have been initialised on CPU but a strategy can
        # later ``.to(cuda)`` the wrapping module. Stale ``self.diffusion.device``
        # then leads to ``cuda`` cond + ``cpu`` samples (cf. May 2026
        # padnet multicontrast cluster failure: "tensors on cuda:0,
        # different from other tensors on cpu" in wrapper_CUDA_cat).
        if cond is not None and cond.device.type != "meta":
            device = cond.device
        elif measured_kspace is not None and measured_kspace.device.type != "meta":
            device = measured_kspace.device
        else:
            device = self.diffusion.device
        batch_size = shape[0]

        # DDIMSampler is not an nn.Module — its precomputed alpha/sigma
        # tensors don't follow ``.to(device)`` on the wrapping module.
        # If a strategy moved padnet (or any sampler-owning model) to cuda
        # after sampler construction, ``_extract`` would do an ``a.gather``
        # on cpu vs cuda-resident ``x`` and crash with
        # ``Expected all tensors to be on the same device, but found
        # cuda:0 and cpu`` (2026-05-10 cluster smoke: padnet_multicontrast).
        if self.alphas_cumprod.device != device:
            self.alphas_cumprod = self.alphas_cumprod.to(device)
            self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
            self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
            self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
            self.ddim_alphas = self.ddim_alphas.to(device)
            self.ddim_sigmas = self.ddim_sigmas.to(device)

        # Start from pure noise
        x = torch.randn(shape, device=device, generator=generator)

        if return_intermediates:
            intermediates = [x.clone()]

        # Get DDIM timesteps
        timesteps = self._get_ddim_timesteps(num_inference_steps)

        for i, t in enumerate(timesteps):
            # Get current timestep index
            t_index = t.item()

            # Create timestep tensor
            t_tensor = torch.full(
                (batch_size,),
                t_index,
                device=device,
                dtype=torch.long,
            )

            # Predict noise using model
            if guidance_scale != 1.0 and cond is not None:
                # Classifier-free guidance
                # Unconditional prediction
                try:
                    noise_uncond = model(x, t_tensor, cond=None)
                except TypeError:
                    noise_uncond = model(x, t_tensor)

                # Conditional prediction
                try:
                    noise_cond = model(x, t_tensor, cond=cond)
                except TypeError:
                    noise_cond = model(x, t_tensor)

                # Combine predictions
                predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                # Standard prediction
                try:
                    predicted_noise = model(x, t_tensor, cond=cond)
                except TypeError:
                    predicted_noise = model(x, t_tensor)

            # Ensure predicted_noise is on the correct device
            predicted_noise = predicted_noise.to(device)

            # Clip model output if requested
            if self.use_clipped_model_output:
                predicted_noise = torch.clamp(predicted_noise, -1.0, 1.0)

            # DDIM sampling step
            x = self._ddim_step(x, predicted_noise, t_tensor, num_inference_steps)

            # Apply data consistency if available
            if self.data_consistency is not None and measured_kspace is not None:
                x = self.data_consistency(x, measured_kspace, mask)

            if return_intermediates:
                intermediates.append(x.clone())

        if return_intermediates:
            return x, intermediates
        return x

    def _ddim_step(
        self,
        x: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: torch.Tensor,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Single DDIM sampling step.

        Args:
            x: Current sample
            predicted_noise: Model's noise prediction
            t: Current timestep
            num_inference_steps: Total number of inference steps

        Returns:
            Updated sample for next step

        """
        # Get current timestep values
        alphas_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        alphas_cumprod_prev_t = self._extract(self.alphas_cumprod_prev, t, x.shape)

        # Get DDIM-specific values
        ddim_sigma_t = self._extract(self.ddim_sigmas, t, x.shape)

        # Predict x_0 (original image) from noise prediction
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x.shape,
        )
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x.shape)

        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t

        # Clip predicted x0 for stability
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alphas_cumprod_prev_t - ddim_sigma_t**2) * predicted_noise

        # Compute x_{t-1}
        x_prev = (
            torch.sqrt(alphas_cumprod_prev_t)
            * (
                (x - torch.sqrt(1.0 - alphas_cumprod_t) * predicted_noise)
                / torch.sqrt(alphas_cumprod_t)
            )
            + dir_xt
        )

        # Add noise if eta > 0
        if self.eta > 0:
            noise = torch.randn_like(x)
            x_prev = x_prev + ddim_sigma_t * noise

        return x_prev

    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: torch.Size,
    ) -> torch.Tensor:
        """Extract values from tensor a at indices t, broadcasting to
        x_shape.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DDIMSamplerConfig:
    """Configuration class for DDIM sampler.

    Args:
        eta: Controls stochasticity (0.0 = deterministic, 1.0 = DDPM-like)
        num_inference_steps: Number of sampling steps
        use_clipped_model_output: Whether to clip model output
        guidance_scale: Scale for classifier-free guidance
        dc_weight: Weight for data consistency projection (0.0 = disabled)

    """

    def __init__(
        self,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: bool = False,
        guidance_scale: float = 1.0,
        dc_weight: float = 0.0,
    ):
        """__init__.

        Args:
            eta (float): Description.
            num_inference_steps (int): Description.
            use_clipped_model_output (bool): Description.
            guidance_scale (float): Description.
            dc_weight (float): Description.
        """
        self.eta = eta
        self.num_inference_steps = num_inference_steps
        self.use_clipped_model_output = use_clipped_model_output
        self.guidance_scale = guidance_scale
        self.dc_weight = dc_weight

        # Validate parameters
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be between 0.0 and 1.0")
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if guidance_scale < 1.0:
            raise ValueError("guidance_scale must be >= 1.0")
        if dc_weight < 0.0:
            raise ValueError("dc_weight must be >= 0.0")


def create_ddim_sampler(
    diffusion: Diffusion,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    dc_weight: float = 0.0,
) -> DDIMSampler:
    """Factory function to create a DDIM sampler.

    Args:
        diffusion: Base diffusion process
        eta: Controls stochasticity (0.0 = deterministic, 1.0 = DDPM-like)
        use_clipped_model_output: Whether to clip model output
        dc_weight: Weight for data consistency projection

    Returns:
        Configured DDIM sampler

    """
    return DDIMSampler(
        diffusion=diffusion,
        eta=eta,
        use_clipped_model_output=use_clipped_model_output,
        dc_weight=dc_weight,
    )


def create_fast_ddim_sampler(diffusion: Diffusion) -> DDIMSampler:
    """Create a fast DDIM sampler with optimized settings for speed.

    Args:
        diffusion: Base diffusion process

    Returns:
        Fast DDIM sampler (deterministic, 20 steps)

    """
    return DDIMSampler(
        diffusion=diffusion,
        eta=0.0,  # Deterministic for speed
        use_clipped_model_output=True,  # Stability
    )


def create_quality_ddim_sampler(diffusion: Diffusion) -> DDIMSampler:
    """Create a high-quality DDIM sampler with optimized settings for quality.

    Args:
        diffusion: Base diffusion process

    Returns:
        Quality DDIM sampler (50 steps, slight stochasticity)

    """
    return DDIMSampler(
        diffusion=diffusion,
        eta=0.1,  # Slight stochasticity for quality
        use_clipped_model_output=True,  # Stability
    )
