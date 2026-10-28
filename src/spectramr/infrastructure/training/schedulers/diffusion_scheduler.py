"""Diffusion Scheduler Module

This module provides the DiffusionScheduler for managing noise scheduling,
timestep sampling, and diffusion process utilities.
"""

import torch
from torch import nn


class DiffusionScheduler:
    """Handles diffusion noise scheduling and sampling."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: torch.device | None = None,
    ):
        """Initialize diffusion parameters.

        Args:
            num_timesteps: Number of diffusion timesteps
            beta_schedule: Type of beta schedule ('linear' or 'cosine')
            device: Device for tensor operations
        """
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device or torch.device("cpu")

        # Setup noise schedule
        self._setup_noise_schedule()

    def _setup_noise_schedule(self):
        # Beta schedule types are simple strings, not sampler types
        """_setup_noise_schedule.

        Returns:
            Any: Description.
        """
        BETA_LINEAR = "linear"
        BETA_COSINE = "cosine"

        # Pitfall #9 — raise on an unsupported schedule rather than silently
        # coercing to ``linear``. The old coercion meant a typo'd or
        # not-yet-implemented schedule (e.g. ``sqrt`` / ``quadratic``, both
        # advertised by ``training.noise_schedule``) trained on a linear
        # schedule with no trace.
        if self.beta_schedule not in {BETA_LINEAR, BETA_COSINE}:
            raise ValueError(
                f"Unknown beta schedule {self.beta_schedule!r}. "
                f"Supported: {sorted({BETA_LINEAR, BETA_COSINE})}."
            )

        if self.beta_schedule == BETA_LINEAR:
            self.betas = torch.linspace(
                self.beta_start,
                self.beta_end,
                self.num_timesteps,
                device=self.device,
            )
        elif self.beta_schedule == BETA_COSINE:
            t = torch.linspace(
                0,
                self.num_timesteps,
                self.num_timesteps + 1,
                device=self.device,
            )
            alphas_cumprod = torch.cos(0.5 * torch.pi * t / self.num_timesteps) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = torch.clamp(self.betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")

        # Calculate cumulative products
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for diffusion.

        Args:
            batch_size: Number of samples in batch

        Returns:
            Timestep tensor of shape [batch_size]

        """
        return torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion process: q(x_t | x_0).

        Args:
            x_start: Clean input tensor
            t: Timestep tensor
            noise: Optional noise tensor

        Returns:
            Noisy tensor at timestep t

        """
        # Ensure all inputs are on the same device
        device = self.device
        x_start = x_start.to(device, non_blocking=True)
        t = t.to(device=device, dtype=torch.long)

        if noise is None:
            noise = torch.randn_like(x_start, device=device)
        else:
            noise = noise.to(device, non_blocking=True)

        # Ensure schedule tensors are on the correct device
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device, non_blocking=True)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device, non_blocking=True
        )

        sqrt_alphas_cumprod_t = self._extract(
            sqrt_alphas_cumprod,
            t,
            x_start.shape,
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape,
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract(
        self,
        tensor: torch.Tensor,
        t: torch.Tensor,
        x_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Extract values from tensor at given timesteps.

        Args:
            tensor: Source tensor (1D schedule values)
            t: Timestep indices (shape: [batch_size])
            x_shape: Target shape to broadcast to (e.g., [batch, channels, height, width])

        Returns:
            Extracted tensor with proper shape for broadcasting

        """
        # Ensure t is on the correct device and dtype
        t = t.to(device=tensor.device, dtype=torch.long)

        # Ensure tensor is on the correct device
        tensor = tensor.to(device=self.device)

        # Also move t to the target device after gathering
        batch_size = t.shape[0]

        # No clamp: a timestep outside [0, T) comes from a different schedule
        # (#2306), and a clamp answered it with the last entry. gather's own
        # bounds check raises instead, without a host sync per step.
        out = tensor.gather(-1, t)
        # Reshape to (batch_size, 1, 1, ...) to match all spatial/channel dimensions
        # This ensures proper broadcasting for all tensor dimensions
        return (
            out.contiguous()
            .view(batch_size, *((1,) * (len(x_shape) - 1)))
            .to(self.device, non_blocking=True)
        )

    def get_diffusion_schedule_info(self) -> dict[str, torch.Tensor]:
        """Get information about the diffusion schedule.

        Returns:
            Dictionary with schedule tensors

        """
        return {
            "betas": self.betas,
            "alphas": self.alphas,
            "alphas_cumprod": self.alphas_cumprod,
            "sqrt_alphas_cumprod": self.sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": self.sqrt_one_minus_alphas_cumprod,
        }

    def compute_diffusion_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diffusion loss.

        Args:
            model_output: Model prediction
            target: Target tensor
            timesteps: Timestep tensor
            noise: Original noise tensor

        Returns:
            Loss tensor

        """
        # Simple MSE loss for diffusion
        return nn.functional.mse_loss(model_output, noise)

    def get_noise_prediction(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Extract noise prediction from model output.

        Default implementation assumes model directly predicts noise.
        Subclasses can override for different prediction formats.

        Args:
            model_output: Raw model output
            x_t: Noisy input at timestep t
            timesteps: Timestep tensor

        Returns:
            Predicted noise tensor

        """
        # Default: assume model output is predicted noise
        return model_output

    def denoise_step(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Single denoising step.

        Args:
            x_t: Noisy tensor at timestep t
            predicted_noise: Predicted noise
            timesteps: Timestep tensor

        Returns:
            Denoised tensor at timestep t-1

        """
        # Ensure all inputs are on the same device
        device = self.device
        x_t = x_t.to(device, non_blocking=True)
        predicted_noise = predicted_noise.to(device, non_blocking=True)
        timesteps = timesteps.to(device=device, dtype=torch.long)

        alphas_t = self._extract(self.alphas.to(device, non_blocking=True), timesteps, x_t.shape)
        alphas_cumprod_t = self._extract(
            self.alphas_cumprod.to(device, non_blocking=True),
            timesteps,
            x_t.shape,
        )

        alphas_cumprod_prev_full = torch.cat(
            [
                torch.ones(1, device=device),
                self.alphas_cumprod[:-1].to(device, non_blocking=True),
            ]
        )
        alphas_cumprod_prev = self._extract(
            alphas_cumprod_prev_full,
            timesteps,
            x_t.shape,
        )

        # DDIM deterministic projection of x0
        pred_x0 = (x_t - torch.sqrt(1.0 - alphas_cumprod_t) * predicted_noise) / torch.sqrt(
            alphas_cumprod_t
        )

        # Option: strictly follow analytical DDPM step with stochastic tracking
        # posterior variance sigma_t^2
        posterior_variance = (
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod_t) * (1.0 - alphas_t)
        )

        # Determine if we inject noise (stochastic DDPM regime)
        # For DDIM this would be zero. We default to DDPM here as per exact variance schedule specs
        noise = torch.randn_like(x_t, device=device)
        # No noise when t == 0
        nonzero_mask = (timesteps > 0).float()
        nonzero_mask = nonzero_mask.view(-1, *([1] * (len(x_t.shape) - 1)))

        # DDPM step
        model_mean = (1.0 / torch.sqrt(alphas_t)) * (
            x_t - ((1.0 - alphas_t) / torch.sqrt(1.0 - alphas_cumprod_t)) * predicted_noise
        )

        x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

        return x_prev
