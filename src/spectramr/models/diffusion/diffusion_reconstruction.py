"""Diffusion prior-based reconstruction utilities.

This module provides Plug-and-Play (PnP) and Regularization by Denoising (RED)
reconstruction algorithms that leverage pre-trained diffusion models as priors
for solving inverse problems in MRI reconstruction.
"""

from collections.abc import Callable

import torch
from torch import nn

from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.models.registry import register_model


class DiffusionPrior(nn.Module):
    """Wrapper for diffusion models used as priors in reconstruction."""

    def __init__(
        self,
        diffusion_model: nn.Module,
        noise_scheduler: Callable,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ):
        """__init__.

        Args:
            diffusion_model (nn.Module): Description.
            noise_scheduler (Callable): Description.
            num_inference_steps (int): Description.
            guidance_scale (float): Description.
        """
        super().__init__()
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single denoising step using the diffusion prior.

        Args:
            x_t: Noisy input at timestep t [B, C, H, W]
            t: Timestep tensor [B]
            conditioning: Optional conditioning information

        Returns:
            Denoised output [B, C, H, W]

        """
        # Forward pass through diffusion model with timestep conditioning
        with torch.no_grad():
            noise_pred = self.diffusion_model(x_t, t, conditioning)

        # Noise schedule parameters
        alpha_t = self.noise_scheduler.alpha_t(t)
        alpha_t_prev = self.noise_scheduler.alpha_t_prev(t)
        sigma_t = self.noise_scheduler.sigma_t(t)

        # Reshape for broadcasting
        view_shape = (-1, *([1] * (x_t.ndim - 1)))
        alpha_t = alpha_t.view(view_shape)
        alpha_t_prev = alpha_t_prev.view(view_shape)
        sigma_t = sigma_t.view(view_shape)

        # Predict x_0 via Tweedie's formula: x_0 = (x_t - sqrt(1-α_t)*ε) / sqrt(α_t)
        x_0_hat = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # Apply classifier-free guidance if specified
        if self.guidance_scale != 1.0 and conditioning is not None:
            noise_pred_uncond = self.diffusion_model(x_t, t, None)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)
            # Recompute x_0 with guided noise prediction
            x_0_hat = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        # DDPM posterior mean: x_{t-1} = sqrt(α_{t-1}) * x_0_hat + sqrt(1-α_{t-1}) * ε
        x_t_prev = torch.sqrt(alpha_t_prev) * x_0_hat + sigma_t * torch.randn_like(x_t)

        return x_t_prev

    def sample_prior(
        self,
        shape: tuple[int, ...],
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the diffusion prior.

        Args:
            shape: Output shape (B, C, H, W)
            conditioning: Optional conditioning information

        Returns:
            Sample from prior [B, C, H, W]

        """
        x_t = torch.randn(shape, device=self.diffusion_model.device)

        for t in reversed(range(self.num_inference_steps)):
            t_tensor = torch.full((shape[0],), t, device=x_t.device, dtype=torch.long)
            x_t = self.denoise_step(x_t, t_tensor, conditioning)

        return x_t


class DataConsistency(nn.Module):
    """Data consistency module for reconstruction."""

    def __init__(self, forward_operator: Callable, lambda_dc: float = 1.0):
        """__init__.

        Args:
            forward_operator (Callable): Description.
            lambda_dc (float): Description.
        """
        super().__init__()
        self.forward_operator = forward_operator
        self.lambda_dc = lambda_dc

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply data consistency step.

        Args:
            x: Current reconstruction estimate [B, C, H, W]
            y: Measured k-space data [B, C, H, W] (complex)
            mask: Optional sampling mask [B, 1, H, W]

        Returns:
            Data-consistent reconstruction [B, C, H, W]

        forward method for DataConsistency.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Forward model: x -> k-space
        x_k = self.forward_operator(x)  # FFT or NUFFT

        # Replace measured data where available
        if mask is not None:
            x_k = mask * y + (1 - mask) * x_k
        else:
            x_k = y

        # Inverse transform back to image space using physics module
        x_dc = ifft2c(x_k).real

        # Weighted combination
        x_updated = x + self.lambda_dc * (x_dc - x)

        return x_updated


class PnPReconstruction(nn.Module):
    """Plug-and-Play reconstruction using diffusion prior."""

    def __init__(
        self,
        diffusion_prior: DiffusionPrior,
        data_consistency: DataConsistency,
        num_iterations: int = 10,
        lambda_reg: float = 0.1,
    ):
        """__init__.

        Args:
            diffusion_prior (DiffusionPrior): Description.
            data_consistency (DataConsistency): Description.
            num_iterations (int): Description.
            lambda_reg (float): Description.
        """
        super().__init__()
        self.diffusion_prior = diffusion_prior
        self.data_consistency = data_consistency
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg

    def denoise_current_estimate(self, x: torch.Tensor, timestep: int = 10) -> torch.Tensor:
        """Denoise the current estimate using the diffusion prior.

        This acts as the proximal operator for the regularization term.
        The timestep controls how aggressively to denoise (higher = more aggressive).

        Args:
            x: Current reconstruction estimate [B, C, H, W]
            timestep: Diffusion timestep (higher = more noise assumed)

        Reference: Chung, H., et al. "Diffusion Posterior Sampling." ICLR 2023.
        """
        t_step = torch.full((x.shape[0],), timestep, device=x.device, dtype=torch.long)

        with torch.no_grad():
            # Predict noise
            noise_pred = self.diffusion_prior.diffusion_model(x, t_step)

            # Tweedie's Formula: Estimate x_0 from x_t
            # x_0 = (x_t - sqrt(1-alpha)*eps) / sqrt(alpha)
            alpha_t = self.diffusion_prior.noise_scheduler.alpha_t(t_step)

            # Reshape for broadcasting
            alpha_t = alpha_t.view(-1, *([1] * (x.ndim - 1)))

            x_denoised = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

        return x_denoised

    def reconstruct(
        self,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform PnP reconstruction with time-dependent denoising.

        Args:
            y: Measured k-space data [B, C, H, W] (complex)
            mask: Sampling mask [B, 1, H, W]
            conditioning: Optional conditioning for diffusion prior
            x_init: Initial reconstruction estimate [B, C, H, W]

        Returns:
            Reconstructed image [B, C, H, W]

        Note:
            Uses annealed denoising: timestep decreases from high to low across
            iterations, allowing hierarchical detail recovery (global -> fine).
        """
        if x_init is None:
            # Initialize with zero-filled reconstruction using physics module
            x = ifft2c(y).real
        else:
            x = x_init

        # Annealed timesteps: decay from high (200) to low (10) across iterations
        # This captures global structure first, then fine details
        t_max, t_min = 200, 10
        timesteps = torch.linspace(t_max, t_min, self.num_iterations).long()

        for i in range(self.num_iterations):
            # 1. Regularization step (Denoising with time-dependent σ)
            # Higher timestep = assume more noise = denoise more aggressively
            t_current = int(timesteps[i].item())
            x_reg = self.denoise_current_estimate(x, timestep=t_current)

            # 2. Data consistency step
            x = self.data_consistency(x_reg, y, mask)

        return x


class REDReconstruction(nn.Module):
    """Regularization by Denoising (RED) reconstruction."""

    def __init__(
        self,
        denoiser: Callable,
        data_consistency: DataConsistency,
        num_iterations: int = 10,
        lambda_reg: float = 0.1,
        noise_level: float = 0.1,
    ):
        """__init__.

        Args:
            denoiser (Callable): Description.
            data_consistency (DataConsistency): Description.
            num_iterations (int): Description.
            lambda_reg (float): Description.
            noise_level (float): Description.
        """
        super().__init__()
        self.denoiser = denoiser
        self.data_consistency = data_consistency
        self.num_iterations = num_iterations
        self.lambda_reg = lambda_reg
        self.noise_level = noise_level

    def reconstruct(
        self,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        x_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform RED reconstruction.

        Args:
            y: Measured k-space data [B, C, H, W] (complex)
            mask: Sampling mask [B, 1, H, W]
            x_init: Initial reconstruction estimate [B, C, H, W]

        Returns:
            Reconstructed image [B, C, H, W]

        """
        if x_init is None:
            # Initialize with zero-filled reconstruction using physics module
            x = ifft2c(y).real
        else:
            x = x_init

        for _i in range(self.num_iterations):
            # Add noise for regularization by denoising
            noise = torch.randn_like(x) * self.noise_level
            x_noisy = x + noise

            # Denoising step
            x_denoised = self.denoiser(x_noisy)

            # Data consistency step
            x = self.data_consistency(x_denoised, y, mask)

        return x


class DiffusionPosteriorSampling(nn.Module):
    """Posterior sampling using diffusion models for reconstruction."""

    def __init__(
        self,
        diffusion_model: nn.Module,
        noise_scheduler: Callable,
        measurement_operator: Callable,
        num_posterior_steps: int = 100,
        eta: float = 0.5,
    ):
        """__init__.

        Args:
            diffusion_model (nn.Module): Description.
            noise_scheduler (Callable): Description.
            measurement_operator (Callable): Description.
            num_posterior_steps (int): Description.
            eta (float): Description.
        """
        super().__init__()
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler
        self.measurement_operator = measurement_operator
        self.num_posterior_steps = num_posterior_steps
        self.eta = eta

    def reconstruct(
        self,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform posterior sampling reconstruction.

        Args:
            y: Measured data [B, C, H, W]
            mask: Optional measurement mask

        Returns:
            Reconstructed image [B, C, H, W]

        """
        # Start from random noise
        x = torch.randn_like(y)

        for t in range(self.num_posterior_steps):
            # Current timestep
            t_tensor = torch.full((y.shape[0],), t, device=y.device, dtype=torch.long)

            # 1. Predict noise (Score estimation)
            # We need gradients for the correction step, so we enable grad for x
            with torch.enable_grad():
                x = x.detach().requires_grad_(True)
                noise_pred = self.diffusion_model(x, t_tensor)

                # 2. Calculate Likelihood Gradient (Data Consistency Term)
                # Estimate x_0 from x_t (Tweedie's formula approximation)
                alpha_t = self.noise_scheduler.alpha_t(t_tensor)
                # Reshape alpha_t for broadcasting
                alpha_t_view = alpha_t.view(-1, *([1] * (x.ndim - 1)))

                x_0_hat = (x - torch.sqrt(1 - alpha_t_view) * noise_pred) / torch.sqrt(alpha_t_view)

                # Forward projection A(x_0_hat)
                # We assume measurement_operator handles the physics (FFT, Sensitivity, etc.)
                k_pred = self.measurement_operator(x_0_hat)

                # L2 Data Fidelity: || M * (A(x) - y) ||^2
                # We calculate this only where we have data (mask)
                if mask is not None:
                    # Ensure mask is broadcastable and correct type
                    mask_bool = mask.bool() if mask.dtype != torch.bool else mask
                    # Residual in k-space
                    residual = k_pred - y
                    # Apply mask
                    masked_residual = torch.where(mask_bool, residual, torch.zeros_like(residual))
                    # Loss
                    fidelity_loss = torch.norm(masked_residual) ** 2
                else:
                    fidelity_loss = torch.norm(k_pred - y) ** 2

                # Gradient of likelihood w.r.t x_t
                # We want to minimize fidelity loss, so we move in direction of -grad
                grad_fidelity = torch.autograd.grad(fidelity_loss, x)[0]

            # 3. Correction Step (Manifold Constrained Gradient)
            # Scale gradient by noise variance or a learned parameter
            # For standard MCG, we often scale by 1/sigma_t or similar.
            # Here we use a simplified step size control.
            # correction_scale can be a hyperparameter.
            # We'll use a heuristic based on signal scale or just a fixed small step for now.
            # Ideally this should be exposed as a hyperparameter.
            # TODO: Expose correction_scale as a configurable hyperparameter
            correction_scale = 1.0 / (
                torch.norm(grad_fidelity) + 1e-8
            )  # Normalize gradient roughly
            # Or better, use the noise schedule variance
            sigma_t = self.noise_scheduler.sigma_t(t_tensor)
            sigma_t_view = sigma_t.view(-1, *([1] * (x.ndim - 1)))

            # Weighted correction: stronger correction at lower noise levels?
            # Actually, at high noise (t large), x_0_hat is poor, so trust prior more.
            # At low noise (t small), x_0_hat is good, trust data more.
            # Common heuristic: step_size ~ 1/||grad|| * scale

            # We will apply the correction to the noise prediction or directly to x
            # Standard MCG modifies the score: score_new = score_prior - scale * grad_likelihood
            # noise_pred is -sigma * score. So:
            # noise_pred_new = noise_pred + sigma * scale * grad_likelihood

            # Let's use the formulation where we adjust x directly in the update step
            # or adjust the estimated score.

            # Adjusting noise_pred (effectively guiding the diffusion):
            # We want to move x towards lower fidelity loss.
            # The score points towards higher data density.
            # The likelihood gradient points towards higher data consistency.
            # We combine them.

            # Using a fixed guidance scale for now, can be parameterized later
            # This is similar to classifier guidance
            guidance_strength = 0.1  # This should ideally be a class attribute

            # noise_pred corresponds to -sigma * score
            # We want score_total = score_prior + w * score_likelihood
            # score_likelihood = - grad_fidelity
            # So score_total = score_prior - w * grad_fidelity
            # noise_pred_total = -sigma * score_total = noise_pred + sigma * w * grad_fidelity

            noise_pred = noise_pred + sigma_t_view * guidance_strength * grad_fidelity

            # 4. Posterior Sampling Step (Standard Reverse SDE/DDPM)
            alpha_t_prev = self.noise_scheduler.alpha_t_prev(t_tensor)
            alpha_t_prev = alpha_t_prev.view(-1, *([1] * (x.ndim - 1)))
            sigma_t = sigma_t.view(-1, *([1] * (x.ndim - 1)))

            # Direction pointing to x_t
            # Note: We detach everything here to stop gradients accumulating
            x = x.detach()
            noise_pred = noise_pred.detach()

            dir_xt = torch.sqrt(1 - alpha_t_prev - self.eta * sigma_t**2) * (
                (x - torch.sqrt(1 - alpha_t_view) * noise_pred) / torch.sqrt(alpha_t_view)
            )

            # Random noise
            noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

            # Update x
            x = dir_xt + torch.sqrt(self.eta * sigma_t**2) * noise

        return x


@register_model(name="diffusion_reconstruction", training_mode="reconstruction")
class ReconstructionWithDiffusionPrior(nn.Module):
    """High-level interface for diffusion prior-based reconstruction."""

    def __init__(
        self,
        method: str = "pnp",
        diffusion_model: nn.Module | None = None,
        noise_scheduler: Callable | None = None,
        forward_operator: Callable | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            method (str): Description.
            diffusion_model (Optional[nn.Module]): Description.
            noise_scheduler (Optional[Callable]): Description.
            forward_operator (Optional[Callable]): Description.
        """
        super().__init__()
        self.method = method

        if method == "pnp":
            # Filter kwargs for DiffusionPrior
            prior_kwargs = {
                k: v for k, v in kwargs.items() if k in ["num_inference_steps", "guidance_scale"]
            }
            diffusion_prior = DiffusionPrior(diffusion_model, noise_scheduler, **prior_kwargs)
            data_consistency = DataConsistency(forward_operator)
            self.reconstructor = PnPReconstruction(
                diffusion_prior,
                data_consistency,
                **kwargs,
            )

        elif method == "red":
            data_consistency = DataConsistency(forward_operator)
            self.reconstructor = REDReconstruction(
                diffusion_model,
                data_consistency,
                **kwargs,
            )

        elif method == "posterior":
            self.reconstructor = DiffusionPosteriorSampling(
                diffusion_model,
                noise_scheduler,
                forward_operator,
                **kwargs,
            )

        else:
            raise ValueError(f"Unknown reconstruction method: {method}")

    def forward(
        self,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Perform reconstruction.

        Args:
            y: Measured data [B, C, H, W]
            mask: Optional measurement mask
            **kwargs: Additional arguments for reconstruction

        Returns:
            Reconstructed image [B, C, H, W]

        """
        return self.reconstructor.reconstruct(y, mask, **kwargs)


def create_diffusion_reconstruction_config(
    method: str = "pnp",
    num_iterations: int = 10,
    lambda_reg: float = 0.1,
    lambda_dc: float = 1.0,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    eta: float = 0.5,
) -> dict[str, str | int | float]:
    """Create configuration for diffusion-based reconstruction.

    Args:
        method: Reconstruction method ("pnp", "red", "posterior")
        num_iterations: Number of iterations
        lambda_reg: Regularization weight
        lambda_dc: Data consistency weight
        num_inference_steps: Number of diffusion inference steps
        guidance_scale: Guidance scale for conditional generation
        eta: Noise scale for posterior sampling

    Returns:
        Configuration dictionary

    """
    return {
        "method": method,
        "num_iterations": num_iterations,
        "lambda_reg": lambda_reg,
        "lambda_dc": lambda_dc,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "eta": eta,
    }
