"""Configurable Noise Schedules for Diffusion Models
==================================================

This module provides comprehensive noise scheduling for diffusion models
with support for linear, cosine, and custom schedules.

Features:
- Linear and cosine noise schedules
- Custom schedule implementations
- Model-specific schedule recommendations
- Command-line argument integration
- Compatibility with all diffusion variants

Author: spectraMR Research Project
Date: August 2025
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class NoiseScheduleType(Enum):
    """Available noise schedule types"""

    LINEAR = "linear"
    COSINE = "cosine"
    SIGMOID = "sigmoid"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    CUSTOM = "custom"


@dataclass
class NoiseScheduleConfig:
    """Configuration for noise schedules"""

    schedule_type: str = "cosine"
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Cosine schedule parameters
    cosine_s: float = 0.008

    # Sigmoid schedule parameters
    sigmoid_range: tuple[float, float] = (-3.0, 3.0)

    # Exponential schedule parameters
    exponential_decay: float = 0.99

    # Polynomial schedule parameters
    polynomial_power: float = 2.0

    # Custom schedule parameters
    custom_beta_values: list[float] | None = None


class BaseNoiseSchedule(ABC):
    """Abstract base class for noise schedules"""

    def __init__(self, config: NoiseScheduleConfig):
        """__init__.

        Args:
            config (NoiseScheduleConfig): Description.
        """
        self.config = config
        self.num_timesteps = config.num_timesteps

        # Compute beta schedule
        self.betas = self._compute_betas()

        # Precompute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]],
        )

        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Precompute values for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        # Clamp to avoid division by zero
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20),
        )

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    @abstractmethod
    def _compute_betas(self) -> torch.Tensor:
        """Compute beta values for the schedule"""

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to images at specified timesteps (forward process)

        Args:
            x_start: Original images [batch_size, channels, height, width]
            noise: Noise to add [batch_size, channels, height, width]
            timesteps: Timesteps [batch_size]

        Returns:
            Noisy images

        """
        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod,
            timesteps,
            x_start.shape,
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            x_start.shape,
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior mean and variance for reverse process

        Args:
            x_start: Predicted original images
            x_t: Current noisy images
            timesteps: Current timesteps

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance)

        """
        posterior_mean_coef1_t = self._extract(
            self.posterior_mean_coef1,
            timesteps,
            x_t.shape,
        )
        posterior_mean_coef2_t = self._extract(
            self.posterior_mean_coef2,
            timesteps,
            x_t.shape,
        )

        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t

        posterior_variance_t = self._extract(
            self.posterior_variance,
            timesteps,
            x_t.shape,
        )
        posterior_log_variance_t = self._extract(
            self.posterior_log_variance_clipped,
            timesteps,
            x_t.shape,
        )

        return posterior_mean, posterior_variance_t, posterior_log_variance_t

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Predict original image from current noisy image and predicted noise

        Args:
            x_t: Current noisy images
            timesteps: Current timesteps
            noise: Predicted noise

        Returns:
            Predicted original images

        """
        sqrt_recip_alphas_cumprod_t = self._extract(
            1.0 / self.sqrt_alphas_cumprod,
            timesteps,
            x_t.shape,
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod,
            timesteps,
            x_t.shape,
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Extract values from tensor a at indices t, reshaping for broadcasting

        Args:
            a: Tensor to extract from
            t: Indices
            x_shape: Shape for broadcasting

        Returns:
            Extracted and reshaped tensor

        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def get_schedule_info(self) -> dict[str, Any]:
        """Get information about the schedule"""
        return {
            "schedule_type": self.config.schedule_type,
            "num_timesteps": self.num_timesteps,
            "beta_start": self.config.beta_start,
            "beta_end": self.config.beta_end,
            "beta_min": torch.min(self.betas).item(),
            "beta_max": torch.max(self.betas).item(),
            "alpha_cumprod_min": torch.min(self.alphas_cumprod).item(),
            "alpha_cumprod_max": torch.max(self.alphas_cumprod).item(),
        }

    def plot_schedule(self, save_path: str | None = None):
        """Plot the noise schedule"""
        import matplotlib.pyplot as plt  # deferred: ~0.5 s at import

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        timesteps = np.arange(self.num_timesteps)

        # Plot betas
        ax1.plot(timesteps, self.betas.numpy())
        ax1.set_title("Beta Schedule")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Beta")
        ax1.grid(True)

        # Plot alphas
        ax2.plot(timesteps, self.alphas.numpy())
        ax2.set_title("Alpha Schedule")
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Alpha")
        ax2.grid(True)

        # Plot cumulative alphas
        ax3.plot(timesteps, self.alphas_cumprod.numpy())
        ax3.set_title("Cumulative Alpha Schedule")
        ax3.set_xlabel("Timestep")
        ax3.set_ylabel("Alpha Cumprod")
        ax3.grid(True)

        # Plot noise levels
        ax4.plot(timesteps, self.sqrt_one_minus_alphas_cumprod.numpy())
        ax4.set_title("Noise Level (sqrt(1-alpha_cumprod))")
        ax4.set_xlabel("Timestep")
        ax4.set_ylabel("Noise Level")
        ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.debug(f"Schedule plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


class LinearNoiseSchedule(BaseNoiseSchedule):
    """Linear noise schedule"""

    def _compute_betas(self) -> torch.Tensor:
        """Compute linear beta schedule"""
        return torch.linspace(
            self.config.beta_start,
            self.config.beta_end,
            self.num_timesteps,
            dtype=torch.float32,
        )


class CosineNoiseSchedule(BaseNoiseSchedule):
    """Cosine noise schedule (improved)"""

    def _compute_betas(self) -> torch.Tensor:
        """Compute cosine beta schedule"""

        def alpha_bar(t):
            """alpha_bar.

            Args:
                t (Any): Description.
            Returns:
                Any: Description.
            """
            return (
                math.cos(
                    (t + self.config.cosine_s) / (1 + self.config.cosine_s) * math.pi / 2,
                )
                ** 2
            )

        betas = []
        for i in range(self.num_timesteps):
            t1 = i / self.num_timesteps
            t2 = (i + 1) / self.num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))

        return torch.tensor(betas, dtype=torch.float32)


class SigmoidNoiseSchedule(BaseNoiseSchedule):
    """Sigmoid-based noise schedule"""

    def _compute_betas(self) -> torch.Tensor:
        """Compute sigmoid beta schedule"""

        def sigmoid(x):
            """sigmoid.

            Args:
                x (Any): Description.
            Returns:
                Any: Description.
            """
            return 1 / (1 + np.exp(-x))

        # Create sigmoid curve
        x_min, x_max = self.config.sigmoid_range
        x = np.linspace(x_min, x_max, self.num_timesteps)
        sigmoid_values = sigmoid(x)

        # Normalize to beta range
        betas = (
            self.config.beta_start
            + (self.config.beta_end - self.config.beta_start) * sigmoid_values
        )

        return torch.tensor(betas, dtype=torch.float32)


class ExponentialNoiseSchedule(BaseNoiseSchedule):
    """Exponential noise schedule"""

    def _compute_betas(self) -> torch.Tensor:
        """Compute exponential beta schedule"""
        # decay = self.config.exponential_decay  # Unused
        betas = []

        for i in range(self.num_timesteps):
            # Exponential growth from beta_start to beta_end
            progress = i / (self.num_timesteps - 1)
            beta = (
                self.config.beta_start * (self.config.beta_end / self.config.beta_start) ** progress
            )
            betas.append(beta)

        return torch.tensor(betas, dtype=torch.float32)


class PolynomialNoiseSchedule(BaseNoiseSchedule):
    """Polynomial noise schedule"""

    def _compute_betas(self) -> torch.Tensor:
        """Compute polynomial beta schedule"""
        power = self.config.polynomial_power
        progress = torch.linspace(0, 1, self.num_timesteps, dtype=torch.float32)

        # Apply polynomial transformation
        poly_progress = progress**power

        # Map to beta range
        betas = (
            self.config.beta_start + (self.config.beta_end - self.config.beta_start) * poly_progress
        )

        return betas


class CustomNoiseSchedule(BaseNoiseSchedule):
    """Custom noise schedule with user-defined values"""

    def _compute_betas(self) -> torch.Tensor:
        """Use custom beta values"""
        if self.config.custom_beta_values is None:
            logger.debug("Warning: No custom beta values provided, falling back to linear schedule")
            return torch.linspace(
                self.config.beta_start,
                self.config.beta_end,
                self.num_timesteps,
                dtype=torch.float32,
            )

        custom_betas = torch.tensor(self.config.custom_beta_values, dtype=torch.float32)

        # Interpolate if needed
        if len(custom_betas) != self.num_timesteps:
            logger.debug(
                f"Interpolating custom betas from {len(custom_betas)} to {self.num_timesteps} timesteps"
            )
            import torch.nn.functional as F

            custom_betas = F.interpolate(
                custom_betas.unsqueeze(0).unsqueeze(0),
                size=self.num_timesteps,
                mode="linear",
                align_corners=True,
            ).squeeze()

        return custom_betas


class NoiseScheduleFactory:
    """Factory for creating noise schedules"""

    @staticmethod
    def get_model_specific_schedule_recommendations() -> dict[str, dict[str, Any]]:
        """Get recommended schedule configurations for different model types"""
        return {
            # Standard diffusion models
            "diffusion": {
                "schedule_type": "cosine",
                "num_timesteps": 1000,
                "beta_start": 1e-4,
                "beta_end": 0.02,
                "cosine_s": 0.008,
            },
            "cold_diffusion": {
                "schedule_type": "linear",
                "num_timesteps": 500,
                "beta_start": 1e-4,
                "beta_end": 0.015,
            },
            "score_based_diffusion": {
                "schedule_type": "exponential",
                "num_timesteps": 2000,
                "beta_start": 1e-5,
                "beta_end": 0.01,
                "exponential_decay": 0.995,
            },
            # Enhanced diffusion models
            "chi_square_diffusion": {
                "schedule_type": "polynomial",
                "num_timesteps": 800,
                "beta_start": 5e-5,
                "beta_end": 0.025,
                "polynomial_power": 1.5,
            },
            "stable_diffusion": {
                "schedule_type": "cosine",
                "num_timesteps": 1000,
                "beta_start": 1e-4,
                "beta_end": 0.012,
                "cosine_s": 0.010,
            },
            "stable_small_diffusion": {
                "schedule_type": "cosine",
                "num_timesteps": 500,
                "beta_start": 2e-4,
                "beta_end": 0.015,
                "cosine_s": 0.006,
            },
            # Transformer diffusion models
            "vit_diffusion": {
                "schedule_type": "sigmoid",
                "num_timesteps": 1000,
                "beta_start": 1e-4,
                "beta_end": 0.018,
                "sigmoid_range": (-2.5, 2.5),
            },
            "swin_diffusion": {
                "schedule_type": "cosine",
                "num_timesteps": 1200,
                "beta_start": 8e-5,
                "beta_end": 0.016,
                "cosine_s": 0.012,
            },
            # KAN diffusion models
            "diffusion_kan": {
                "schedule_type": "polynomial",
                "num_timesteps": 600,
                "beta_start": 2e-4,
                "beta_end": 0.02,
                "polynomial_power": 2.2,
            },
            "cold_diffusion_kan": {
                "schedule_type": "linear",
                "num_timesteps": 400,
                "beta_start": 1e-4,
                "beta_end": 0.018,
            },
            "small_kan_diffusion": {
                "schedule_type": "cosine",
                "num_timesteps": 300,
                "beta_start": 3e-4,
                "beta_end": 0.025,
                "cosine_s": 0.005,
            },
        }

    @staticmethod
    def create_noise_schedule(
        schedule_type: str = "cosine",
        num_timesteps: int = 1000,
        model_type: str | None = None,
        custom_config: dict[str, Any] | None = None,
    ) -> BaseNoiseSchedule:
        """Create a noise schedule

        Args:
            schedule_type: Type of schedule to create
            num_timesteps: Number of diffusion timesteps
            model_type: Model type for automatic recommendations
            custom_config: Custom configuration overrides

        Returns:
            Configured noise schedule

        """
        # Start with default config
        config_dict = {
            "schedule_type": schedule_type,
            "num_timesteps": num_timesteps,
            "beta_start": 1e-4,
            "beta_end": 0.02,
            "cosine_s": 0.008,
            "sigmoid_range": (-3.0, 3.0),
            "exponential_decay": 0.99,
            "polynomial_power": 2.0,
        }

        # Apply model-specific recommendations
        if model_type:
            recommendations = NoiseScheduleFactory.get_model_specific_schedule_recommendations()
            if model_type in recommendations:
                config_dict.update(recommendations[model_type])
                # Update schedule type from recommendations
                schedule_type = config_dict["schedule_type"]

        # Apply custom overrides
        if custom_config:
            config_dict.update(custom_config)

        # Create config object
        config = NoiseScheduleConfig(**config_dict)

        # Create appropriate schedule class
        schedule_classes = {
            "linear": LinearNoiseSchedule,
            "cosine": CosineNoiseSchedule,
            "sigmoid": SigmoidNoiseSchedule,
            "exponential": ExponentialNoiseSchedule,
            "polynomial": PolynomialNoiseSchedule,
            "custom": CustomNoiseSchedule,
        }

        if schedule_type not in schedule_classes:
            raise ValueError(
                f"Unknown schedule_type {schedule_type!r}; available: {sorted(schedule_classes)}"
            )

        return schedule_classes[schedule_type](config)

    @staticmethod
    def get_available_schedule_types() -> list[str]:
        """Get list of available schedule types"""
        return [schedule_type.value for schedule_type in NoiseScheduleType]


class NoiseScheduleManager:
    """Manager for handling multiple noise schedules and transitions"""

    def __init__(self, schedules: dict[str, BaseNoiseSchedule]):
        """__init__.

        Args:
            schedules (dict[str, BaseNoiseSchedule]): Description.
        """
        self.schedules = schedules
        self.current_schedule_name = list(schedules.keys())[0] if schedules else None
        self.current_schedule = (
            schedules[self.current_schedule_name] if self.current_schedule_name else None
        )

    def switch_schedule(self, schedule_name: str):
        """Switch to a different schedule"""
        if schedule_name in self.schedules:
            self.current_schedule_name = schedule_name
            self.current_schedule = self.schedules[schedule_name]
            logger.debug(f"Switched to {schedule_name} schedule")
        else:
            raise KeyError(
                f"Unknown schedule {schedule_name!r}; available: {sorted(self.schedules)}"
            )

    def add_schedule(self, name: str, schedule: BaseNoiseSchedule):
        """Add a new schedule"""
        self.schedules[name] = schedule
        logger.debug(f"Added {name} schedule")

    def compare_schedules(self, save_path: str | None = None):
        """Compare all available schedules"""
        import matplotlib.pyplot as plt  # deferred: ~0.5 s at import

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        colors = ["blue", "red", "green", "orange", "purple", "brown"]

        for i, (name, schedule) in enumerate(self.schedules.items()):
            color = colors[i % len(colors)]
            timesteps = np.arange(schedule.num_timesteps)

            # Plot betas
            axes[0].plot(timesteps, schedule.betas.numpy(), label=name, color=color)

            # Plot alphas_cumprod
            axes[1].plot(
                timesteps,
                schedule.alphas_cumprod.numpy(),
                label=name,
                color=color,
            )

            # Plot noise levels
            axes[2].plot(
                timesteps,
                schedule.sqrt_one_minus_alphas_cumprod.numpy(),
                label=name,
                color=color,
            )

            # Plot posterior variance
            axes[3].plot(
                timesteps,
                schedule.posterior_variance.numpy(),
                label=name,
                color=color,
            )

        axes[0].set_title("Beta Schedules")
        axes[0].set_xlabel("Timestep")
        axes[0].set_ylabel("Beta")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].set_title("Cumulative Alpha Schedules")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Alpha Cumprod")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].set_title("Noise Level Schedules")
        axes[2].set_xlabel("Timestep")
        axes[2].set_ylabel("Noise Level")
        axes[2].legend()
        axes[2].grid(True)

        axes[3].set_title("Posterior Variance Schedules")
        axes[3].set_xlabel("Timestep")
        axes[3].set_ylabel("Posterior Variance")
        axes[3].legend()
        axes[3].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.debug(f"Schedule comparison saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_current_schedule(self) -> BaseNoiseSchedule | None:
        """Get the currently active schedule"""
        return self.current_schedule

    def get_schedule_info(self) -> dict[str, Any]:
        """Get information about all schedules"""
        info = {
            "current_schedule": self.current_schedule_name,
            "available_schedules": list(self.schedules.keys()),
            "schedules_info": {},
        }

        for name, schedule in self.schedules.items():
            info["schedules_info"][name] = schedule.get_schedule_info()

        return info


def create_diffusion_noise_system(
    noise_schedule: str = "cosine",
    num_timesteps: int = 1000,
    model_type: str | None = None,
    create_multiple: bool = False,
    **kwargs,
) -> dict[str, Any]:
    """Create a comprehensive diffusion noise system

    Args:
        noise_schedule: Primary noise schedule type
        num_timesteps: Number of diffusion timesteps
        model_type: Model type for recommendations
        create_multiple: Create multiple schedules for comparison
        **kwargs: Additional configuration

    Returns:
        Dictionary containing noise schedule system

    """
    # Create primary schedule
    primary_schedule = NoiseScheduleFactory.create_noise_schedule(
        schedule_type=noise_schedule,
        num_timesteps=num_timesteps,
        model_type=model_type,
        custom_config=kwargs,
    )

    schedules = {noise_schedule: primary_schedule}

    # Create additional schedules if requested
    if create_multiple:
        all_types = NoiseScheduleFactory.get_available_schedule_types()

        for schedule_type in all_types:
            if schedule_type != noise_schedule:
                try:
                    additional_schedule = NoiseScheduleFactory.create_noise_schedule(
                        schedule_type=schedule_type,
                        num_timesteps=num_timesteps,
                        model_type=model_type,
                        custom_config=kwargs,
                    )
                    schedules[schedule_type] = additional_schedule
                except Exception as e:
                    logger.debug(f"Failed to create {schedule_type} schedule: {e}")

    # Create manager
    manager = NoiseScheduleManager(schedules)

    return {
        "manager": manager,
        "primary_schedule": primary_schedule,
        "noise_schedule_type": noise_schedule,
        "num_timesteps": num_timesteps,
        "available_types": NoiseScheduleFactory.get_available_schedule_types(),
        "schedule_info": manager.get_schedule_info(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diffusion Noise Schedule System")
    parser.add_argument(
        "--noise_schedule",
        default="cosine",
        choices=NoiseScheduleFactory.get_available_schedule_types(),
        help="Noise schedule type",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
        help="Number of timesteps",
    )
    parser.add_argument("--model_type", default="diffusion", help="Model type")
    parser.add_argument("--plot", action="store_true", help="Plot the schedule")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple schedules",
    )

    args = parser.parse_args()

    # Create noise system
    noise_system = create_diffusion_noise_system(
        noise_schedule=args.noise_schedule,
        num_timesteps=args.num_timesteps,
        model_type=args.model_type,
        create_multiple=args.compare,
    )

    logger.debug(f"Created {args.noise_schedule} noise schedule for {args.model_type}")
    logger.debug(f"Schedule info: {noise_system['schedule_info']}")

    # Plot if requested
    if args.plot:
        primary_schedule = noise_system["primary_schedule"]
        primary_schedule.plot_schedule(f"{args.noise_schedule}_schedule.png")

    # Compare schedules if requested
    if args.compare:
        manager = noise_system["manager"]
        manager.compare_schedules("schedule_comparison.png")

    # Test the schedule
    schedule = noise_system["primary_schedule"]

    # Create dummy data
    batch_size = 4
    x_start = torch.randn(batch_size, 1, 64, 64)
    noise = torch.randn_like(x_start)
    timesteps = torch.randint(
        low=0,
        high=args.num_timesteps,
        size=(batch_size,),
        dtype=torch.long,
    )

    # Test forward process
    x_t = schedule.add_noise(x_start, noise, timesteps)
    logger.debug(f"Added noise at timesteps {timesteps.tolist()}")
    logger.debug(f"Input shape: {x_start.shape}, Output shape: {x_t.shape}")

    # Test reverse process
    x_start_pred = schedule.predict_start_from_noise(x_t, timesteps, noise)
    logger.debug(f"Predicted start shape: {x_start_pred.shape}")

    logger.debug("[OK] Noise schedule system working correctly!")
