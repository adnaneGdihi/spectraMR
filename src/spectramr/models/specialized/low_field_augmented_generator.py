"""Low-Field Style Augmented Generator
===================================

Generator with InstanceNorm statistics shift for low-field MRI adaptation.
Adapts high-field trained models to low-field imaging conditions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from spectramr.models.interfaces import IGenerator
from spectramr.models.registry import register_model


class IStyleAugmentation(nn.Module, ABC):
    """Interface for style augmentation modules."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply style augmentation to input tensor.

        forward method for IStyleAugmentation.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""

    @abstractmethod
    def get_augmentation_params(self) -> dict[str, Any]:
        """Get current augmentation parameters."""

    @abstractmethod
    def set_augmentation_params(self, params: dict[str, Any]) -> None:
        """Set augmentation parameters."""


class InstanceNormStatsShift(IStyleAugmentation):
    """Instance normalization statistics shift for domain adaptation.

    Shifts the mean and variance of InstanceNorm layers to adapt
    high-field trained models to low-field imaging conditions.
    """

    def __init__(
        self,
        num_features: int,
        shift_scale: float = 1.0,
        learnable: bool = True,
        eps: float = 1e-5,
    ):
        """Initialize InstanceNorm statistics shift.

        Args:
            num_features: Number of feature channels
            shift_scale: Scaling factor for shift magnitude
            learnable: Whether shift parameters are learnable
            eps: Small value for numerical stability

        """
        super().__init__()
        self.num_features = num_features
        self.shift_scale = shift_scale
        self.eps = eps

        # Learnable shift parameters for mean and variance
        if learnable:
            self.mean_shift = nn.Parameter(
                torch.zeros(num_features),
                requires_grad=True,
            )
            self.var_shift = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        else:
            self.register_buffer("mean_shift", torch.zeros(num_features))
            self.register_buffer("var_shift", torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply InstanceNorm statistics shift.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor with shifted statistics

        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        # Compute instance statistics
        mean = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)  # (B, C, 1, 1)

        # Apply shift to statistics
        shifted_mean = mean + self.shift_scale * self.mean_shift.view(1, -1, 1, 1)
        shifted_var = var * (1 + self.shift_scale * self.var_shift.view(1, -1, 1, 1))

        # Normalize with shifted statistics
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Denormalize with shifted statistics
        x_augmented = x_normalized * torch.sqrt(shifted_var + self.eps) + shifted_mean

        return x_augmented

    def get_augmentation_params(self) -> dict[str, Any]:
        """Get current augmentation parameters."""
        return {
            "mean_shift": self.mean_shift.detach().cpu(),
            "var_shift": self.var_shift.detach().cpu(),
            "shift_scale": self.shift_scale,
        }

    def set_augmentation_params(self, params: dict[str, Any]) -> None:
        """Set augmentation parameters."""
        if "mean_shift" in params:
            self.mean_shift.data.copy_(params["mean_shift"])
        if "var_shift" in params:
            self.var_shift.data.copy_(params["var_shift"])
        if "shift_scale" in params:
            self.shift_scale = params["shift_scale"]


class AdaptiveInstanceNormShift(IStyleAugmentation):
    """Adaptive InstanceNorm shift with conditional modulation.

    Uses conditional inputs to modulate the shift parameters,
    enabling dynamic adaptation based on input characteristics.
    """

    def __init__(
        self,
        num_features: int,
        condition_dim: int = 64,
        shift_scale: float = 1.0,
        eps: float = 1e-5,
    ):
        """Initialize adaptive InstanceNorm shift.

        Args:
            num_features: Number of feature channels
            condition_dim: Dimension of conditional input
            shift_scale: Scaling factor for shift magnitude
            eps: Small value for numerical stability

        """
        super().__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        self.shift_scale = shift_scale
        self.eps = eps

        # Modulation networks for mean and variance shifts
        self.mean_modulator = nn.Sequential(
            nn.Linear(condition_dim, num_features),
            nn.LayerNorm(num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )

        self.var_modulator = nn.Sequential(
            nn.Linear(condition_dim, num_features),
            nn.LayerNorm(num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply adaptive InstanceNorm statistics shift.

        Args:
            x: Input tensor of shape (B, C, H, W)
            condition: Conditional input of shape (B, condition_dim)

        Returns:
            Output tensor with adaptively shifted statistics

        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input, got {x.dim()}D")

        if condition is None:
            # Fall back to zero condition
            condition = torch.zeros(x.size(0), self.condition_dim, device=x.device)

        # Generate shift parameters from condition
        mean_shift = self.mean_modulator(condition)  # (B, num_features)
        var_shift = self.var_modulator(condition)  # (B, num_features)

        # Compute instance statistics
        mean = x.mean(dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)  # (B, C, 1, 1)

        # Apply adaptive shift to statistics
        shifted_mean = mean + self.shift_scale * mean_shift.view(
            -1,
            self.num_features,
            1,
            1,
        )
        shifted_var = var * (1 + self.shift_scale * var_shift.view(-1, self.num_features, 1, 1))

        # Normalize with original statistics
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Denormalize with shifted statistics
        x_augmented = x_normalized * torch.sqrt(shifted_var + self.eps) + shifted_mean

        return x_augmented

    def get_augmentation_params(self) -> dict[str, Any]:
        """Get current augmentation parameters."""
        return {"shift_scale": self.shift_scale, "condition_dim": self.condition_dim}

    def set_augmentation_params(self, params: dict[str, Any]) -> None:
        """Set augmentation parameters."""
        if "shift_scale" in params:
            self.shift_scale = params["shift_scale"]


class LowFieldConditionEncoder(nn.Module):
    """Encoder for low-field imaging conditions.

    Extracts relevant features from low-field images to condition
    the style augmentation process.
    """

    def __init__(
        self,
        input_channels: int = 1,
        condition_dim: int = 64,
        feature_maps: tuple[int, ...] = (32, 64, 128),
    ):
        """Initialize condition encoder.

        Args:
            input_channels: Number of input channels
            condition_dim: Output condition dimension
            feature_maps: Feature map sizes for encoder

        """
        super().__init__()
        self.condition_dim = condition_dim

        # Encoder layers
        layers = []
        in_channels = input_channels

        for out_channels in feature_maps:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ],
            )
            in_channels = out_channels

        self.encoder = nn.Sequential(*layers)

        # Global pooling and projection to condition space
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.condition_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps[-1], condition_dim),
            nn.LayerNorm(condition_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode low-field condition.

        Args:
            x: Input low-field image of shape (B, C, H, W)

        Returns:
            Condition vector of shape (B, condition_dim)

        """
        features = self.encoder(x)  # (B, feature_maps[-1], H', W')
        pooled = self.global_pool(features)  # (B, feature_maps[-1], 1, 1)
        condition = self.condition_proj(pooled)  # (B, condition_dim)

        return condition


@register_model(name="low_field_augmented", training_mode="reconstruction")
class LowFieldStyleAugmentedGenerator(IGenerator, nn.Module):
    """Generator with low-field style augmentation.

    Augments a base generator with InstanceNorm statistics shift
    to adapt high-field trained models to low-field conditions.
    """

    def __init__(
        self,
        base_generator: IGenerator,
        style_augmentation: IStyleAugmentation,
        condition_encoder: nn.Module | None = None,
        adaptation_layers: nn.ModuleList | None = None,
        **kwargs,
    ):
        """Initialize low-field style augmented generator.

        Args:
            base_generator: Base SR generator
            style_augmentation: Style augmentation module
            condition_encoder: Optional condition encoder for
                adaptive augmentation
            adaptation_layers: Optional additional adaptation layers
            **kwargs: Additional parameters

        """
        super().__init__()
        self.base_generator = base_generator
        self.style_augmentation = style_augmentation
        self.condition_encoder = condition_encoder
        self.adaptation_layers = adaptation_layers or nn.ModuleList()

        # Logger for adaptation monitoring
        self.logger = logging.getLogger(__name__)

    def forward(
        self,
        x: torch.Tensor,
        low_field_reference: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with style augmentation.

        Args:
            x: Input high-field image of shape (B, C, H, W)
            low_field_reference: Optional low-field reference for conditioning
            **kwargs: Additional arguments for base generator

        Returns:
            Super-resolved output adapted to low-field style

        """
        # Extract condition if encoder is available
        condition = None
        if self.condition_encoder is not None and low_field_reference is not None:
            condition = self.condition_encoder(low_field_reference)

        # Apply style augmentation
        if isinstance(self.style_augmentation, AdaptiveInstanceNormShift):
            x_augmented = self.style_augmentation(x, condition)
        else:
            x_augmented = self.style_augmentation(x)

        # Apply adaptation layers if any
        for layer in self.adaptation_layers:
            x_augmented = layer(x_augmented)

        # Apply base generator
        output = self.base_generator(x_augmented, **kwargs)

        return output

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate method for IGenerator interface."""
        return self.forward(x, **kwargs)

    def get_output_shape(self) -> tuple[int, ...]:
        """Get output shape - same as base generator."""
        return self.base_generator.get_output_shape()

    def get_parameter_count(self) -> int:
        """Get total parameter count including augmentation modules."""
        total_params = self.base_generator.get_parameter_count()

        # Add style augmentation parameters
        total_params += sum(p.numel() for p in self.style_augmentation.parameters())

        # Add condition encoder parameters
        if self.condition_encoder is not None:
            total_params += sum(p.numel() for p in self.condition_encoder.parameters())

        # Add adaptation layer parameters
        for layer in self.adaptation_layers:
            total_params += sum(p.numel() for p in layer.parameters())

        return total_params

    def get_adaptation_stats(self) -> dict[str, Any]:
        """Get statistics about the adaptation modules."""
        stats = {
            "style_augmentation_params": self.style_augmentation.get_augmentation_params(),
            "has_condition_encoder": self.condition_encoder is not None,
            "num_adaptation_layers": len(self.adaptation_layers),
            "total_adaptation_params": sum(
                p.numel() for layer in self.adaptation_layers for p in layer.parameters()
            ),
        }

        if self.condition_encoder is not None:
            stats["condition_dim"] = self.condition_encoder.condition_dim

        return stats


class LowFieldAugmentationBundle:
    """Bundle for creating low-field style augmented generators.

    Provides factory methods for common low-field adaptation setups.
    """

    @staticmethod
    def create_basic_shift_augmentation(
        base_model_type: str = "standard_unet",
        shift_scale: float = 0.1,
        **kwargs,
    ) -> dict[str, Any]:
        """Create config for basic InstanceNorm shift augmentation.

        Args:
            base_model_type: Type of base SR model
            shift_scale: Scale for statistics shift
            **kwargs: Additional configuration parameters

        Returns:
            Configuration dictionary for model factory

        """
        return {
            "model_type": "low_field_augmented",
            "base_model_type": base_model_type,
            "style_augmentation_type": "instance_norm_shift",
            "shift_scale": shift_scale,
            "learnable_shift": True,
            **kwargs,
        }

    @staticmethod
    def create_adaptive_shift_augmentation(
        base_model_type: str = "enhanced_unet",
        condition_dim: int = 64,
        shift_scale: float = 0.1,
        **kwargs,
    ) -> dict[str, Any]:
        """Create config for adaptive InstanceNorm shift with conditioning.

        Args:
            base_model_type: Type of base SR model
            condition_dim: Dimension of condition vector
            shift_scale: Scale for statistics shift
            **kwargs: Additional configuration parameters

        Returns:
            Configuration dictionary for model factory

        """
        return {
            "model_type": "low_field_augmented",
            "base_model_type": base_model_type,
            "style_augmentation_type": "adaptive_instance_norm_shift",
            "condition_dim": condition_dim,
            "shift_scale": shift_scale,
            "use_condition_encoder": True,
            **kwargs,
        }

    @staticmethod
    def create_full_adaptation_augmentation(
        base_model_type: str = "vit_kan",
        adaptation_layers: int = 2,
        **kwargs,
    ) -> dict[str, Any]:
        """Create config for full adaptation with multiple layers.

        Args:
            base_model_type: Type of base SR model
            adaptation_layers: Number of additional adaptation layers
            **kwargs: Additional configuration parameters

        Returns:
            Configuration dictionary for model factory

        """
        return {
            "model_type": "low_field_augmented",
            "base_model_type": base_model_type,
            "style_augmentation_type": "adaptive_instance_norm_shift",
            "condition_dim": 128,
            "shift_scale": 0.2,
            "use_condition_encoder": True,
            "num_adaptation_layers": adaptation_layers,
            "adaptatioin_channels": 64,
            **kwargs,
        }


# Factory function for creating low-field augmented generators
def create_low_field_augmented_generator(
    base_model_type: str,
    style_augmentation_type: str,
    factory,
    shift_scale: float = 0.1,
    condition_dim: int = 64,
    use_condition_encoder: bool = False,
    num_adaptation_layers: int = 0,
    adaptatioin_channels: int = 64,
    **kwargs,
) -> IGenerator:
    """Create a low-field style augmented generator.

    Args:
        base_model_type: Type of base SR generator
        style_augmentation_type: Type of style augmentation
        factory: Model factory instance
        shift_scale: Scale for statistics shift
        condition_dim: Dimension of condition vector
        use_condition_encoder: Whether to use condition encoder
        num_adaptation_layers: Number of additional adaptation layers
        adaptatioin_channels: Channels for adaptation layers
        **kwargs: Additional parameters for base generator

    Returns:
        Configured low-field augmented generator

    """
    # Create base generator
    base_generator = factory.create_generator(base_model_type, **kwargs)

    # Get number of features from base generator (assume first conv layer)
    num_features = kwargs.get("in_channels", 1) * 64  # Rough estimate

    # Create style augmentation
    if style_augmentation_type == "instance_norm_shift":
        style_augmentation = InstanceNormStatsShift(
            num_features=num_features,
            shift_scale=shift_scale,
            learnable=True,
        )
    elif style_augmentation_type == "adaptive_instance_norm_shift":
        style_augmentation = AdaptiveInstanceNormShift(
            num_features=num_features,
            condition_dim=condition_dim,
            shift_scale=shift_scale,
        )
    else:
        raise ValueError(f"Unknown style augmentation type: {style_augmentation_type}")

    # Create condition encoder if requested
    condition_encoder = None
    if use_condition_encoder:
        condition_encoder = LowFieldConditionEncoder(
            input_channels=kwargs.get("in_channels", 1),
            condition_dim=condition_dim,
        )

    # Create adaptation layers if requested
    adaptation_layers = nn.ModuleList()
    if num_adaptation_layers > 0:
        for _i in range(num_adaptation_layers):
            adaptation_layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features, adaptatioin_channels, 3, padding=1),
                    nn.InstanceNorm2d(adaptatioin_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(adaptatioin_channels, num_features, 3, padding=1),
                    nn.InstanceNorm2d(num_features),
                    nn.ReLU(inplace=True),
                ),
            )

    # Create augmented generator
    augmented_generator = LowFieldStyleAugmentedGenerator(
        base_generator=base_generator,
        style_augmentation=style_augmentation,
        condition_encoder=condition_encoder,
        adaptation_layers=adaptation_layers,
        **kwargs,
    )

    return augmented_generator


# Alias for backward compatibility
LowFieldAugmentedGenerator = LowFieldStyleAugmentedGenerator
