"""Model Compression Utilities

Provides comprehensive model compression techniques for MRI super-resolution models,
including quantization, pruning, knowledge distillation, and efficient architectures.

Author: spectraMR Research Team
"""

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for model compression."""

    # Quantization settings
    quantize_weights: bool = True
    quantize_activations: bool = False
    weight_bits: int = 8
    activation_bits: int = 8

    # Pruning settings
    pruning_method: str = "l1"  # "l1", "l2", "random", "structured"
    pruning_ratio: float = 0.3
    pruning_schedule: str = "linear"  # "linear", "exponential", "cosine"

    # Knowledge distillation
    use_distillation: bool = False
    teacher_model: nn.Module | None = None
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.5

    # Architecture optimization
    use_depthwise_separable: bool = False
    channel_pruning_ratio: float = 0.0

    # General settings
    target_compression_ratio: float = 2.0
    preserve_accuracy: bool = True
    calibration_samples: int = 100


@dataclass
class CompressionStats:
    """Statistics for compression performance."""

    original_size_mb: float = 0
    compressed_size_mb: float = 0
    compression_ratio: float = 1.0
    accuracy_drop: float = 0
    inference_speedup: float = 1.0
    memory_reduction: float = 1.0
    quantization_stats: dict[str, Any] = field(default_factory=dict)
    pruning_stats: dict[str, Any] = field(default_factory=dict)


class QuantizationManager:
    """Manages quantization of neural network models for reduced precision
    inference.
    """

    def __init__(self, config: CompressionConfig):
        """__init__.

        Args:
            config (CompressionConfig): Description.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor | None = None,
    ) -> nn.Module:
        """Quantize model weights and optionally activations.

        Args:
            model: Model to quantize
            calibration_data: Data for calibration (optional)

        Returns:
            Quantized model

        """
        self.logger.info("Starting model quantization...")

        quantized_model = copy.deepcopy(model)

        if self.config.quantize_weights:
            quantized_model = self._quantize_weights(quantized_model)

        if self.config.quantize_activations and calibration_data is not None:
            quantized_model = self._quantize_activations(
                quantized_model,
                calibration_data,
            )

        self.logger.info("Model quantization completed.")
        return quantized_model

    def _quantize_weights(self, model: nn.Module) -> nn.Module:
        """Quantize model weights to specified bit precision."""
        bits = self.config.weight_bits

        def quantize_tensor(tensor: torch.Tensor) -> torch.Tensor:
            """quantize_tensor.

            Args:
                tensor (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            if bits == 8:
                # 8-bit quantization
                scale = tensor.abs().max() / 127
                quantized = torch.round(tensor / scale).clamp(-127, 127)
                return quantized * scale
            if bits == 4:
                # 4-bit quantization
                scale = tensor.abs().max() / 7
                quantized = torch.round(tensor / scale).clamp(-7, 7)
                return quantized * scale
            self.logger.warning(f"Unsupported quantization bits: {bits}")
            return tensor

        for _name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                with torch.no_grad():
                    module.weight.data = quantize_tensor(module.weight.data)
                    if module.bias is not None:
                        module.bias.data = quantize_tensor(module.bias.data)

        return model

    def _quantize_activations(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
    ) -> nn.Module:
        """Quantize activations using calibration data.

        .. warning:: **LIMITATION**: Activation quantization is not yet
           implemented.  Full activation quantization requires either:

           1. **PyTorch QAT** (``torch.quantization.prepare_qat`` →
              ``convert``) for training-time quantization, or
           2. **Post-training quantization** via TensorRT / ONNX Runtime.

           This method currently returns the model unmodified.
        """
        self.logger.warning(
            "Activation quantization is a no-op. Use PyTorch QAT or "
            "TensorRT post-training quantization for real activation "
            "quantization.  Returning model unchanged."
        )
        return model

    def get_quantization_stats(self, model: nn.Module) -> dict[str, Any]:
        """Get quantization statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        param_bits = self.config.weight_bits

        return {
            "quantization_bits": param_bits,
            "original_bits": 32,
            "compression_ratio": 32 / param_bits,
            "total_parameters": total_params,
        }


class PruningManager:
    """Manages model pruning for reducing model size and computation."""

    def __init__(self, config: CompressionConfig):
        """__init__.

        Args:
            config (CompressionConfig): Description.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pruning_masks = {}

    def prune_model(
        self,
        model: nn.Module,
        training_data: torch.Tensor | None = None,
    ) -> nn.Module:
        """Apply pruning to the model.

        Args:
            model: Model to prune
            training_data: Data for pruning calibration (optional)

        Returns:
            Pruned model

        """
        self.logger.info(f"Starting {self.config.pruning_method} pruning...")

        pruned_model = copy.deepcopy(model)

        if self.config.pruning_method == "l1":
            pruned_model = self._l1_pruning(pruned_model)
        elif self.config.pruning_method == "l2":
            pruned_model = self._l2_pruning(pruned_model)
        elif self.config.pruning_method == "random":
            pruned_model = self._random_pruning(pruned_model)
        elif self.config.pruning_method == "structured":
            pruned_model = self._structured_pruning(pruned_model)

        self.logger.info("Model pruning completed.")
        return pruned_model

    def _l1_pruning(self, model: nn.Module) -> nn.Module:
        """Apply L1 norm-based pruning with global thresholding."""
        # 1. Gather all weights into one list
        all_weights = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                all_weights.append(module.weight.data.view(-1))

        if not all_weights:
            return model

        # 2. Concat and sort ONCE
        global_weights = torch.cat(all_weights)
        # Use abs() for magnitude-based pruning
        threshold = torch.quantile(global_weights.abs(), self.config.pruning_ratio)

        # 3. Apply masks
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                # Create mask
                mask = (module.weight.data.abs() > threshold).float()

                # Apply pruning
                module.weight.data *= mask
                self.pruning_masks[name] = mask

        return model

    def _l2_pruning(self, model: nn.Module) -> nn.Module:
        """Apply L2 norm-based pruning."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                with torch.no_grad():
                    # Calculate L2 norm for each filter/channel
                    if isinstance(module, nn.Linear):
                        norms = torch.norm(module.weight.data, p=2, dim=1)
                    else:
                        norms = torch.norm(module.weight.data, p=2, dim=(1, 2, 3))

                    # Calculate threshold
                    threshold = torch.quantile(norms, self.config.pruning_ratio)

                    # Create mask
                    mask = (norms > threshold).float()

                    # Apply pruning
                    if isinstance(module, nn.Linear):
                        mask = mask.unsqueeze(1).expand_as(module.weight.data)
                    else:
                        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        mask = mask.expand_as(module.weight.data)

                    module.weight.data *= mask
                    self.pruning_masks[name] = mask

        return model

    def _random_pruning(self, model: nn.Module) -> nn.Module:
        """Apply random pruning."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                with torch.no_grad():
                    # Create random mask
                    mask = torch.rand_like(module.weight.data) > self.config.pruning_ratio
                    module.weight.data *= mask.float()
                    self.pruning_masks[name] = mask.float()

        return model

    def _structured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply structured pruning (remove entire channels/filters)."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                with torch.no_grad():
                    # Calculate importance scores for each channel
                    norms = torch.norm(module.weight.data, p=2, dim=(1, 2, 3))

                    # Select channels to keep
                    num_keep = int(norms.size(0) * (1 - self.config.pruning_ratio))
                    _, indices = torch.topk(norms, num_keep)

                    # Create new weight tensor with selected channels
                    new_weight = module.weight.data[indices]
                    module.weight.data = new_weight

                    # Update output channels
                    module.out_channels = num_keep

                    # Store pruning info
                    self.pruning_masks[name] = indices

        return model

    def get_pruning_stats(self, model: nn.Module) -> dict[str, Any]:
        """Get pruning statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        pruned_params = sum(
            (mask == 0).sum().item()
            for mask in self.pruning_masks.values()
            if isinstance(mask, torch.Tensor)
        )

        return {
            "pruning_method": self.config.pruning_method,
            "pruning_ratio": self.config.pruning_ratio,
            "total_parameters": total_params,
            "pruned_parameters": pruned_params,
            "sparsity_ratio": pruned_params / total_params if total_params > 0 else 0,
        }


class KnowledgeDistillationManager:
    """Manages knowledge distillation for model compression."""

    def __init__(self, config: CompressionConfig):
        """__init__.

        Args:
            config (CompressionConfig): Description.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def distillation_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor,
        criterion: Callable,
    ) -> torch.Tensor:
        """Calculate distillation loss combining student-target and student-teacher losses.

        Args:
            student_output: Output from student model
            teacher_output: Output from teacher model
            target: Ground truth target
            criterion: Base loss function

        Returns:
            Combined distillation loss

        """
        if self.config.teacher_model is None:
            return criterion(student_output, target)

        # Hard loss (student vs target)
        hard_loss = criterion(student_output, target)

        # Soft loss (student vs teacher)
        with torch.no_grad():
            teacher_logits = self.config.teacher_model(target)

        # Temperature scaling
        T = self.config.distillation_temperature
        student_logits = student_output / T
        teacher_logits = teacher_logits / T

        # KL divergence loss
        soft_loss = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        ) * (T * T)

        # Combine losses
        alpha = self.config.distillation_alpha
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss

        return total_loss


class ModelCompressionPipeline:
    """Complete model compression pipeline combining multiple techniques."""

    def __init__(self, config: CompressionConfig = None):
        """__init__.

        Args:
            config (CompressionConfig): Description.
        """
        self.config = config or CompressionConfig()
        self.quantization_manager = QuantizationManager(self.config)
        self.pruning_manager = PruningManager(self.config)
        self.distillation_manager = KnowledgeDistillationManager(self.config)
        self.stats = CompressionStats()
        self.logger = logging.getLogger(__name__)

    def compress_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor | None = None,
        training_data: torch.Tensor | None = None,
    ) -> nn.Module:
        """Apply complete compression pipeline to model.

        Args:
            model: Model to compress
            calibration_data: Data for quantization calibration
            training_data: Data for pruning calibration

        Returns:
            Compressed model

        """
        self.logger.info("Starting model compression pipeline...")

        # Store original model info
        original_params = sum(p.numel() for p in model.parameters())
        original_size = original_params * 4 / (1024 * 1024)  # MB (float32)
        self.stats.original_size_mb = original_size

        compressed_model = copy.deepcopy(model)

        # Apply pruning
        if self.config.pruning_ratio > 0:
            compressed_model = self.pruning_manager.prune_model(
                compressed_model,
                training_data,
            )

        # Apply quantization
        if self.config.quantize_weights:
            compressed_model = self.quantization_manager.quantize_model(
                compressed_model,
                calibration_data,
            )

        # Calculate final statistics
        final_params = sum(p.numel() for p in compressed_model.parameters())
        final_size = final_params * self.config.weight_bits / 8 / (1024 * 1024)
        self.stats.compressed_size_mb = final_size
        self.stats.compression_ratio = original_size / final_size if final_size > 0 else 1

        # Collect detailed stats
        self.stats.quantization_stats = self.quantization_manager.get_quantization_stats(
            compressed_model
        )
        self.stats.pruning_stats = self.pruning_manager.get_pruning_stats(
            compressed_model,
        )

        self.logger.info(
            f"Compression completed. Ratio: {self.stats.compression_ratio:.2f}x, "
            f"Size: {original_size:.1f}MB -> {final_size:.1f}MB",
        )

        return compressed_model

    def get_compression_stats(self) -> CompressionStats:
        """Get compression statistics."""
        return self.stats

    def distillation_loss(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target: torch.Tensor,
        criterion: Callable,
    ) -> torch.Tensor:
        """Calculate distillation loss."""
        return self.distillation_manager.distillation_loss(
            student_output,
            teacher_output,
            target,
            criterion,
        )


# Convenience functions
def compress_model_for_inference(
    model: nn.Module,
    target_compression_ratio: float = 2.0,
    quantize_bits: int = 8,
    pruning_ratio: float = 0.2,
) -> nn.Module:
    """Convenience function for inference-focused model compression.

    Args:
        model: Model to compress
        target_compression_ratio: Desired compression ratio
        quantize_bits: Quantization precision
        pruning_ratio: Pruning ratio

    Returns:
        Compressed model

    """
    config = CompressionConfig(
        target_compression_ratio=target_compression_ratio,
        weight_bits=quantize_bits,
        pruning_ratio=pruning_ratio,
        preserve_accuracy=True,
    )

    pipeline = ModelCompressionPipeline(config)
    return pipeline.compress_model(model)


class EfficientBlock(nn.Module):
    """EfficientBlock class."""

    def __init__(self, in_ch, out_ch, use_depthwise=False):
        """__init__.

        Args:
            in_ch (Any): Description.
            out_ch (Any): Description.
            use_depthwise (Any): Description.
        """
        super().__init__()
        if use_depthwise:
            # Depthwise separable convolution
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for EfficientBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.conv(x)


# Renamed from ``efficient_unet`` to break the collision with
# ``generators/efficient_unet.py:EfficientUNetGenerator``. The two
# classes are unrelated (this is a compression-pipeline lightweight
# UNet; the generator is a memory-efficient inverted-residual UNet).
# See TODO/audit/09_models_registry_generators.md §3.2.
@register_model(name="compression_efficient_unet", training_mode="compression")
class EfficientUNet(nn.Module):
    """EfficientUNet class."""

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 64,
        use_depthwise: bool = False,
    ):
        """__init__.

        Args:
            input_channels (int): Description.
            output_channels (int): Description.
            base_channels (int): Description.
            use_depthwise (bool): Description.
        """
        super().__init__()
        self.encoder1 = EfficientBlock(input_channels, base_channels, use_depthwise)
        self.encoder2 = EfficientBlock(
            base_channels,
            base_channels * 2,
            use_depthwise,
        )
        self.decoder1 = EfficientBlock(
            base_channels * 2,
            base_channels,
            use_depthwise,
        )
        self.final = nn.Conv2d(base_channels, output_channels, 1)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for EfficientUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        d1 = self.decoder1(
            F.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=False),
        )
        return self.final(d1)


def create_efficient_architecture(
    input_channels: int = 1,
    output_channels: int = 1,
    base_channels: int = 64,
    use_depthwise: bool = False,
) -> nn.Module:
    """Create an efficient architecture with optional depthwise separable convolutions.

    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels
        base_channels: Base number of channels
        use_depthwise: Whether to use depthwise separable convolutions

    Returns:
        Efficient model architecture

    """
    return EfficientUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=base_channels,
        use_depthwise=use_depthwise,
    )


# Test/example usage
if __name__ == "__main__":
    # Create a simple test model
    model = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 1, 3, padding=1),
    )

    # Test compression pipeline
    config = CompressionConfig(quantize_weights=True, weight_bits=8, pruning_ratio=0.1)

    pipeline = ModelCompressionPipeline(config)

    # Compress model
    compressed = pipeline.compress_model(model)

    # Get stats
    stats = pipeline.get_compression_stats()

    logger.debug(f"Original size: {stats.original_size_mb:.2f}MB")
    logger.debug(f"Compressed size: {stats.compressed_size_mb:.2f}MB")
    logger.debug(f"Compression ratio: {stats.compression_ratio:.2f}x")
    logger.debug("Model compression test completed!")
