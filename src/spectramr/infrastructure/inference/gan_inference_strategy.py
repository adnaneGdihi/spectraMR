"""GAN Inference Strategy

Inference strategy for Generative Adversarial Networks.
Handles generator inference while managing discriminator components.
"""

from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.restore import weights

from .base_inference_strategy import BaseInferenceStrategy


class GANInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for GAN models.

    Handles generator inference for GANs trained with adversarial losses.
    Manages both generator and discriminator components during inference.
    """

    # Phase 2: GAN generator is a single-pass forward — tile-then-aggregate
    # is numerically equivalent to whole-volume inference (modulo Hann
    # blending at borders). Opt in.
    supports_grid_tiling: bool = True

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize GAN inference strategy.

        Args:
            model: The generator model
            device: Device to run inference on
            config: Configuration dictionary
        """
        super().__init__(model, device, config)
        self.discriminator = None
        self.gan_config = self.config.get("gan", {})

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.GAN

    def load_checkpoint(self, checkpoint_path: str, **kwargs: Any) -> None:
        """Load the generator as the base class does (EMA when kept), and the
        discriminator when the file carries one and ``gan.discriminator`` is configured.

        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional loading parameters
        """
        super().load_checkpoint(checkpoint_path)

        if "discriminator" in codecs.read(checkpoint_path)["models"]:
            discriminator_config = self.gan_config.get("discriminator", {})
            if discriminator_config:
                from spectramr.models.factories.model_factory import get_model_factory

                factory = get_model_factory()
                self.discriminator = factory.create_discriminator(
                    discriminator_config.get("type", "patch_gan"),
                    in_channels=discriminator_config.get("in_channels", 1),
                    **discriminator_config,
                )

                # Cast to nn.Module to access PyTorch methods
                if isinstance(self.discriminator, nn.Module):
                    weights(checkpoint_path, "discriminator", self.discriminator, policy="exact")
                    self.discriminator = self.discriminator.to(self.device)
                    self.discriminator.eval()

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Preprocess input tensor for GAN inference.

        Args:
            input_tensor: Raw input tensor
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed tensor ready for generator inference
        """
        # Ensure tensor is on correct device
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        # Ensure tensor is detached
        input_tensor = input_tensor.detach()

        # Apply any generator-specific preprocessing
        if hasattr(self.model, "preprocess_input"):
            input_tensor = self.model.preprocess_input(input_tensor)

        return input_tensor

    def run_inference(self, input_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run GAN generator inference.

        Args:
            input_tensor: Preprocessed input tensor
            **kwargs: Additional inference parameters

        Returns:
            Generator output tensor
        """
        with torch.no_grad():
            # Generator forward pass
            output = self.model(input_tensor)

            # Handle different model output formats
            if isinstance(output, dict):
                if "fake_image" in output:
                    output = output["fake_image"]
                elif "reconstruction" in output:
                    output = output["reconstruction"]
                elif "output" in output:
                    output = output["output"]
                else:
                    # Take first tensor value
                    output = next(iter(output.values()))

        return output

    def postprocess_output(self, output_tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Postprocess GAN generator output.

        Args:
            output_tensor: Raw generator output
            **kwargs: Additional postprocessing parameters

        Returns:
            Postprocessed output tensor
        """
        # Ensure output is detached
        output_tensor = output_tensor.detach().cpu()

        # Apply tanh activation if needed (common in GANs)
        if self.gan_config.get("final_activation", "").lower() == "tanh":
            output_tensor = torch.tanh(output_tensor)

        # Convert from [-1, 1] to [0, 1] range
        if self.gan_config.get("normalize_output", True):
            output_tensor = (output_tensor + 1.0) / 2.0

        # Clamp outputs to valid range
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        return output_tensor

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about this GAN inference strategy."""
        info = super().get_strategy_info()
        info.update(
            {
                "has_discriminator": self.discriminator is not None,
                "gan_config": self.gan_config,
            }
        )
        return info

    @torch.no_grad()
    def infer_single(self, input_data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Run inference on a single input tensor.

        Args:
            input_data: Single input tensor
            **kwargs: Additional parameters

        Returns:
            Output tensor
        """
        return self.infer(input_data, **kwargs)

    @torch.no_grad()
    def infer_batch(
        self, input_data_list: list[torch.Tensor], batch_size: int = 4, **kwargs: Any
    ) -> list[torch.Tensor]:
        """Run inference on a batch of input tensors.

        Args:
            input_data_list: List of input tensors
            batch_size: Batch size for processing
            **kwargs: Additional parameters

        Returns:
            List of output tensors
        """
        results = []
        for input_data in input_data_list:
            result = self.infer_single(input_data, **kwargs)
            results.append(result)
        return results
