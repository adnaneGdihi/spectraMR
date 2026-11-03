"""Reconstruction Inference Strategy

Inference strategy for standard reconstruction models (autoencoders, U-Net, etc.).
"""

import torch

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.infrastructure.inference.predict_contrast import CONDITIONING_KEYS

from .base_inference_strategy import BaseInferenceStrategy


class ReconstructionInferenceStrategy(BaseInferenceStrategy):
    """Inference strategy for reconstruction models.

    Handles standard forward-pass inference for models trained with
    reconstruction losses (L1, L2, perceptual losses, etc.).
    """

    # Phase 2: single-pass forward — fully compatible with tiling.
    supports_grid_tiling: bool = True
    #: ``run_inference`` hands ``contrast_idx`` / ``acquisition_params`` to the forward (#2410).
    forwards_contrast_conditioning: bool = True

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.reconstruction

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input tensor for reconstruction inference.

        For reconstruction models, preprocessing typically involves
        normalization and ensuring correct tensor dimensions.

        Args:
            input_tensor: Raw input tensor (B, C, H, W)
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed tensor ready for model inference
        """
        # Ensure tensor is on correct device
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        # Ensure tensor is in evaluation mode (no gradients)
        input_tensor = input_tensor.detach()

        # Apply any model-specific preprocessing
        if hasattr(self.model, "preprocess_input"):
            input_tensor = self.model.preprocess_input(input_tensor)

        return input_tensor

    def run_inference(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run reconstruction inference.

        Args:
            input_tensor: Preprocessed input tensor
            **kwargs: Additional inference parameters

        Returns:
            Model output tensor
        """
        with torch.no_grad():
            conditioning = {k: kwargs[k] for k in CONDITIONING_KEYS if k in kwargs}
            output = self.model(input_tensor, **conditioning)

            # Handle different model output formats
            if isinstance(output, dict):
                # Some models return dict with multiple outputs
                if "reconstruction" in output:
                    output = output["reconstruction"]
                elif "output" in output:
                    output = output["output"]
                else:
                    # Take first tensor value
                    output = next(iter(output.values()))

        return output

    def postprocess_output(self, output_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Postprocess reconstruction model output.

        Args:
            output_tensor: Raw model output
            **kwargs: Additional postprocessing parameters

        Returns:
            Postprocessed output tensor
        """
        # Ensure output is detached from computation graph
        output_tensor = output_tensor.detach().cpu()

        # Apply any model-specific postprocessing
        if hasattr(self.model, "postprocess_output"):
            output_tensor = self.model.postprocess_output(output_tensor)

        # Clamp outputs to valid range if needed
        if self.config.get("clamp_output"):
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        return output_tensor

    @torch.no_grad()
    def infer_single(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
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
        self, input_data_list: list[torch.Tensor], batch_size: int = 4, **kwargs
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
