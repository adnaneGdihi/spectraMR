#!/usr/bin/env python3
"""DINOv2-based Perceptual Loss
=============================

Perceptual loss using DINOv2 features for medical image quality assessment.
DINOv2 provides superior feature representations compared to traditional
VGG-based approaches, especially for medical imaging tasks.
"""

import logging
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.losses.registry import register_loss

from .base_loss import BaseLoss

logger = logging.getLogger(__name__)


@register_loss(name="dino_perceptual", aliases=["DINOv2PerceptualLoss"], domain="image")
class DINOv2PerceptualLoss(BaseLoss):
    """DINOv2-based perceptual loss for medical images.

    **DOMAIN**: IMAGE-SPACE ONLY
    **Input**: [B, 1, H, W] or [B, 3, H, W] (auto-converts to RGB)
    **3D Support**: Yes (Vision Transformer adapter available)

    Pre-trained ViT features for perceptual similarity. Do NOT use with k-space [B, 2, H, W].
    Auto-converts grayscale to RGB for ViT input projection.

    Based on: DINOv2 (Oquab et al., 2023)
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        layers: list[int] | None = None,
        use_pretrained: bool = True,
        normalize_input: bool = True,
        spatial: bool = False,
        reduction: str = "mean",
    ):
        """Initialize DINOv2 perceptual loss.

        Args:
            model_name: DINOv2 model variant ('vit_base_patch14_dinov2',
                'vit_small_patch14_dinov2', etc.)
            layers: Transformer layers to extract features from.
                Defaults to [4, 8, 12] for base model.
            use_pretrained: Whether to use pre-trained DINOv2 weights
            normalize_input: Whether to normalize input images to ImageNet stats
            spatial: Whether to return spatial feature maps instead of pooled
                features
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for DINOv2PerceptualLoss. Install with: pip install timm"
            ) from exc

        super().__init__(reduction)

        self.model_name = model_name
        self.spatial = spatial
        self.normalize_input = normalize_input

        # Default layers for different model sizes
        if layers is None:
            if "small" in model_name:
                layers = [3, 6, 9]
            elif "base" in model_name:
                layers = [4, 8, 12]
            elif "large" in model_name:
                layers = [6, 12, 18]
            elif "giant" in model_name:
                layers = [8, 16, 24]
            else:
                layers = [4, 8, 12]  # fallback

        self.layers = layers

        # Load DINOv2 model
        dino_model: Any = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=0,  # Remove classification head
        )
        self.dino_model = dino_model

        # Get expected input size
        self.expected_size = int(dino_model.patch_embed.img_size[0])

        # Adapt for single-channel medical images
        self._adapt_for_medical_input()

        # Create resize layer for input adaptation
        self.resize = nn.Upsample(
            size=(self.expected_size, self.expected_size),
            mode="bilinear",
            align_corners=False,
        )

        # Freeze DINOv2 parameters to prevent training
        for param in self.dino_model.parameters():
            param.requires_grad_(False)

        # Freeze all parameters
        for param in self.dino_model.parameters():
            param.requires_grad = False

        # Register hooks to extract features from specified layers
        self.feature_maps: dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []

        def hook_fn(name: str) -> Callable[[nn.Module, Any, torch.Tensor], None]:
            """hook_fn.

            Args:
                name (Any): Description.
            Returns:
                Any: Description.
            """

            def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
                """hook.

                Args:
                    module (Any): Description.
                    input (Any): Description.
                    output (Any): Description.
                Returns:
                    Any: Description.
                """
                self.feature_maps[name] = output

            return hook

        # Register hooks for specified layers
        for layer_idx in self.layers:
            layer_name = f"blocks.{layer_idx}"
            try:
                # Access blocks as model.blocks[int(layer_idx)]
                block = dino_model.blocks[int(layer_idx)]
                hook = block.register_forward_hook(hook_fn(layer_name))
                self.hooks.append(hook)
            except (IndexError, ValueError):
                logger.warning(f"Could not register hook for layer {layer_name}")

        # Normalization parameters (ImageNet stats, but adapted for medical images)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _adapt_for_medical_input(self) -> None:
        """Adapt DINOv2 model for single-channel medical images.

        Replaces the 3-channel input projection with a 1-channel version
        that expands to 3 channels internally.
        """
        # Get original patch embedding layer
        patch_embed: Any = self.dino_model.patch_embed

        # Create new projection layer for 1-channel input
        old_proj = patch_embed.proj
        new_proj = nn.Conv2d(
            1,  # Single channel input
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )

        # Initialize weights by replicating across channels
        with torch.no_grad():
            # Average the original 3-channel weights across input channels
            new_weight = old_proj.weight.mean(dim=1, keepdim=True)
            new_proj.weight.copy_(new_weight)
            if old_proj.bias is not None and new_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        # Replace the projection layer
        patch_embed.proj = new_proj

        logger.info(f"Adapted {self.model_name} for 1-channel medical input")

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input images."""
        if self.normalize_input:
            # For medical images, we might want different normalization
            # For now, use ImageNet stats but adapted
            return (x - self.mean[:, :1]) / self.std[:, :1]
        return x

    def _extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from specified layers."""
        # Resize input to expected size
        x_resized = self.resize(x)

        # Clear previous feature maps
        self.feature_maps.clear()

        # Forward pass through model (gradients flow for differentiability)
        _ = self.dino_model(x_resized)

        return self.feature_maps.copy()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute DINOv2 perceptual loss between two images.

        Args:
            pred: Predicted image tensor [B, 1, H, W] (grayscale medical image)
            target: Target image tensor [B, 1, H, W] (grayscale medical image)

        Returns:
            Perceptual loss value
        """
        self._validate_inputs(pred, target)

        if pred.dim() != 4 or target.dim() != 4:
            raise ValueError("Input images must be 4D tensors [B, C, H, W]")

        if pred.shape[1] != 1 or target.shape[1] != 1:
            raise ValueError("Input images must be single-channel (grayscale)")

        # Normalize inputs
        pred_norm = self._normalize_input(pred)
        target_norm = self._normalize_input(target)

        # Extract features
        features_pred = self._extract_features(pred_norm)
        features_target = self._extract_features(target_norm)

        # Compute loss across layers
        loss_per_image = None
        layer_weights = [1.0 / len(self.layers)] * len(self.layers)

        for layer_name, weight in zip(self.layers, layer_weights, strict=False):
            key = f"blocks.{layer_name}"
            if key not in features_pred or key not in features_target:
                continue

            feat_pred = features_pred[key]  # [B, N, C] where N is sequence length
            feat_target = features_target[key]

            # Global average pooling across sequence dimension
            feat_pred_pooled = feat_pred.mean(dim=1)  # [B, C]
            feat_target_pooled = feat_target.mean(dim=1)  # [B, C]

            layer_loss_per_image = F.mse_loss(
                feat_pred_pooled, feat_target_pooled, reduction="none"
            ).mean(dim=1)  # [B]

            if loss_per_image is None:
                loss_per_image = weight * layer_loss_per_image
            else:
                loss_per_image = loss_per_image + weight * layer_loss_per_image

        if loss_per_image is None:
            raise RuntimeError("No valid layers found for feature extraction")

        return self._apply_reduction(loss_per_image)
