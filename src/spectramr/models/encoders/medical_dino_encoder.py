"""Medical-Adapted DINOv2 Encoder
==============================

DINOv2 Vision Transformer adapted for single-channel medical imaging.
Replaces the standard 3-channel patch embedding with a 1-channel version
for grayscale MRI and medical imaging applications.
"""

import logging

import torch
from torch import nn

from spectramr.models.interfaces.latent_access import LatentAccessMixin
from spectramr.models.lora import LoRAAdapter
from spectramr.models.registry import register_model
from spectramr.shared.utils.metaclass import ModuleABCMeta

logger = logging.getLogger(__name__)


class LoRAWrapper(nn.Module):
    """LoRAWrapper class."""

    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 1.0):
        """__init__.

        Args:
            original_layer (nn.Linear): Description.
            rank (int): Description.
            alpha (float): Description.
        """
        super().__init__()
        self.original_layer = original_layer
        self.lora_adapter = LoRAAdapter(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            r=rank,
            lora_alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for LoRAWrapper.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.original_layer(x) + self.lora_adapter(x)

    @property
    def lora_layer(self):
        """lora_layer.

        Returns:
            Any: Description.
        """
        return self.lora_adapter


# Renamed to ``medical_dino_encoder`` to break the collision with the
# generator-mode wrapper at ``generators/foundation_generators.py``. The
# legacy ``medical_dino`` key continues to point at MedicalDINOv2Encoder
# via ``model_factory.py``'s explicit override (line ~1765); this
# decorator simply gives the encoder its own discoverable name. See
# TODO/audit/09_models_registry_generators.md §3.4.
@register_model(name="medical_dino_encoder", training_mode="encoder")
class MedicalDINOv2Encoder(LatentAccessMixin, nn.Module, metaclass=ModuleABCMeta):
    # Fixed metaclass conflict using ModuleABCMeta
    """DINOv2 Vision Transformer adapted for medical imaging.

    This encoder modifies the standard DINOv2 ViT to accept single-channel
    medical images (grayscale MRI, CT, etc.) by replacing the 3-channel
    patch embedding projection with a 1-channel version.

    The adaptation preserves the pre-trained weights while enabling
    effective feature extraction from medical imaging data.

    Based on: "DINOv2: Learning Robust Visual Features without Supervision"
    (Oquab et al., 2023)
    """

    def __init__(
        self,
        in_channels: int = 1,
        model_name: str = "vit_base_patch14_dinov2",
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 768,
        use_pretrained: bool = True,
        freeze_backbone: bool = True,
        num_register_tokens: int = 4,
    ):
        """Initialize Medical DINOv2 Encoder.

        Args:
            model_name: DINOv2 model variant from timm
            img_size: Expected input image size (square)
            patch_size: Patch size for ViT
            embed_dim: Embedding dimension
            use_pretrained: Whether to load pre-trained weights
            freeze_backbone: Whether to freeze model parameters
            num_register_tokens: Number of register tokens (DINOv2 feature)
        """
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for MedicalDINOv2Encoder. Install with: pip install timm"
            ) from exc

        super().__init__()

        if in_channels != 1:
            raise ValueError(
                f"MedicalDINOv2Encoder only supports single-channel input "
                f"(grayscale medical images), got in_channels={in_channels}"
            )

        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.freeze_backbone = freeze_backbone
        self.num_register_tokens = num_register_tokens

        # Load base DINOv2 model
        self.dino_model = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,  # Remove classification head
            global_pool="",  # No global pooling
        )

        # Get actual embed_dim from loaded model
        self.embed_dim = self.dino_model.embed_dim

        # Adapt for single-channel medical input
        self._adapt_for_medical_input()

        # Create resize layer for input adaptation
        self.resize = nn.Upsample(size=(img_size, img_size), mode="bilinear", align_corners=False)

        # Freeze parameters if requested
        if freeze_backbone:
            for param in self.dino_model.parameters():
                param.requires_grad_(False)

        # Normalization for medical images (adapted ImageNet stats)
        self.register_buffer("mean", torch.tensor([0.485]).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229]).view(1, 1, 1, 1))

        logger.info(
            f"Initialized MedicalDINOv2Encoder with {model_name} adapted for single-channel input"
        )

    def _adapt_for_medical_input(self):
        """Adapt DINOv2 model for single-channel medical images.

        Replaces the 3-channel patch embedding projection with a 1-channel
        version that maintains the learned representations.
        """
        # Get original patch embedding
        patch_embed = self.dino_model.patch_embed
        old_proj = patch_embed.proj

        # Create new 1-channel projection layer
        new_proj = nn.Conv2d(
            1,  # Single channel input (grayscale)
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )

        # Initialize weights by averaging across input channels
        with torch.no_grad():
            # Original weights shape: [out_channels, 3, kernel_h, kernel_w]
            # New weights shape: [out_channels, 1, kernel_h, kernel_w]
            new_weight = old_proj.weight.mean(dim=1, keepdim=True)
            new_proj.weight.copy_(new_weight)

            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        # Replace the projection layer
        patch_embed.proj = new_proj

        # Update input channels in patch embedding
        patch_embed.in_chans = 1

        logger.debug("Adapted patch embedding for 1-channel medical input")

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for DINOv2 preprocessing."""
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through medical-adapted DINOv2.

        Args:
            x: Input tensor [B, 1, H, W] (single-channel medical image)

        Returns:
            Encoded features [B, embed_dim] (CLS token or pooled)

        forward method for MedicalDINOv2Encoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() != 4:
            raise ValueError(f"Input must be 4D tensor [B, C, H, W], got {x.shape}")

        if x.shape[1] != 1:
            raise ValueError(f"Input must be single-channel (grayscale), got {x.shape[1]} channels")

        # Resize to expected input size
        x_resized = self.resize(x)

        # Normalize
        x_norm = self._normalize_input(x_resized)

        # Forward through DINOv2
        features = self.dino_model(x_norm)  # [B, N, embed_dim]

        # Extract CLS token (first token) as primary representation
        cls_token = features[:, 0]  # [B, embed_dim]

        # Store latent state for compatibility
        self._set_latent_state(embedding=cls_token, sequence=features, pooled=cls_token)

        return cls_token

    def extract_features_at_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Extract features from a specific transformer layer.

        Args:
            x: Input tensor [B, 1, H, W]
            layer_idx: Layer index to extract features from

        Returns:
            Features from specified layer [B, N, embed_dim]
        """
        if x.dim() != 4 or x.shape[1] != 1:
            raise ValueError("Input must be [B, 1, H, W]")

        x_resized = self.resize(x)
        x_norm = self._normalize_input(x_resized)

        # Forward pass up to specified layer
        x_patches = self.dino_model.patch_embed(x_norm)
        x_patches = self.dino_model._pos_embed(x_patches)

        # Add register tokens if present
        if hasattr(self.dino_model, "reg_tokens"):
            reg_tokens = self.dino_model.reg_tokens.expand(x_patches.shape[0], -1, -1)
            x_patches = torch.cat([reg_tokens, x_patches], dim=1)

        # Forward through transformer blocks up to layer_idx
        for i, block in enumerate(self.dino_model.blocks):
            x_patches = block(x_patches)
            if i == layer_idx:
                break

        return x_patches

    def get_intermediate_features(
        self, x: torch.Tensor, layers: list[int] | None = None
    ) -> dict[int, torch.Tensor]:
        """Extract features from multiple intermediate layers.

        Args:
            x: Input tensor [B, 1, H, W]
            layers: List of layer indices to extract from

        Returns:
            Dictionary mapping layer indices to features
        """
        if layers is None:
            # Default layers for different model sizes
            if "small" in self.model_name:
                layers = [3, 6, 9]
            elif "base" in self.model_name:
                layers = [4, 8, 12]
            elif "large" in self.model_name:
                layers = [6, 12, 18]
            else:
                layers = [4, 8, 12]

        features = {}
        for layer_idx in layers:
            features[layer_idx] = self.extract_features_at_layer(x, layer_idx)

        return features

    @property
    def name(self) -> str:
        """Get the model name."""
        return "medical_dino"

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_lora_params(self) -> list[torch.nn.Parameter]:
        """Get all LoRA parameters for optimizer."""
        lora_params = []

        def collect_lora_params(module):
            """collect_lora_params.

            Args:
                module (Any): Description.
            Returns:
                Any: Description.
            """
            for child in module.children():
                if isinstance(child, LoRAWrapper):
                    lora_params.extend(child.lora_layer.parameters())
                else:
                    collect_lora_params(child)

        collect_lora_params(self)
        return lora_params

    def enable_lora(
        self,
        rank: int = 8,
        alpha: float = 1.0,
        target_modules: list[str] | None = None,
    ):
        """Enable LoRA fine-tuning for specified modules.

        Args:
            rank: LoRA rank (lower = more parameter efficient)
            alpha: Scaling factor for LoRA updates
            target_modules: List of module names to apply LoRA to.
                          Defaults to attention and MLP layers.
        """
        if target_modules is None:
            target_modules = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]

        self.lora_config = {
            "rank": rank,
            "alpha": alpha,
            "target_modules": target_modules,
        }

        lora_applied = 0

        # Apply LoRA to transformer blocks
        for block_idx, block in enumerate(self.dino_model.blocks):
            for module_path in target_modules:
                # Split module path (e.g., 'attn.qkv' -> ['attn', 'qkv'])
                path_parts = module_path.split(".")
                current_module = block

                # Navigate to the target module
                try:
                    for part in path_parts:
                        current_module = getattr(current_module, part)

                    if isinstance(current_module, nn.Linear):
                        # Wrap with LoRA
                        lora_wrapper = LoRAWrapper(current_module, rank=rank, alpha=alpha)
                        # Set the wrapper back in the correct location
                        parent_module = block
                        for part in path_parts[:-1]:
                            parent_module = getattr(parent_module, part)
                        setattr(parent_module, path_parts[-1], lora_wrapper)
                        lora_applied += 1
                        logger.debug(f"Applied LoRA to block {block_idx}.{module_path}")
                except AttributeError:
                    # Module path doesn't exist, skip
                    continue

        # Apply LoRA to patch embedding if requested
        if "patch_embed" in target_modules and hasattr(self.dino_model.patch_embed, "proj"):
            proj_layer = self.dino_model.patch_embed.proj
            if isinstance(proj_layer, nn.Linear):
                lora_wrapper = LoRAWrapper(proj_layer, rank=rank, alpha=alpha)
                self.dino_model.patch_embed.proj = lora_wrapper
                lora_applied += 1
                logger.debug("Applied LoRA to patch_embed.proj")

        logger.info(f"Enabled LoRA fine-tuning with rank={rank}, applied to {lora_applied} modules")
        self.lora_enabled = True

    def disable_lora(self):
        """Disable LoRA and restore original weights."""
        if not hasattr(self, "lora_enabled") or not self.lora_enabled:
            logger.warning("LoRA is not enabled")
            return

        lora_removed = 0

        # Remove LoRA from transformer blocks
        for block in self.dino_model.blocks:
            for module_name, module in block.named_children():
                if isinstance(module, LoRAWrapper):
                    # Restore original layer
                    setattr(block, module_name, module.original_layer)
                    lora_removed += 1

        # Remove LoRA from patch embedding
        if hasattr(self.dino_model.patch_embed, "proj"):
            proj_module = self.dino_model.patch_embed.proj
            if isinstance(proj_module, LoRAWrapper):
                self.dino_model.patch_embed.proj = proj_module.original_layer
                lora_removed += 1

        logger.info(f"Disabled LoRA fine-tuning, removed from {lora_removed} modules")
        self.lora_enabled = False

    def unfreeze_layers(self, num_layers: int = 1):
        """Unfreeze the last N transformer layers for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the end
        """
        if not self.freeze_backbone:
            logger.warning("Backbone is not frozen, no layers to unfreeze")
            return

        total_blocks = len(self.dino_model.blocks)
        for i in range(max(0, total_blocks - num_layers), total_blocks):
            for param in self.dino_model.blocks[i].parameters():
                param.requires_grad_(True)

        logger.info(f"Unfroze last {num_layers} transformer layers")

    def get_differential_lr_param_groups(
        self,
        base_lr: float = 1e-4,
        backbone_lr_multiplier: float = 0.1,
        head_lr_multiplier: float = 1.0,
        unfrozen_layers_lr_multiplier: float = 1.0,
    ) -> list[dict]:
        """Create parameter groups with differential learning rates.

        Args:
            base_lr: Base learning rate for unfrozen parameters
            backbone_lr_multiplier: Multiplier for frozen backbone parameters
            head_lr_multiplier: Multiplier for task-specific head parameters
            unfrozen_layers_lr_multiplier: Multiplier for unfrozen transformer layers

        Returns:
            List of parameter group dictionaries for optimizer
        """
        param_groups = []

        # Backbone parameters (patch embedding, position embeddings, etc.)
        backbone_params = []
        backbone_params.extend(self.dino_model.patch_embed.parameters())
        if hasattr(self.dino_model, "pos_embed"):
            backbone_params.append(self.dino_model.pos_embed)
        if hasattr(self.dino_model, "cls_token"):
            backbone_params.append(self.dino_model.cls_token)
        if hasattr(self.dino_model, "reg_tokens"):
            backbone_params.append(self.dino_model.reg_tokens)

        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": base_lr * backbone_lr_multiplier,
                    "name": "backbone",
                }
            )

        # Transformer blocks - separate frozen vs unfrozen
        frozen_blocks = []
        unfrozen_blocks = []

        for _i, block in enumerate(self.dino_model.blocks):
            block_params = list(block.parameters())
            if any(p.requires_grad for p in block_params):
                unfrozen_blocks.extend(block_params)
            else:
                frozen_blocks.extend(block_params)

        if frozen_blocks:
            param_groups.append(
                {
                    "params": frozen_blocks,
                    "lr": base_lr * backbone_lr_multiplier,
                    "name": "frozen_blocks",
                }
            )

        if unfrozen_blocks:
            param_groups.append(
                {
                    "params": unfrozen_blocks,
                    "lr": base_lr * unfrozen_layers_lr_multiplier,
                    "name": "unfrozen_blocks",
                }
            )

        # Task-specific head parameters (if any adaptation layers exist)
        head_params = []
        # Add any additional layers specific to medical adaptation
        # For now, this is empty but can be extended

        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": base_lr * head_lr_multiplier,
                    "name": "head",
                }
            )

        # Default group for any remaining parameters
        all_grouped_params = set()
        for group in param_groups:
            all_grouped_params.update(group["params"])

        remaining_params = []
        for param in self.parameters():
            if param not in all_grouped_params:
                remaining_params.append(param)

        if remaining_params:
            param_groups.append({"params": remaining_params, "lr": base_lr, "name": "default"})

        group_info = [f"{g['name']}: {g['lr']:.2e}" for g in param_groups]
        logger.info(
            f"Created {len(param_groups)} parameter groups with differential LRs: {group_info}"
        )

        return param_groups


class MedicalDINOv2SmallEncoder(MedicalDINOv2Encoder):
    """Medical-adapted DINOv2 Small encoder."""

    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(model_name="vit_small_patch14_dinov2", embed_dim=384, **kwargs)


class MedicalDINOv2BaseEncoder(MedicalDINOv2Encoder):
    """Medical-adapted DINOv2 Base encoder."""

    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(model_name="vit_base_patch14_dinov2", embed_dim=768, **kwargs)


class MedicalDINOv2LargeEncoder(MedicalDINOv2Encoder):
    """Medical-adapted DINOv2 Large encoder."""

    def __init__(self, **kwargs):
        """__init__."""
        super().__init__(model_name="vit_large_patch14_dinov2", embed_dim=1024, **kwargs)


# Alias for backward compatibility
MedicalDinoEncoder = MedicalDINOv2Encoder
