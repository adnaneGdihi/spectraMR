"""X-Diffusion Generator - Cross-Modal Diffusion
==============================================

Implementation of X-Diffusion for cross-modal generation between 2D and 3D.
Adapted for spectraMR project following SOLID principles.
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.timestep_embedding import sinusoidal_timestep_embedding
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="x_diffusion", training_mode="diffusion")
class XDiffusionGenerator(IGenerator, nn.Module):
    """X-Diffusion Generator for cross-modal generation.

    This generator implements cross-modal diffusion between 2D images
    and 3D volumes, enabling bidirectional generation and translation.

    Attributes:
        name (str): Model identifier
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        resolution_2d (Tuple[int, int]): 2D resolution
        resolution_3d (Tuple[int, int, int]): 3D resolution
        num_timesteps (int | None): Diffusion horizon T the time code spans
        cross_attention_dim (int): Cross-attention feature dimension

    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        resolution_2d: tuple[int, int] = (256, 256),
        resolution_3d: tuple[int, int, int] = (64, 128, 128),
        num_timesteps: int | None = None,
        cross_attention_dim: int = 512,
        conditioning_dim: int = 768,  # For DINOv2 or other conditioning
        num_layers: int = 8,
        num_heads: int = 8,
        diffusion_process: Any = None,  # Domain-specific noise model
        **kwargs: Any,
    ) -> None:
        """Initialize X-Diffusion generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            resolution_2d: 2D resolution (H, W)
            resolution_3d: 3D resolution (D, H, W)
            num_timesteps: Diffusion horizon T, injected by the builder from
                ``training.diffusion.timesteps``. Required to embed a nonzero step.
            cross_attention_dim: Cross-attention feature dimension
            conditioning_dim: Dimension of conditioning features
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            diffusion_process: Domain-specific diffusion process

        """
        super().__init__()

        self._name = "x_diffusion"
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Explicit type annotations for static type checkers
        self.resolution_2d: tuple[int, int] = resolution_2d
        self.resolution_3d: tuple[int, int, int] = resolution_3d
        self.num_timesteps = num_timesteps
        self.cross_attention_dim = cross_attention_dim
        self.conditioning_dim = conditioning_dim
        self.diffusion_process = diffusion_process

        # 2D encoder (for 2D -> 3D generation)
        self.encoder_2d = self._build_2d_encoder(cross_attention_dim)

        # 3D encoder (for 3D -> 2D generation)
        self.encoder_3d = self._build_3d_encoder(cross_attention_dim)

        # Cross-modal attention layers
        self.cross_attention_layers = nn.ModuleList(
            [CrossAttentionBlock(cross_attention_dim, num_heads) for _ in range(num_layers)],
        )

        # 2D decoder
        self.decoder_2d = self._build_2d_decoder(cross_attention_dim)

        # 3D decoder
        self.decoder_3d = self._build_3d_decoder(cross_attention_dim)

        # Time embedding for diffusion
        self.time_embed = nn.Sequential(
            nn.Linear(cross_attention_dim, cross_attention_dim * 4),
            nn.SiLU(),
            nn.Linear(cross_attention_dim * 4, cross_attention_dim),
        )

        # Conditioning projection for cross-attention
        self.conditioning_proj = nn.Linear(conditioning_dim, cross_attention_dim)

        # Output projections
        self.output_proj_2d = nn.Sequential(
            nn.Conv2d(cross_attention_dim, out_channels, 1),
            nn.Tanh(),
        )

        self.output_proj_3d = nn.Sequential(
            nn.Conv3d(cross_attention_dim, out_channels, 1),
            nn.Tanh(),
        )

        self.logger = logging.getLogger(__name__)

    def _build_2d_encoder(self, dim: int) -> nn.Module:
        """Build 2D encoder."""
        return nn.Sequential(
            nn.Conv2d(self.in_channels, dim // 4, 4, 2, 1),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim // 2, 4, 2, 1),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(dim // 2, dim, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def _build_3d_encoder(self, dim: int) -> nn.Module:
        """Build 3D encoder."""
        return nn.Sequential(
            nn.Conv3d(self.in_channels, dim // 4, 4, 2, 1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv3d(dim // 4, dim // 2, 4, 2, 1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv3d(dim // 2, dim, 4, 2, 1),  # 16 -> 8
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

    def _build_2d_decoder(self, dim: int) -> nn.Module:
        """Build 2D decoder that outputs exact target resolution."""
        return nn.Sequential(
            # Start from coarse 4x4 grid
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 4, dim // 8, 4, 2, 1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 8, dim, 4, 2, 1),  # 32 -> 64
            nn.ReLU(),
            # Final adaptive upsampling to exact target size
            nn.Upsample(size=self.resolution_2d, mode="bilinear", align_corners=False),
        )

    def _build_3d_decoder(self, dim: int) -> nn.Module:
        """Build 3D decoder that outputs exact target resolution."""
        # Calculate required upsampling factor from coarse grid to target
        d_target, h_target, w_target = self.resolution_3d
        # Start from 2x2x2 grid to reach target with stride-2 layers
        # 2 -> 4 -> 8 -> 16 -> 32 -> 64 for D
        # But we need asymmetric: D=64, H=128, W=128
        # Use adaptive upsampling for exact sizing
        return nn.Sequential(
            # Start from coarse 2x2x2 grid
            nn.ConvTranspose3d(dim, dim // 2, 4, 2, 1),  # 2 -> 4
            nn.ReLU(),
            nn.ConvTranspose3d(dim // 2, dim // 4, 4, 2, 1),  # 4 -> 8
            nn.ReLU(),
            nn.ConvTranspose3d(dim // 4, dim // 8, 4, 2, 1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose3d(dim // 8, dim, 4, 2, 1),  # 16 -> 32
            nn.ReLU(),
            # Final adaptive upsampling to exact target size
            nn.Upsample(size=self.resolution_3d, mode="trilinear", align_corners=False),
        )

    @property
    def name(self) -> str:
        """Returns the model name."""
        return self._name

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through X-Diffusion generator.

        Args:
            x: Input tensor (2D image or 3D volume)
            conditioning: Optional conditioning features (e.g., DINOv2 embeddings)
            timesteps: ``[B]`` diffusion steps of ``x``; ``None`` is the clean
                endpoint ``t = 0``, the one-shot synthesis :meth:`generate` runs.

        Returns:
            Generated output in the target modality

        forward method for XDiffusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            conditioning (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() == 4:  # 2D input -> 3D output
            output = self._generate_3d_from_2d(x, conditioning, timesteps)
            expected_shape = self.get_output_shape(x.shape)
            assert output.shape == expected_shape, (
                f"Output shape {output.shape} != expected {expected_shape}"
            )
            return output
        if x.dim() == 5:  # 3D input -> 2D output
            output = self._generate_2d_from_3d(x, conditioning, timesteps)
            expected_shape = self.get_output_shape(x.shape)
            assert output.shape == expected_shape, (
                f"Output shape {output.shape} != expected {expected_shape}"
            )
            return output
        raise ValueError(f"Unsupported input dimension: {x.dim()}")

    def _time_embedding(self, features: torch.Tensor, timesteps: torch.Tensor | None) -> torch.Tensor:
        """Embed the strategy's steps on the horizon-derived sinusoid (#2440)."""
        batch = features.shape[0]
        if timesteps is None:
            # At t = 0 every phase is zero, so the code is the same on any ladder.
            code = sinusoidal_timestep_embedding(
                torch.zeros(batch, device=features.device), features.shape[-1], scale="integer"
            )
        else:
            if timesteps.shape[0] != batch:
                raise ValueError(
                    f"XDiffusionGenerator received {timesteps.shape[0]} timesteps "
                    f"for a batch of {batch}."
                )
            if self.num_timesteps is None:
                raise ValueError(
                    "XDiffusionGenerator needs num_timesteps to embed a diffusion step; "
                    "the builder injects training.diffusion.timesteps."
                )
            code = sinusoidal_timestep_embedding(
                timesteps,
                features.shape[-1],
                max_timesteps=float(self.num_timesteps),
                scale="horizon_band",
            )
        return self.time_embed(code.to(features.dtype))

    def _generate_3d_from_2d(
        self,
        x_2d: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate 3D volume from 2D image."""
        features_2d = self.encoder_2d(x_2d)  # (B, cross_attention_dim)
        time_emb = self._time_embedding(features_2d, timesteps)

        # Prepare conditioning for cross-attention
        cond_features = None
        if conditioning is not None:
            cond_features = self.conditioning_proj(conditioning)
            # Add sequence dimension for attention: (B, 1, C)
            cond_features = cond_features.unsqueeze(1)

        # Apply cross-attention layers
        features = features_2d + time_emb
        features = features.unsqueeze(1)  # Add sequence dimension
        for layer in self.cross_attention_layers:
            features = layer(features, cond_features)
        features = features.squeeze(1)  # Remove sequence dimension

        # Seed a coarse 2x2x2 grid then decode to exact target
        # Expand features to spatial dimensions for 3D decoder
        features_expanded = features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        features_expanded = features_expanded.expand(-1, -1, 2, 2, 2)
        features_3d = features_expanded
        decoded_3d: torch.Tensor = self.decoder_3d(features_3d)
        output_3d: torch.Tensor = self.output_proj_3d(decoded_3d)
        # Decoder now outputs exact target resolution
        return output_3d

    def _generate_2d_from_3d(
        self,
        x_3d: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate 2D image from 3D volume."""
        features_3d = self.encoder_3d(x_3d)  # (B, cross_attention_dim)
        time_emb = self._time_embedding(features_3d, timesteps)

        # Prepare conditioning for cross-attention
        cond_features = None
        if conditioning is not None:
            cond_features = self.conditioning_proj(conditioning)
            # Add sequence dimension for attention: (B, 1, C)
            cond_features = cond_features.unsqueeze(1)

        # Apply cross-attention layers
        features = features_3d + time_emb
        features = features.unsqueeze(1)  # Add sequence dimension
        for layer in self.cross_attention_layers:
            features = layer(features, cond_features)
        features = features.squeeze(1)  # Remove sequence dimension

        # Seed minimal 4x4 spatial and decode to exact target 2D
        # Expand features to spatial dimensions for 2D decoder
        features_expanded = features.unsqueeze(-1).unsqueeze(-1)
        features_expanded = features_expanded.expand(-1, -1, 4, 4)
        features_2d = features_expanded
        decoded_2d: torch.Tensor = self.decoder_2d(features_2d)
        output_2d: torch.Tensor = self.output_proj_2d(decoded_2d)
        # Decoder now outputs exact target resolution
        return output_2d

    def generate(
        self,
        z: torch.Tensor,
        conditioning: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate samples using cross-modal diffusion.

        Args:
            z: Input tensor in source modality
            conditioning: Optional conditioning features (e.g., DINOv2 embeddings)
            **kwargs: Additional generation parameters

        Returns:
            Generated output in target modality

        """
        return self.forward(z, conditioning)

    def get_output_shape(
        self,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Returns the output shape for a given input shape.

        Args:
            input_shape: Input tensor shape

        Returns:
            Output shape in target modality

        """
        if len(input_shape) == 4:  # 2D input -> 3D output
            return (input_shape[0], self.out_channels, *self.resolution_3d)
        if len(input_shape) == 5:  # 3D input -> 2D output
            return (input_shape[0], self.out_channels, *self.resolution_2d)
        raise ValueError(f"Unsupported input shape: {input_shape}")

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for modality translation with optional conditioning.

    Supports both self-attention (when conditioning is None) and
    cross-attention (when conditioning is provided as key/value).
    """

    def __init__(self, dim: int, num_heads: int):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with optional cross-attention conditioning.

        Args:
            x: Query features of shape (batch, seq_len, dim)
            conditioning: Optional conditioning features for cross-attention,
                of shape (batch, cond_seq_len, dim). If None, performs
                self-attention.

        Returns:
            Output features of shape (batch, seq_len, dim)

        forward method for CrossAttentionBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            conditioning (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Self-attention or cross-attention
        x_norm = self.norm(x)
        if conditioning is not None:
            # Cross-attention: x as query, conditioning as key/value
            cond_norm = self.norm(conditioning)
            attn_out, _ = self.attn(x_norm, cond_norm, cond_norm)
        else:
            # Self-attention
            attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm(x))
        return x
