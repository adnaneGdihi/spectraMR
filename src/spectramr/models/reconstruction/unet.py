"""Configurable UNet Generator
===========================

A unified, highly configurable UNet generator that consolidates multiple
UNet variants into a single, flexible implementation. Supports various
architectural options including attention, deep supervision,
skip connections, and different normalization types.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.dc_settings import DC_SSOT_KEYS
from spectramr.models.blocks.attention import CBAMSpatialAttention as SpatialAttention
from spectramr.models.blocks.attention import ChannelAttention
from spectramr.models.blocks.base import BaseGANBlock
from spectramr.models.blocks.normalization import get_adaptive_groups
from spectramr.models.interfaces import IGenerator, LatentAccessMixin
from spectramr.models.registry import register_model


class AttentionType(Enum):
    """Types of attention mechanisms available."""

    NONE = "none"
    CHANNEL = "channel"
    SPATIAL = "spatial"
    SELF = "self"
    CROSS_CONTRAST_OLMPA = "cross_contrast_olmpa"
    DUAL_DOMAIN = "dual_domain"
    KAN_DUAL_DOMAIN = "kan_dual_domain"
    WAVELET_FREQ = "wavelet_freq"
    KERNELIZED = "kernelized"
    SPARSE = "sparse"
    NULL_SPACE_DUAL_DOMAIN = "null_space_dual_domain"
    HERMITIAN_NULL_SPACE = "hermitian_null_space"


class NormalizationType(Enum):
    """Types of normalization available."""

    BATCH = "batch"
    GROUP = "group"
    INSTANCE = "instance"
    LAYER = "layer"


class BlockType(Enum):
    """Types of residual blocks available."""

    STANDARD = "standard"
    BOTTLENECK = "bottleneck"
    DENSE = "dense"


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time conditioning."""

    def __init__(self, dim: int):
        """__init__.

        Args:
            dim (int): Description.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            time (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SinusoidalPositionEmbeddings.

        Executes PyTorch tensor operations.

        Args:
            time (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeInjectionBlock(nn.Module):
    """Injects time embedding into feature map."""

    def __init__(self, channels: int, time_emb_dim: int):
        """__init__.

        Args:
            channels (int): Description.
            time_emb_dim (int): Description.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels),
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TimeInjectionBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if emb is None:
            return x
        t_emb = self.proj(emb)
        # Broadcast to spatial dimensions (B, C, 1, 1)
        t_emb = t_emb[:, :, None, None]
        return x + t_emb


@dataclass
class UNetConfig:
    """Configuration for UNet."""

    in_channels: int = 3
    out_channels: int = 3
    features: tuple[int, ...] = (64, 128, 256, 512, 1024)
    depth: int = 4
    bilinear: bool = False
    attention_type: AttentionType = AttentionType.NONE
    norm_type: NormalizationType = (
        NormalizationType.GROUP
    )  # GroupNorm for gradient accumulation compatibility
    block_type: BlockType = BlockType.STANDARD
    use_attention: bool = False
    use_residual: bool = True
    dropout: float = 0.0
    deep_supervision: bool = False
    output_activation: nn.Module | None = None
    time_dim: int | None = None
    z_dim: int | None = None  # Added for compatibility with ModelFactory injection


class ConfigurableResidualBlock(BaseGANBlock):
    """Configurable residual block with various options."""

    def __init__(
        self,
        channels: int,
        block_type: BlockType = BlockType.STANDARD,
        norm_type: NormalizationType = NormalizationType.BATCH,
        use_attention: bool = False,
        attention_type: AttentionType = AttentionType.NONE,
        dropout: float = 0.0,
        groups: int = 1,
        dilation: int = 1,
        time_emb_dim: int | None = None,
    ):
        """__init__.

        Args:
            channels (int): Description.
            block_type (BlockType): Description.
            norm_type (NormalizationType): Description.
            use_attention (bool): Description.
            attention_type (AttentionType): Description.
            dropout (float): Description.
            groups (int): Description.
            dilation (int): Description.
            time_emb_dim (int | None): Description.
        """
        super().__init__()

        self.channels = channels
        self.block_type = block_type

        # Time embedding projection
        if time_emb_dim is not None:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, channels),
            )
        else:
            self.time_proj = None

        # Normalization factory
        def create_norm_layer(channels: int) -> nn.Module:
            """create_norm_layer.

            Args:
                channels (int): Description.
            Returns:
                nn.Module: Description.
            """
            if norm_type == NormalizationType.BATCH:
                return nn.BatchNorm2d(channels)
            if norm_type == NormalizationType.GROUP:
                return nn.GroupNorm(get_adaptive_groups(channels), channels)
            if norm_type == NormalizationType.INSTANCE:
                return nn.InstanceNorm2d(channels)
            if norm_type == NormalizationType.LAYER:
                return nn.LayerNorm([channels, 1, 1])
            return nn.Identity()

        norm_layer = create_norm_layer

        # Main convolution path
        if block_type == BlockType.STANDARD:
            self.conv1 = nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                groups=groups,
                dilation=dilation,
                bias=False,
            )
            self.norm1 = norm_layer(channels)
            self.conv2 = nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                groups=groups,
                dilation=dilation,
                bias=False,
            )
            self.norm2 = norm_layer(channels)
        elif block_type == BlockType.BOTTLENECK:
            reduced_channels = channels // 4
            self.conv1 = nn.Conv2d(channels, reduced_channels, 1, bias=False)
            self.norm1 = norm_layer(reduced_channels)
            self.conv2 = nn.Conv2d(
                reduced_channels,
                reduced_channels,
                3,
                padding=dilation,
                groups=groups,
                dilation=dilation,
                bias=False,
            )
            self.norm2 = norm_layer(reduced_channels)
            self.conv3 = nn.Conv2d(reduced_channels, channels, 1, bias=False)
            self.norm3 = norm_layer(channels)
        else:
            raise NotImplementedError(
                f"ConfigurableResidualBlock does not implement "
                f"block_type={block_type!r}; supported: "
                f"{BlockType.STANDARD!r}, {BlockType.BOTTLENECK!r}"
            )

        # Activation and dropout
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Attention mechanism
        if use_attention:
            if attention_type == AttentionType.CHANNEL:
                self.attention = ChannelAttention(channels)
            elif attention_type == AttentionType.SPATIAL:
                self.attention = SpatialAttention()
            else:
                raise ValueError(
                    f"ConfigurableResidualBlock does not implement "
                    f"attention_type={attention_type!r}; supported: "
                    f"AttentionType.CHANNEL, AttentionType.SPATIAL"
                )
        else:
            self.attention = nn.Identity()

        # Skip connection scaling (clamped in forward for stability)
        self.skip_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.
        """
        identity = x

        if self.block_type == BlockType.STANDARD:
            out = self.conv1(x)
            out = self.norm1(out)

            # Inject time embedding
            if self.time_proj is not None and emb is not None:
                t_emb = self.time_proj(emb)
                # Broadcast to spatial dimensions (B, C, 1, 1)
                t_emb = t_emb[:, :, None, None]
                out = out + t_emb

            out = self.activation(out)
            out = self.dropout(out)

            out = self.conv2(out)
            out = self.norm2(out)
        elif self.block_type == BlockType.BOTTLENECK:
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.norm2(out)

            # Inject time embedding in the middle of bottleneck
            if self.time_proj is not None and emb is not None:
                t_emb = self.time_proj(emb)
                # Broadcast to spatial dimensions (B, C, 1, 1)
                t_emb = t_emb[:, :, None, None]
                out = out + t_emb

            out = self.activation(out)
            out = self.dropout(out)

            out = self.conv3(out)
            out = self.norm3(out)

        # Attention
        out = self.attention(out)

        # Skip connection with scaling. Clamp to a stable range to avoid
        # exploding or sign-flipped residuals during early training.
        skip_scale = torch.clamp(self.skip_scale, 0.0, 1.0)
        return self.activation(out + skip_scale * identity)


class TimeAwareSequential(nn.Sequential):
    """Sequential container that passes time embeddings to compatible modules.

    .. deprecated::
        Import from ``spectramr.models.blocks.time_aware_sequential`` instead.
    """

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.
        """
        for module in self:
            if isinstance(module, (ConfigurableResidualBlock, TimeInjectionBlock)):
                x = module(x, emb)
            else:
                x = module(x)
        return x


@register_model(
    name="configurable_unet",
    training_mode="reconstruction",
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class UNet(LatentAccessMixin, nn.Module, IGenerator):
    """A highly configurable UNet generator that consolidates
    multiple UNet variants.

    This generator supports various architectural options:
    - Different attention mechanisms (channel, spatial, self-attention)
    - Multiple normalization types (batch, group, instance, layer)
    - Various residual block types (standard, bottleneck, dense)
    - Deep supervision for better gradient flow
    - Configurable feature maps and depth
    - Bilinear vs transposed convolution upsampling
    """

    __expected_kwargs__ = {f.name for f in fields(UNetConfig)}

    def __init__(
        self,
        config: UNetConfig | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            config (UNetConfig | None): Description.
        """
        super().__init__()

        if config is None:
            # Pop keys not in UNetConfig but passed by builder or diffusion wrappers
            kwargs.pop("acceleration_config", None)
            kwargs.pop("features", None)
            kwargs.pop("base_channels", None)
            kwargs.pop("channel_multipliers", None)
            kwargs.pop("num_res_blocks", None)
            kwargs.pop("attention_resolutions", None)
            kwargs.pop("model_type", None)
            kwargs.pop("feature_dim", None)
            kwargs.pop("max_features", None)
            kwargs.pop("num_residual_blocks", None)
            # ``physics.data_consistency`` is forwarded into every **kwargs
            # generator by ``generator_kwargs`` step 3c, unconditionally — the
            # block is always present (schema default) and step 3c does not
            # consult ``enabled``. UNet builds no DC layer, so it discards the
            # whole set. Derive it from DC_SSOT_KEYS rather than restating it:
            # a hand-copied subset is what broke here when the table grew the
            # three noise keys and this list did not (non-negotiable 17).
            for _dc_kwarg, _dc_field in DC_SSOT_KEYS:
                kwargs.pop(_dc_kwarg, None)
            # Top-level model.* fields auto-forwarded by GeneratorBuilder.build()
            # since the May 2026 spec-card refactor — UNetConfig doesn't know
            # about them and the strict TypeError below otherwise breaks every
            # ``standard_unet`` arm. Their owner is ``SKIP_MODEL_FIELDS`` in
            # src/spectramr/infrastructure/builders/generator_contract.py.
            kwargs.pop("spatial_dims", None)
            kwargs.pop("input_type", None)
            kwargs.pop("output_type", None)
            kwargs.pop("input_domain", None)
            kwargs.pop("output_domain", None)
            kwargs.pop("model_domain", None)
            # ``kan_type`` is an evidential/KAN-UNet wrapper kwarg; the
            # vanilla UNetConfig path doesn't consume it (it's handled by
            # the wrapper class on the way down).
            kwargs.pop("kan_type", None)
            # Weight-init fields live on the top-level model schema and are
            # auto-forwarded by GeneratorBuilder; they're applied later by
            # initialization_integration.py rather than the model constructor.
            kwargs.pop("weight_init_type", None)
            kwargs.pop("weight_init_gain", None)
            kwargs.pop("weight_init_activation", None)
            kwargs.pop("apply_weight_init", None)
            # Map 'residual' → 'use_residual' (common config alias)
            residual_val = kwargs.pop("residual", None)
            if residual_val is not None and "use_residual" not in kwargs:
                kwargs["use_residual"] = residual_val

            # Validate kwargs against UNetConfig fields
            valid_keys = {f.name for f in fields(UNetConfig)}
            for key in kwargs:
                if key not in valid_keys:
                    raise TypeError(
                        f"Unexpected keyword argument '{key}' for UNetConfig",
                    )
            config = UNetConfig(**kwargs)
        self.config = config

        assert isinstance(config.in_channels, int) and config.in_channels > 0
        assert isinstance(config.out_channels, int) and config.out_channels > 0
        assert isinstance(config.depth, int) and config.depth > 0
        assert isinstance(config.bilinear, bool)
        assert isinstance(config.attention_type, AttentionType)
        assert isinstance(config.norm_type, NormalizationType)
        assert isinstance(config.block_type, BlockType)
        assert isinstance(config.use_attention, bool)
        assert isinstance(config.use_residual, bool)
        assert isinstance(config.dropout, float) and (0.0 <= config.dropout < 1.0)
        assert isinstance(config.deep_supervision, bool)

        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.features = config.features
        self.depth = config.depth
        self.bilinear = config.bilinear
        self.attention_type = config.attention_type
        self.norm_type = config.norm_type
        self.block_type = config.block_type
        self.use_attention = config.use_attention
        self.use_residual = config.use_residual
        self.dropout = config.dropout
        self.deep_supervision = config.deep_supervision
        self.time_dim = config.time_dim
        self.output_activation = config.output_activation or (
            nn.Tanh() if config.use_attention else nn.Identity()
        )

        # Time embedding MLP
        if self.time_dim is not None:
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(self.time_dim),
                nn.Linear(self.time_dim, self.time_dim * 4),
                nn.SiLU(),
                nn.Linear(self.time_dim * 4, self.time_dim),
            )
        else:
            self.time_mlp = None

        # Generate descriptive name based on configuration
        name_parts = ["ConfigurableUNet"]
        if config.attention_type != AttentionType.NONE:
            name_parts.append(str(config.attention_type.value).title())
        if config.block_type != BlockType.STANDARD:
            name_parts.append(str(config.block_type.value).title())
        if config.deep_supervision:
            name_parts.append("DeepSup")
        if config.use_residual:
            name_parts.append("Residual")
        if config.bilinear:
            name_parts.append("Bilinear")
        else:
            name_parts.append("TransConv")
        self._name = "_".join(name_parts)

        # Build encoder
        self.encoder = nn.ModuleList()
        # Use depth to determine number of encoder blocks
        # depth encoder blocks total (including initial)
        max_features = config.depth + 1
        effective_features = (
            config.features[:max_features]
            if max_features <= len(config.features)
            else config.features
        )

        # First encoder block (no downsampling)
        self.encoder.append(
            self._make_encoder_block(
                self.in_channels,
                effective_features[0],
                include_pool=False,
            ),
        )

        # Subsequent encoder blocks (with downsampling)
        for i in range(len(effective_features) - 1):
            self.encoder.append(
                self._make_encoder_block(
                    effective_features[i],
                    effective_features[i + 1],
                    include_pool=True,
                ),
            )

        # Build decoder
        self.decoder = nn.ModuleList()
        factor = 2 if config.bilinear else 1

        for i in reversed(range(len(effective_features) - 1)):
            in_channels = effective_features[i + 1] // factor
            out_channels = effective_features[i]
            # For both bilinear and non-bilinear, account for skip connection
            # channels since concatenation happens before the decoder block
            in_channels = effective_features[i + 1] + effective_features[i]
            if not config.bilinear:
                # For transposed conv, the upsampling is done by
                # ConvTranspose2d so we don't divide by factor
                pass
            block = self._make_decoder_block(in_channels, out_channels)
            self.decoder.append(block)

        # Output convolution
        self.outc = nn.Conv2d(
            config.features[0],
            config.out_channels,
            kernel_size=1,
        )

        # Deep supervision outputs
        if config.deep_supervision:
            self.deep_outputs = nn.ModuleList(
                [
                    nn.Conv2d(feature, config.out_channels, kernel_size=1)
                    for feature in reversed(config.features[:-1])
                ],
            )

    def _make_encoder_block(
        self,
        in_channels: int,
        out_channels: int,
        include_pool: bool = True,
    ) -> nn.Module:
        """Create an encoder block with optional downsampling."""
        layers = []

        # Downsampling (optional for first block)
        if include_pool:
            layers.append(nn.MaxPool2d(2))

        # Double convolution with residual if enabled
        if self.config.use_residual:
            # First, adjust channels if needed
            if in_channels != out_channels:
                layers.append(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                )
                layers.append(self._get_norm_layer(out_channels))
                layers.append(nn.ReLU(inplace=True))

            layers.append(
                ConfigurableResidualBlock(
                    out_channels,
                    block_type=self.config.block_type,
                    norm_type=self.config.norm_type,
                    use_attention=self.config.use_attention,
                    attention_type=self.config.attention_type,
                    dropout=self.config.dropout,
                    time_emb_dim=self.time_dim,
                ),
            )
        else:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    self._get_norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )

            # Inject time embedding if configured
            if self.time_dim is not None:
                layers.append(TimeInjectionBlock(out_channels, self.time_dim))

            layers.extend(
                [
                    (
                        nn.Dropout2d(self.config.dropout)
                        if self.config.dropout > 0
                        else nn.Identity()
                    ),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    self._get_norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                    (
                        nn.Dropout2d(self.config.dropout)
                        if self.config.dropout > 0
                        else nn.Identity()
                    ),
                ],
            )

        return TimeAwareSequential(*layers)

    def _make_decoder_block(
        self,
        in_channels: int,
        out_channels: int,
    ) -> nn.Module:
        """Create a decoder block with upsampling."""
        if self.config.bilinear:
            # Upsampling handled in forward via interpolation; skip keeps shape
            up = nn.Identity()
        else:
            # ConvTranspose2d performs both upsampling and channel reduction
            up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        reduction_layers: list[nn.Module] = []
        if self.config.bilinear:
            # Reduce concatenated skip channels back to the target width
            reduction_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    self._get_norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                ],
            )

        if self.config.use_residual:
            conv_body: nn.Module = ConfigurableResidualBlock(
                out_channels,
                block_type=self.config.block_type,
                norm_type=self.config.norm_type,
                use_attention=self.config.use_attention,
                attention_type=self.config.attention_type,
                dropout=self.config.dropout,
                time_emb_dim=self.time_dim,
            )
        else:
            conv_body_layers = [
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    bias=False,
                ),
                self._get_norm_layer(out_channels),
                nn.ReLU(inplace=True),
            ]

            # Inject time embedding if configured
            if self.time_dim is not None:
                conv_body_layers.append(TimeInjectionBlock(out_channels, self.time_dim))

            conv_body_layers.extend(
                [
                    (
                        nn.Dropout2d(self.config.dropout)
                        if self.config.dropout > 0
                        else nn.Identity()
                    ),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        3,
                        padding=1,
                        bias=False,
                    ),
                    self._get_norm_layer(out_channels),
                    nn.ReLU(inplace=True),
                    (
                        nn.Dropout2d(self.config.dropout)
                        if self.config.dropout > 0
                        else nn.Identity()
                    ),
                ]
            )
            conv_body = TimeAwareSequential(*conv_body_layers)

        conv_block = TimeAwareSequential(*reduction_layers, conv_body)
        return TimeAwareSequential(up, conv_block)

    def _get_norm_layer(self, channels: int) -> nn.Module:
        """Get normalization layer based on type."""
        norm_type = self.norm_type
        if isinstance(norm_type, str):
            try:
                norm_type = NormalizationType(norm_type.lower())
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported normalization type: {norm_type}",
                ) from exc

        if norm_type == NormalizationType.BATCH:
            return nn.BatchNorm2d(channels)
        if norm_type == NormalizationType.GROUP:
            return nn.GroupNorm(get_adaptive_groups(channels), channels)
        if norm_type == NormalizationType.INSTANCE:
            return nn.InstanceNorm2d(channels)
        if norm_type == NormalizationType.LAYER:
            return nn.LayerNorm([channels, 1, 1])
        raise ValueError(f"Unsupported normalization type: {norm_type}")

    @property
    def name(self) -> str:
        """Return descriptive model name based on configuration."""
        components = ["ConfigurableUNet"]

        if self.config.use_attention:
            attn_type = self.config.attention_type.value.title()
            components.append(f"{attn_type}Attn")
        if self.config.use_residual:
            components.append(f"{self.config.block_type.value.title()}Res")
        if self.config.deep_supervision:
            components.append("DeepSup")
        if self.config.bilinear:
            components.append("Bilinear")
        else:
            components.append("TransConv")

        return "_".join(components)

    def forward(
        self,
        x: torch.Tensor,
        *,
        encoder_state: dict[str, torch.Tensor] | None = None,
        encoder_feature_maps: Iterable[torch.Tensor] | None = None,
        encoder_bottleneck: torch.Tensor | None = None,
        encoder_embedding: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the configurable UNet.

        Args:
            x: Input tensor [N, C, H, W] where N is batch size,
               C is number of channels, H, W are spatial dimensions
            encoder_state: Optional cached encoder state for reuse
            encoder_feature_maps: Optional cached feature maps from encoder
            encoder_bottleneck: Optional cached bottleneck features
            encoder_embedding: Optional cached embedding features
            time: Optional time steps for diffusion [N]

        Returns:
            Output tensor [N, C_out, H, W] preserving input
            spatial dimensions H, W

        When encoder artefacts are provided, the decoder reuses those cached
        features instead of recomputing the encoder stack.

        forward method for UNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""

        input_tensor = x
        input_height, input_width = input_tensor.shape[-2:]

        # Compute time embedding if time is provided and model is configured for it
        t_emb = None
        if self.time_mlp is not None and time is not None:
            t_emb = self.time_mlp(time)

        # Contrast conditioning: additively fuse contrast embedding into time embedding for DECODER
        contrast_emb = _kwargs.get("contrast_emb")
        combined_emb = t_emb
        if contrast_emb is not None:
            combined_emb = (t_emb + contrast_emb) if t_emb is not None else contrast_emb

        if encoder_state is not None:
            if encoder_feature_maps is None:
                encoder_feature_maps = encoder_state.get("skips")
            if encoder_bottleneck is None:
                encoder_bottleneck = encoder_state.get("bottleneck")
            if encoder_embedding is None:
                encoder_embedding = encoder_state.get("embedding")

        use_external = encoder_feature_maps is not None and encoder_bottleneck is not None

        if use_external:
            shallow_to_deep = list(encoder_feature_maps)
            decoder_input = encoder_bottleneck
        else:
            encoder_outputs: list[torch.Tensor] = []
            current = input_tensor
            for encoder_block in self.encoder:
                if isinstance(encoder_block, TimeAwareSequential):
                    current = encoder_block(current, t_emb)
                else:
                    current = encoder_block(current)
                encoder_outputs.append(current)
            decoder_input = encoder_outputs[-1]
            shallow_to_deep = encoder_outputs[:-1]
            if encoder_embedding is None:
                encoder_embedding = torch.mean(
                    decoder_input,
                    dim=(-2, -1),
                )

        decoder_skip_sequence = list(reversed(shallow_to_deep))

        if self.deep_supervision and hasattr(self, "deep_outputs"):
            deep_outputs = []
        else:
            deep_outputs = None

        current = decoder_input
        for i, decoder_block in enumerate(self.decoder):
            if self.bilinear:
                current = F.interpolate(
                    current,
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                )
                if i < len(decoder_skip_sequence):
                    skip = decoder_skip_sequence[i]
                    if skip.shape[2:] != current.shape[2:]:
                        skip = F.interpolate(
                            skip,
                            size=current.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        )
                    current = torch.cat([skip, current], dim=1)

                if isinstance(decoder_block, TimeAwareSequential):
                    current = decoder_block(current, combined_emb)
                else:
                    current = decoder_block(current)
            else:
                if i < len(decoder_skip_sequence):
                    skip = decoder_skip_sequence[i]
                    if skip.shape[2:] != current.shape[2:]:
                        skip = F.interpolate(
                            skip,
                            size=current.shape[2:],
                            mode="bilinear",
                            align_corners=True,
                        )
                    current = torch.cat([skip, current], dim=1)

                if isinstance(decoder_block, TimeAwareSequential):
                    current = decoder_block(current, combined_emb)
                else:
                    current = decoder_block(current)

            if deep_outputs is not None and i < len(self.deep_outputs):
                out = self.deep_outputs[i](current)
                out = F.interpolate(
                    out,
                    size=(input_height, input_width),
                    mode="bilinear",
                    align_corners=False,
                )
                deep_outputs.append(out)

        if deep_outputs is not None and len(deep_outputs) > 0:
            logits = self.outc(current)
            final_out = F.interpolate(
                logits,
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
            )
            deep_outputs.append(final_out)
            current = torch.mean(torch.stack(deep_outputs), dim=0)
        else:
            current = self.outc(current)
            # Restore the documented contract ("Output ... preserving input
            # spatial dimensions H, W"): a non-power-of-two-divisible input
            # (e.g. W=436) floors through the strided encoder and the decoder
            # upsamples back to 432, not 436 — so ``self.outc(current)`` is 4 px
            # short of the target and a downstream L1/loss raises "size of tensor
            # a (432) must match b (436)" (mrixfields b25_cartoon_texture). The
            # deep-supervision branch above already interpolates to
            # (input_height, input_width); mirror it here. Guarded so the common
            # divisible case stays an exact no-op (no bilinear resample).
            if current.shape[-2:] != (input_height, input_width):
                current = F.interpolate(
                    current,
                    size=(input_height, input_width),
                    mode="bilinear",
                    align_corners=False,
                )

        if encoder_embedding is None:
            encoder_embedding = torch.mean(
                decoder_input,
                dim=(-2, -1),
            )

        self._set_latent_state(
            embedding=encoder_embedding,
            feature_maps=shallow_to_deep,
            bottleneck=decoder_input,
        )

        return self.output_activation(current)

    def generate(
        self,
        z: torch.Tensor | None = None,
        *,
        batch_size: int = 1,
        height: int = 64,
        width: int = 64,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate samples - UNet treats z as an input image."""

        if z is None:
            z = self._sample_latents(
                batch_size=batch_size,
                height=height,
                width=width,
                device=device,
                **kwargs,
            )
        if z is None:
            raise ValueError("Latent tensor must not be empty.")
        return self.forward(z)

    def _sample_latents(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Sample latent inputs compatible with the generator."""

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        device = device or next(self.parameters()).device
        return torch.randn(
            batch_size,
            self.in_channels,
            height,
            width,
            device=device,
        )

    def get_output_shape(self, input_shape: tuple[int, ...] | None = None) -> tuple[int, ...]:
        """Calculate output shape for given input shape.

        Args:
            input_shape: Input tensor shape [N, C, H, W]. If None,
                        returns default shape (1, out_channels, 256, 256)

        Returns:
            Output tensor shape [N, C_out, H, W] where spatial
            dimensions H, W are preserved from input
        """
        if input_shape is None:
            return (1, self.out_channels, 256, 256)
        n, _, h, w = input_shape
        return (n, self.out_channels, h, w)

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_configurable_unet_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    features: tuple[int, ...] = (64, 128, 256, 512, 1024),
    depth: int = 4,
    bilinear: bool = False,
    attention_type: str = "none",
    norm_type: str = "batch",
    block_type: str = "standard",
    use_attention: bool = False,
    use_residual: bool = True,
    dropout: float = 0.0,
    deep_supervision: bool = False,
    output_activation: nn.Module | None = None,
    **kwargs,
) -> ConfigurableUNetGenerator:
    """Factory function to create a ConfigurableUNetGenerator with
    string-based config.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        features: Feature map sizes for each level
        depth: Depth of the U-Net
        bilinear: Whether to use bilinear upsampling
        attention_type: Type of attention mechanism
            ("none", "channel", "spatial", "self")
        norm_type: Type of normalization
            ("batch", "group", "instance", "layer")
        block_type: Type of residual block
            ("standard", "bottleneck", "dense")
        use_attention: Whether to use attention mechanisms
        use_residual: Whether to use residual connections
        dropout: Dropout rate
        deep_supervision: Whether to use deep supervision
        output_activation: Output activation function
        **kwargs: Additional arguments

    Returns:
        ConfigurableUNetGenerator instance

    """
    # Convert string arguments to enums
    attention_enum = AttentionType(attention_type.lower())
    norm_enum = NormalizationType(norm_type.lower())
    block_enum = BlockType(block_type.lower())

    return ConfigurableUNetGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        depth=depth,
        bilinear=bilinear,
        attention_type=attention_enum,
        norm_type=norm_enum,
        block_type=block_enum,
        use_attention=use_attention,
        use_residual=use_residual,
        dropout=dropout,
        deep_supervision=deep_supervision,
        output_activation=output_activation,
        **kwargs,
    )


# Convenience functions for backward compatibility
def create_standard_unet_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    bilinear: bool = False,
) -> ConfigurableUNetGenerator:
    """Create a standard UNet generator."""
    return create_configurable_unet_generator(
        in_channels=in_channels,
        out_channels=out_channels,
        bilinear=bilinear,
        attention_type="none",
        norm_type="batch",
        block_type="standard",
        use_attention=False,
        use_residual=False,
        deep_supervision=False,
    )


def create_enhanced_unet_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    bilinear: bool = False,
    deep_supervision: bool = True,
    use_attention: bool = False,
) -> ConfigurableUNetGenerator:
    """Create an enhanced UNet generator with optional features."""
    return create_configurable_unet_generator(
        in_channels=in_channels,
        out_channels=out_channels,
        bilinear=bilinear,
        attention_type="channel" if use_attention else "none",
        norm_type="batch",
        block_type="standard",
        use_attention=use_attention,
        use_residual=True,
        deep_supervision=deep_supervision,
    )


def create_attention_unet_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    bilinear: bool = False,
) -> ConfigurableUNetGenerator:
    """Create an attention UNet generator."""
    return create_configurable_unet_generator(
        in_channels=in_channels,
        out_channels=out_channels,
        bilinear=bilinear,
        attention_type="channel",
        norm_type="batch",
        block_type="standard",
        use_attention=True,
        use_residual=True,
        deep_supervision=False,
        output_activation=nn.Tanh(),
    )


# Class-name aliases for the canonical UNet (registered as "configurable_unet").
# Previously these were three empty `pass`-body subclasses with two
# @register_model decorators. The registrations now live in
# ``src/models/stubs.py::register_aliases()`` (which is the codebase's
# documented mechanism for "this YAML name resolves to that real class").
# Kept as module-level aliases because 41+ call sites import these names
# directly (StandardUNetGenerator: 16, EnhancedUNetGenerator: 7,
# ConfigurableUNetGenerator: 4, StandardUNet: 14).
StandardUNetGenerator = UNet
EnhancedUNetGenerator = UNet
ConfigurableUNetGenerator = UNet
StandardUNet = UNet


__all__ = [
    "AttentionType",
    "BlockType",
    "ConfigurableUNetGenerator",
    "EnhancedUNetGenerator",
    "NormalizationType",
    "StandardUNet",
    "StandardUNetGenerator",
    "UNet",
    "UNetConfig",
    "create_attention_unet_generator",
    "create_configurable_unet_generator",
    "create_enhanced_unet_generator",
    "create_standard_unet_generator",
]
