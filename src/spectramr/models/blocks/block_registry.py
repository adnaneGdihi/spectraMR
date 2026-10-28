"""Unified Block Registry and Dispatcher.

Implements Registry-Dispatcher pattern for neural network blocks.
This provides compositional architecture for constructing models from
registered, reusable components.
"""

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# Import convolutional blocks for U-Net strategies
from .convolutional import DoubleConv

# Previously-orphaned attention blocks (each in its own canonical home under
# blocks/) wired into BLOCK_REGISTRY at the bottom of this file. No cycle back
# to block_registry — these modules import only torch.
from .dual_domain_attention_kan import (
    CrossDomainAttention,
    MultiScaleFreqBandAttention,
    RadialBandAttention,
)

# Import fusion blocks for multi-input fusion strategies
from .fusion import (
    AdaptiveFusionBlock,
    AttentionFusionBlock,
    FeatureFusionBlock,
    MultiModalFusionBlock,
    MultiScaleFusionBlock,
    ResidualFusionBlock,
)
from .graph_attention import GraphAttentionBlock, LocalGraphAttentionBlock
from .interfaces import IBlockStrategy
from .kspace_swin_attention import KSpaceRadialSwinBlock, RadialWindowAttention
from .multi_contrast_kspace_attention import MultiContrastKSpaceCrossAttention

# Import residual blocks
from .residual import PreActResidualBlock, ResidualBlock, UnifiedResidualBlock
from .structure_tensor_attention import StructureTensorAttention
from .vf_modules import SpatiotemporalMarkerAttention

# Trellis components. ``TransformerBlock`` is a legacy alias for
# ``TrellisLegacyBlock`` (see trellis_transformer.py). Per CLAUDE.md #9,
# only ``ImportError`` is allowed to silently disable the registrations
# (e.g. when the optional torch3d dependency is missing); other errors
# should surface so we don't silently skip ~4 block registrations under
# a typo. See TODO/audit/10_models_blocks_layers.md F5.
try:  # pragma: no cover - optional dependency
    from .trellis_attention import (
        MultiHeadAttention,
        TrellisMultiHeadAttention,
        TrellisSelfAttention2D,
    )
    from .trellis_transformer import AbsolutePositionEmbedder, TransformerBlock
except ImportError:  # pragma: no cover - keep registry usable without trellis
    TrellisMultiHeadAttention = None  # type: ignore[assignment]
    AbsolutePositionEmbedder = None  # type: ignore[assignment]
    TransformerBlock = None  # type: ignore[assignment]
    MultiHeadAttention = None  # type: ignore[assignment]
    TrellisSelfAttention2D = None  # type: ignore[assignment]


def create_block(block_type: str, **kwargs: Any) -> nn.Module:
    """Factory that returns the CONCRETE block instance.

    Removes the BlockLayer shim, allowing full JIT graph capture.

    Args:
        block_type: String identifier for block (in BLOCK_REGISTRY)
        **kwargs: Configuration parameters for the block

    Returns:
        Instantiated block module

    Raises:
        ValueError: If block_type not found in registry
    """
    if block_type not in BLOCK_REGISTRY:
        available = ", ".join(sorted(BLOCK_REGISTRY.keys()))
        raise ValueError(f"Unknown block_type: '{block_type}'. Available blocks: {available}")

    block_class = BLOCK_REGISTRY[block_type]
    return block_class(**kwargs)


class BlockLayer(nn.Module):
    """Universal dispatcher for instantiating block types.

    DEPRECATED: Use create_block() instead to avoid graph breaks.

    Args:
        block_type: String identifier for block (in BLOCK_REGISTRY)
        **kwargs: Configuration parameters for the block
    """

    def __init__(self, block_type: str, **kwargs: Any) -> None:
        """__init__.

        Args:
            block_type (str): Description.
        """
        super().__init__()
        import warnings

        warnings.warn(
            "BlockLayer is deprecated and inhibits JIT compilation. Use create_block() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._block = create_block(block_type, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Delegate to instantiated block strategy.

        forward method for BlockLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self._block(x, *args, **kwargs)


# Global registry: Maps string identifiers to block classes
BLOCK_REGISTRY: dict[str, type[IBlockStrategy]] = {}


def register_block(name: str) -> Any:
    """Decorator to register a block in the global registry.

    Args:
        name: String identifier for the block (must be unique)

    Returns:
        Decorator function

    Example:
        >>> @register_block("my_custom_block")
        ... class MyCustomBlock(IBlockStrategy):
        ...     def forward(self, x):
        ...         return x * 2
    """

    def decorator(
        cls: type[IBlockStrategy],
    ) -> type[IBlockStrategy]:
        """decorator.

        Returns:
            type[IBlockStrategy]: Description.
        """
        if name in BLOCK_REGISTRY:
            msg = (
                f"Block '{name}' already registered. Choose different name or unregister existing."
            )
            raise ValueError(msg)
        BLOCK_REGISTRY[name] = cls
        return cls

    return decorator


def unregister_block(name: str) -> None:
    """Remove a block from the registry.

    Args:
        name: String identifier of the block to remove

    Raises:
        ValueError: If the block is not found
    """
    if name not in BLOCK_REGISTRY:
        raise ValueError(f"Block '{name}' is not registered.")
    del BLOCK_REGISTRY[name]


def list_registered_blocks() -> list[str]:
    """Return list of all registered block names.

    Returns:
        Sorted list of registered block identifiers
    """
    return sorted(BLOCK_REGISTRY.keys())


# =========================================================================
# Test / Debugging Blocks
# =========================================================================
# The IdentityBlock is intentionally minimal — it exists as a test fixture
# and debugging tool for validating the block registry dispatcher without
# introducing any transformation.


@register_block("placeholder_identity")
class IdentityBlock(IBlockStrategy):
    """Identity block that returns input unchanged.

    Intended for testing and debugging the block registry dispatcher.
    Not used in production pipelines.
    """

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Pass through: return input unchanged."""
        return x


# =========================================================================
# Attention Strategies
# =========================================================================
# Migrated from attention.py, attention_modules.py, trellis_attention.py during Phase 3.2.


@register_block("self_attention")
class SelfAttentionStrategy(IBlockStrategy):
    """Self-attention mechanism for 2D feature maps.

    Migrated from SelfAttention2D in attention_modules.py.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        ff_multiplier: float = 1.0,
        bias: bool = True,
        norm_eps: float = 1e-5,
    ):
        """__init__.

        Args:
            channels (int): Description.
            num_heads (int): Description.
            dropout (float): Description.
            ff_multiplier (float): Description.
            bias (bool): Description.
            norm_eps (float): Description.
        """
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        hidden_dim = max(int(channels * ff_multiplier), channels)

        self.channels = channels
        self.num_heads = num_heads
        self.attn_norm = nn.LayerNorm(channels, eps=norm_eps)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.ff_norm = nn.LayerNorm(channels, eps=norm_eps)
        ff_layers = [
            nn.Linear(channels, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, channels, bias=bias),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        ]
        self.feed_forward = nn.Sequential(*ff_layers)

    @staticmethod
    def _flatten(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Flatten a feature map to sequence form."""
        batch, channels, height, width = x.shape
        seq = x.view(batch, channels, height * width).transpose(1, 2)
        return seq, height, width

    def _unflatten(self, seq: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Restore a flattened sequence back to 2D spatial layout."""
        batch = seq.size(0)
        return seq.transpose(1, 2).reshape(batch, self.channels, height, width)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        seq, height, width = self._flatten(x)
        attn_input = self.attn_norm(seq)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        seq = seq + self.attn_dropout(attn_output)

        ff_input = self.ff_norm(seq)
        seq = seq + self.feed_forward(ff_input)
        return self._unflatten(seq, height, width)


@register_block("cross_attention")
class CrossAttentionStrategy(IBlockStrategy):
    """Cross-attention mechanism for conditional generation.

    Migrated from CrossAttention in attention.py.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        context_channels: int | None = None,
    ):
        """__init__.

        Args:
            channels (int): Description.
            num_heads (int): Description.
            context_channels (int | None): Description.
        """
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels or channels
        self.mha = nn.MultiheadAttention(
            channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        # Project context to query dimension if needed
        if self.context_channels != channels:
            self.context_proj = nn.Linear(self.context_channels, channels)
        else:
            self.context_proj = None

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            context (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.
        """
        if context is None:
            raise ValueError("context tensor must be provided for cross attention")

        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).transpose(1, 2)
        context = context.view(
            -1,
            self.context_channels,
            context.shape[-2] * context.shape[-1],
        ).transpose(1, 2)

        # Project context to match query dimension if needed
        if self.context_proj is not None:
            context = self.context_proj(context)

        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, context, context)
        attention_value = attention_value + x
        attention_value = self.ff(attention_value) + attention_value
        return attention_value.transpose(-2, -1).view(
            -1,
            self.channels,
            size[0],
            size[1],
        )


@register_block("spatial_attention")
class SpatialAttentionStrategy(IBlockStrategy):
    """Spatial attention mechanism for feature maps.

    Migrated from SpatialAttention in attention.py.
    """

    def __init__(self, in_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        return x * attn


if TrellisMultiHeadAttention is not None:

    @register_block("trellis_attention")
    class TrellisAttentionStrategy(IBlockStrategy):
        """Trellis self-attention mechanism for 2D feature maps.

        Migrated from TrellisSelfAttention2D in
        implementations/blocks/attention.py.
        """

        def __init__(
            self,
            channels: int,
            num_heads: int,
            use_rope: bool = False,
            qk_rms_norm: bool = True,
            ff_multiplier: int = 4,
        ):
            """__init__.

            Args:
                channels (int): Description.
                num_heads (int): Description.
                use_rope (bool): Description.
                qk_rms_norm (bool): Description.
                ff_multiplier (int): Description.
            """
            super().__init__()
            self.channels = channels
            self.num_heads = num_heads
            self.norm = nn.LayerNorm(channels)
            self.attn = TrellisMultiHeadAttention(
                channels=channels,
                num_heads=num_heads,
                attn_type="self",
                use_rope=use_rope,
                qk_rms_norm=qk_rms_norm,
            )
            hidden_dim = channels * ff_multiplier
            self.ff = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, channels),
            )

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            """forward.

            Args:
                x (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            bsz, channels, height, width = x.shape
            seq = x.view(bsz, channels, height * width).transpose(1, 2)
            seq_norm = self.norm(seq)
            attn_out = self.attn(seq_norm)
            seq = seq + attn_out
            seq = seq + self.ff(seq)
            return seq.transpose(1, 2).view(bsz, channels, height, width)


@register_block("diffusion_attention")
class DiffusionAttentionStrategy(IBlockStrategy):
    """Self-attention block for diffusion models.

    Migrated from AttentionBlock in
    diffusion/diffusion_parts/unet_variants/attention.py.
    """

    def __init__(self, channels: int):
        """__init__.

        Args:
            channels (int): Description.
        """
        super().__init__()
        self.channels = channels
        # Ensure num_groups is a divisor of channels for stability
        num_groups = math.gcd(32, channels) if channels > 0 else 1
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Single-head SDPA over H*W tokens; default scale is C**-0.5.
        q = q.reshape(B, C, H * W).transpose(1, 2).unsqueeze(1)
        k = k.reshape(B, C, H * W).transpose(1, 2).unsqueeze(1)
        v = v.reshape(B, C, H * W).transpose(1, 2).unsqueeze(1)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.squeeze(1).transpose(1, 2).reshape(B, C, H, W)
        h = self.proj(h)

        return x + h


# =========================================================================
# TRELLIS Attention Strategies
# =========================================================================
# Additional TRELLIS attention mechanisms for advanced attention patterns


if MultiHeadAttention is not None:

    @register_block("trellis_cross_attention")
    class TrellisCrossAttentionStrategy(IBlockStrategy):
        """TRELLIS cross-attention mechanism for conditional generation.

        Uses the full TRELLIS MultiHeadAttention implementation with
        optional rotary embeddings and RMS normalization.
        """

        def __init__(
            self,
            channels: int,
            ctx_channels: int,
            num_heads: int = 8,
            use_rope: bool = False,
            qk_rms_norm: bool = False,
            qk_rms_norm_cross: bool = False,
        ):
            """__init__.

            Args:
                channels (int): Description.
                ctx_channels (int): Description.
                num_heads (int): Description.
                use_rope (bool): Description.
                qk_rms_norm (bool): Description.
                qk_rms_norm_cross (bool): Description.
            """
            super().__init__()
            self.channels = channels
            self.ctx_channels = ctx_channels
            self.attn = MultiHeadAttention(
                channels=channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                kind="cross",
                use_rope=use_rope,
                qk_rms_norm=qk_rms_norm,
            )

        def forward(self, x: torch.Tensor, context: torch.Tensor, **kwargs) -> torch.Tensor:
            """forward.

            Args:
                x (torch.Tensor): Description.
                context (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            return self.attn(x, context)

    @register_block("trellis_self_attention_rms")
    class TrellisSelfAttentionRMSStrategy(IBlockStrategy):
        """TRELLIS self-attention with RMS normalization for 2D feature maps.

        Uses per-head RMS normalization for improved stability.
        """

        def __init__(
            self,
            channels: int,
            num_heads: int,
            ff_multiplier: int = 4,
            use_rope: bool = False,
        ):
            """__init__.

            Args:
                channels (int): Description.
                num_heads (int): Description.
                ff_multiplier (int): Description.
                use_rope (bool): Description.
            """
            super().__init__()
            self.channels = channels
            self.num_heads = num_heads
            self.norm = nn.LayerNorm(channels)
            self.attn = MultiHeadAttention(
                channels=channels,
                num_heads=num_heads,
                kind="self",
                use_rope=use_rope,
                qk_rms_norm=True,
            )
            hidden_dim = channels * ff_multiplier
            self.ff = nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim, channels),
            )

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            """forward.

            Args:
                x (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            bsz, channels, height, width = x.shape
            seq = x.view(bsz, channels, height * width).transpose(1, 2)
            seq_norm = self.norm(seq)
            attn_out = self.attn(seq_norm)
            seq = seq + attn_out
            seq = seq + self.ff(seq)
            return seq.transpose(1, 2).view(bsz, channels, height, width)


# =========================================================================
# Convolutional Strategies
# =========================================================================
# To be migrated from convolutional.py during Phase 3.4.


# =========================================================================
# Transformer Strategies
# =========================================================================
# To be migrated from transformer implementations during Phase 3.5.


if TransformerBlock is not None:

    @register_block("trellis_transformer_block")
    class TrellisTransformerBlockStrategy(IBlockStrategy):
        """TRELLIS transformer block with residual connections.

        Migrated from TransformerBlock in trellis_transformer.py.
        """

        def __init__(
            self,
            channels: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
        ):
            """__init__.

            Args:
                channels (int): Description.
                num_heads (int): Description.
                mlp_ratio (float): Description.
                qkv_bias (bool): Description.
            """
            super().__init__()
            self.block = TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
            )

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            """forward.

            Args:
                x (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            return self.block(x)


if AbsolutePositionEmbedder is not None:

    @register_block("trellis_absolute_position_embedder")
    class TrellisAbsolutePositionEmbedderStrategy(IBlockStrategy):
        """TRELLIS absolute position embedding for 3D coordinates.

        This is primarily used as a component in larger models rather than
        a standalone block, but registered for consistency.
        """

        def __init__(self, channels: int, in_channels: int = 3):
            """__init__.

            Args:
                channels (int): Description.
                in_channels (int): Description.
            """
            super().__init__()
            self.embedder = AbsolutePositionEmbedder(channels=channels, in_channels=in_channels)

        def forward(self, positions: torch.Tensor, **kwargs) -> torch.Tensor:
            """forward.

            Args:
                positions (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            return self.embedder(positions)


# =========================================================================
# U-Net Strategies
# =========================================================================
# Migrated from unet_parts.py, unet_blocks.py, blocks/unet.py during Phase 3.3.


@register_block("unet_double_conv")
class UNetDoubleConvStrategy(IBlockStrategy):
    """Double convolution block for U-Net.

    Migrated from DoubleConv in convolutional.py.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            mid_channels (int | None): Description.
        """
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.double_conv(x)


@register_block("unet_complex_double_conv")
class UNetComplexDoubleConvStrategy(IBlockStrategy):
    """Complex Double convolution block for U-Net."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            mid_channels (int | None): Description.
        """
        super().__init__()
        from .convolutional import ComplexDoubleConv

        self.double_conv = ComplexDoubleConv(in_channels, out_channels, mid_channels)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.double_conv(x)


@register_block("unet_adagn_double_conv")
class UNetAdaGNDoubleConvStrategy(IBlockStrategy):
    """Double convolution block with AdaGN for U-Net.
    Requires 'emb' kwarg in forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        time_emb_dim: int = 128,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            mid_channels (int | None): Description.
            time_emb_dim (int): Description.
        """
        super().__init__()
        from .convolutional import AdaGNDoubleConv

        self.double_conv = AdaGNDoubleConv(
            in_channels,
            out_channels,
            emb_channels=time_emb_dim,
            mid_channels=mid_channels,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if emb is None:
            raise ValueError("AdaGN strategy requires 'emb' argument")
        return self.double_conv(x, emb)


@register_block("unet_complex_adagn_double_conv")
class UNetComplexAdaGNDoubleConvStrategy(IBlockStrategy):
    """Complex Double convolution block with AdaGN for U-Net.
    Requires 'emb' kwarg in forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        time_emb_dim: int = 128,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            mid_channels (int | None): Description.
            time_emb_dim (int): Description.
        """
        super().__init__()
        from .convolutional import ComplexAdaGNDoubleConv

        self.double_conv = ComplexAdaGNDoubleConv(
            in_channels,
            out_channels,
            emb_channels=time_emb_dim,
            mid_channels=mid_channels,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if emb is None:
            raise ValueError("AdaGN strategy requires 'emb' argument")
        return self.double_conv(x, emb)


@register_block("unet_down")
class UNetDownStrategy(IBlockStrategy):
    """Downsampling block for U-Net.

    Migrated from Down in blocks/unet.py.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.maxpool_conv(x)


@register_block("unet_complex_down")
class UNetComplexDownStrategy(IBlockStrategy):
    """Complex Downsampling block for U-Net.
    Uses AvgPool2d to preserve phase linearity better than MaxPool.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()
        from .convolutional import ComplexDoubleConv

        # AvgPool is linear, safe for Re/Im separation
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            ComplexDoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.maxpool_conv(x)


@register_block("unet_adagn_down")
class UNetAdaGNDownStrategy(IBlockStrategy):
    """Downsampling block with AdaGN for U-Net.
    Requires 'emb' kwarg.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = 128):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_emb_dim (int): Description.
        """
        super().__init__()
        from .convolutional import AdaGNDoubleConv

        self.pool = nn.MaxPool2d(2)
        self.double_conv = AdaGNDoubleConv(
            in_channels,
            out_channels,
            emb_channels=time_emb_dim,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if emb is None:
            raise ValueError("AdaGN strategy requires 'emb' argument")
        x = self.pool(x)
        return self.double_conv(x, emb)


@register_block("unet_complex_adagn_down")
class UNetComplexAdaGNDownStrategy(IBlockStrategy):
    """Complex Downsampling block with AdaGN for U-Net.
    Requires 'emb' kwarg.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int = 128):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            time_emb_dim (int): Description.
        """
        super().__init__()
        from .convolutional import ComplexAdaGNDoubleConv

        self.pool = nn.AvgPool2d(2)
        self.double_conv = ComplexAdaGNDoubleConv(
            in_channels,
            out_channels,
            emb_channels=time_emb_dim,
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if emb is None:
            raise ValueError("AdaGN strategy requires 'emb' argument")
        x = self.pool(x)
        return self.double_conv(x, emb)


@register_block("unet_complex_up")
class UNetComplexUpStrategy(IBlockStrategy):
    """Complex Upsampling block for U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        mid_channels: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bilinear (bool): Description.
            mid_channels (int | None): Description.
        """
        super().__init__()
        from .convolutional import ComplexDoubleConv

        self.out_channels = out_channels
        self.bilinear = bilinear

        # Always use bilinear upsampling for complex to avoid
        # needing ComplexConvTranspose2d for now.
        # Bilinear is linear, so safe for Re/Im.
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Channels are preserved during upsample
        up_out_c = in_channels

        # Upsample does not reduce channels (unlike ConvTranspose2d), so we have
        # in_channels (upsampled) + in_channels // 2 (skip connection)
        concat_c = in_channels + in_channels // 2

        # If mid_channels is not provided, use out_channels (or concat_c // 2)
        if mid_channels is None:
            mid_channels = concat_c // 2

        self.conv = ComplexDoubleConv(concat_c, out_channels, mid_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x1 (torch.Tensor): Description.
            x2 (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


@register_block("unet_up")
class UNetUpStrategy(IBlockStrategy):
    """Upsampling block for U-Net.

    Migrated from Up in blocks/unet.py.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        mid_channels: int | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bilinear (bool): Description.
            mid_channels (int | None): Description.
        """
        super().__init__()
        self.out_channels = out_channels
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # When using bilinear upsampling, we don't change channels during upsampling
            # The channel reduction happens in the DoubleConv
            up_out_c = in_channels
        else:
            # Transposed convolution halves the channels
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_out_c = in_channels // 2

        # Calculate input channels for the DoubleConv
        # It receives the upsampled input + the skip connection input
        # We assume skip connection has same channels as upsampled input (standard U-Net)
        # OR we need to know the skip channels.
        # Standard U-Net: skip channels = up_out_c
        # So total input to DoubleConv is up_out_c * 2
        concat_c = up_out_c * 2

        # If mid_channels is not provided, use out_channels (or concat_c // 2)
        if mid_channels is None:
            mid_channels = concat_c // 2

        self.conv = DoubleConv(concat_c, out_channels, mid_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x1 (torch.Tensor): Description.
            x2 (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = torch.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


@register_block("unet_adagn_up")
class UNetAdaGNUpStrategy(IBlockStrategy):
    """Upsampling block with AdaGN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        mid_channels: int | None = None,
        time_emb_dim: int = 128,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bilinear (bool): Description.
            mid_channels (int | None): Description.
            time_emb_dim (int): Description.
        """
        super().__init__()
        from .convolutional import AdaGNDoubleConv

        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            up_out_c = in_channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            up_out_c = in_channels // 2

        concat_c = up_out_c * 2
        if mid_channels is None:
            mid_channels = concat_c // 2

        self.conv = AdaGNDoubleConv(
            concat_c,
            out_channels,
            emb_channels=time_emb_dim,
            mid_channels=mid_channels,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, emb: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """forward.

        Args:
            x1 (torch.Tensor): Description.
            x2 (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if emb is None:
            raise ValueError("AdaGN strategy requires 'emb' argument")
        x1 = self.up(x1)
        # Pad if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, emb)


@register_block("unet_complex_adagn_up")
class UNetComplexAdaGNUpStrategy(IBlockStrategy):
    """Complex Upsampling block with AdaGN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        mid_channels: int | None = None,
        time_emb_dim: int = 128,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bilinear (bool): Description.
            mid_channels (int | None): Description.
            time_emb_dim (int): Description.
        """
        super().__init__()
        from .convolutional import ComplexAdaGNDoubleConv

        self.bilinear = bilinear
        # Always use bilinear for complex upsampling for now
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Channels preserved
        concat_c = in_channels + in_channels // 2

        if mid_channels is None:
            mid_channels = concat_c // 2

        self.conv = ComplexAdaGNDoubleConv(
            concat_c,
            out_channels,
            emb_channels=time_emb_dim,
            mid_channels=mid_channels,
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, emb: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
        """forward.

        Args:
            x1 (torch.Tensor): Description.
            x2 (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if emb is None:
            raise ValueError("AdaGN strategy requires 'emb' argument")
        x1 = self.up(x1)
        # Pad if necessary
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, emb)


@register_block("unet_out_conv")
class UNetOutConvStrategy(IBlockStrategy):
    """Output convolution block for U-Net.

    Migrated from OutConv in blocks/unet.py.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: nn.Module | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            activation (nn.Module | None): Description.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation or nn.Identity()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.activation(self.conv(x))


@register_block("layer_norm")
class LayerNormStrategy(IBlockStrategy):
    """Layer normalization with configurable pre/post strategy."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
        strategy: str = "pre",
    ):
        """__init__.

        Args:
            channels (int): Description.
            eps (float): Description.
            strategy (str): Description.
        """
        super().__init__()
        self.strategy = strategy
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor, layer_fn: Callable | None = None, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            layer_fn (Callable | None): Description.
        Returns:
            torch.Tensor: Description.
        """
        if layer_fn is None:
            return self.norm(x)

        if self.strategy == "pre":
            return x + layer_fn(self.norm(x))
        if self.strategy == "post":
            return self.norm(x + layer_fn(x))
        raise ValueError(f"Unknown normalization strategy: {self.strategy}")


@register_block("residual_block")
class ResidualBlockStrategy(IBlockStrategy):
    """Standard residual block with GroupNorm and LeakyReLU.

    Migrated from ResidualBlock in blocks/residual.py.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """__init__.

        Args:
            channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
        """
        super().__init__()
        self.block = ResidualBlock(
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.block(x)


@register_block("preact_residual_block")
class PreActResidualBlockStrategy(IBlockStrategy):
    """Pre-activation residual block with GroupNorm and LeakyReLU.

    Migrated from PreActResidualBlock in blocks/residual.py.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """__init__.

        Args:
            channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
        """
        super().__init__()
        self.block = PreActResidualBlock(
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.block(x)


@register_block("unified_residual_block")
class UnifiedResidualBlockStrategy(IBlockStrategy):
    """Unified residual block with configurable options.

    Migrated from UnifiedResidualBlock in blocks/residual.py.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        activation: str = "leaky",
        skip_scale: float = 1.0,
        use_shortcut: bool = True,
        pre_activation: bool = False,
    ):
        """__init__.

        Args:
            channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            bias (bool): Description.
            activation (str): Description.
            skip_scale (float): Description.
            use_shortcut (bool): Description.
            pre_activation (bool): Description.
        """
        super().__init__()
        self.block = UnifiedResidualBlock(
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            activation=activation,
            skip_scale=skip_scale,
            use_shortcut=use_shortcut,
            pre_activation=pre_activation,
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.block(x)


# =========================================================================
# Swin Transformer Blocks
# =========================================================================
# Canonical WindowAttention and SwinBlock - use these instead of duplicates
# in other files (gans/swin_kan_gan.py, spectral/fourier_kan_swin.py, etc.)

from .swin import SwinBlock, WindowAttention, window_partition, window_reverse


@register_block("window_attention")
class WindowAttentionStrategy(IBlockStrategy):
    """Window-based Multi-head Self Attention (W-MSA) with relative position bias.

    Canonical implementation from blocks/swin.py.
    Use this instead of duplicated WindowAttention classes in other files.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_scale: float | None = None,
    ):
        """__init__.

        Args:
            dim (int): Description.
            window_size (int): Description.
            num_heads (int): Description.
            qkv_bias (bool): Description.
            attn_drop (float): Description.
            proj_drop (float): Description.
            qk_scale (float | None): Description.
        """
        super().__init__()
        self.block = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qk_scale=qk_scale,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            mask (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.block(x, mask=mask)


@register_block("swin_block")
class SwinBlockStrategy(IBlockStrategy):
    """Swin Transformer Block with shifted window attention.

    Canonical implementation from blocks/swin.py.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            window_size (int): Description.
            shift_size (int): Description.
            mlp_ratio (float): Description.
            qkv_bias (bool): Description.
            dropout (float): Description.
            attn_dropout (float): Description.
            drop_path (float): Description.
        """
        super().__init__()
        self.block = SwinBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=dropout,
            attn_dropout=attn_dropout,
            drop_path=drop_path,
        )

    def forward(
        self, x: torch.Tensor, input_resolution: tuple[int, int] | None = None, **kwargs
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            input_resolution (tuple[int, int] | None): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.block(x, input_resolution=input_resolution)


# =========================================================================
# Register Fusion Blocks
# =========================================================================
# Register fusion blocks imported from fusion.py
BLOCK_REGISTRY["attention_fusion"] = AttentionFusionBlock
BLOCK_REGISTRY["feature_fusion"] = FeatureFusionBlock
BLOCK_REGISTRY["multiscale_fusion"] = MultiScaleFusionBlock
BLOCK_REGISTRY["adaptive_fusion"] = AdaptiveFusionBlock
BLOCK_REGISTRY["residual_fusion"] = ResidualFusionBlock
BLOCK_REGISTRY["multimodal_fusion"] = MultiModalFusionBlock


# =========================================================================
# Register previously-orphaned attention blocks
# =========================================================================
# These attention modules were defined and exported (each lives in its own
# canonical home under blocks/) but were never constructed by any registered
# model/strategy/block — the 2026-06-24 attention-wiring audit. Wiring them
# into BLOCK_REGISTRY makes them config-reachable via create_block(), and they
# inherit the resolver's raise-on-unknown contract (CLAUDE.md pitfall #9).
# The registry contract is *construction*: create_block calls block_class(
# **kwargs); each block's forward signature is the caller's concern (these are
# not all (B, C, H, W) -> (B, C, H, W) maps). Imports live at the top of the
# module (no cycle); the registrations are here next to the fusion blocks.
BLOCK_REGISTRY["graph_attention"] = GraphAttentionBlock
BLOCK_REGISTRY["local_graph_attention"] = LocalGraphAttentionBlock
BLOCK_REGISTRY["multi_contrast_kspace_cross_attention"] = MultiContrastKSpaceCrossAttention
BLOCK_REGISTRY["kspace_radial_swin"] = KSpaceRadialSwinBlock
BLOCK_REGISTRY["radial_window_attention"] = RadialWindowAttention
BLOCK_REGISTRY["spatiotemporal_marker_attention"] = SpatiotemporalMarkerAttention

# 2026-06-27 follow-up: standalone attention mechanisms that were live only as
# hard-coded sub-components of a generator-used parent block (so not *dead*, but
# not individually config-selectable either). StructureTensorAttention is built
# inside StructureTensorTransformerBlock; RadialBandAttention / CrossDomainAttention
# inside KANGatedDualDomainAttention; MultiScaleFreqBandAttention inside
# WaveletFreqAttentionBlock. Registering them exposes each as a composable unit.
# Their forward contracts are non-trivial (complex / two-input / structure-tensor
# side input) — construction is the registry contract; forward is the caller's.
BLOCK_REGISTRY["structure_tensor_attention"] = StructureTensorAttention
BLOCK_REGISTRY["radial_band_attention"] = RadialBandAttention
BLOCK_REGISTRY["multiscale_freq_band_attention"] = MultiScaleFreqBandAttention
BLOCK_REGISTRY["cross_domain_attention"] = CrossDomainAttention

# TrellisSelfAttention2D is a legacy 2D self-attention variant built on the
# optional trellis stack; register it only when that optional import succeeded
# (mirrors the conditional trellis_* strategy registrations above).
if TrellisSelfAttention2D is not None:
    BLOCK_REGISTRY["trellis_self_attention_2d"] = TrellisSelfAttention2D


__all__ = [
    "IBlockStrategy",
    "BlockLayer",
    "create_block",
    "BLOCK_REGISTRY",
    "register_block",
    "unregister_block",
    "list_registered_blocks",
    # Swin blocks (for direct import)
    "WindowAttention",
    "SwinBlock",
    "window_partition",
    "window_reverse",
]
