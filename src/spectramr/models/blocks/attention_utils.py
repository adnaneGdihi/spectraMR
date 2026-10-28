"""FlashAttention/xFormers Optional Integration Wrapper

This module provides optional integration with FlashAttention and xFormers
libraries for improved attention computation efficiency, with graceful
fallbacks when the libraries are not available.
"""

import logging
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class AttentionBackend:
    """Enumeration of available attention backends"""

    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION = "flash_attention"
    MEMORY_EFFICIENT = "memory_efficient"


class AttentionConfig:
    """Configuration for attention computation"""

    def __init__(
        self,
        backend: str = AttentionBackend.PYTORCH,
        use_flash_attention: bool = False,
        use_xformers: bool = False,
        use_memory_efficient: bool = True,
        flash_attention_version: str = "auto",
        xformers_version: str = "auto",
        enable_math_sdp: bool = True,
        enable_flash_sdp: bool = True,
        enable_mem_efficient_sdp: bool = True,
        dropout: float = 0.0,
        causal: bool = False,
        scale: float | None = None,
    ):
        """__init__.

        Args:
            backend (str): Description.
            use_flash_attention (bool): Description.
            use_xformers (bool): Description.
            use_memory_efficient (bool): Description.
            flash_attention_version (str): Description.
            xformers_version (str): Description.
            enable_math_sdp (bool): Description.
            enable_flash_sdp (bool): Description.
            enable_mem_efficient_sdp (bool): Description.
            dropout (float): Description.
            causal (bool): Description.
            scale (Optional[float]): Description.
        """
        self.backend = backend
        self.use_flash_attention = use_flash_attention
        self.use_xformers = use_xformers
        self.use_memory_efficient = use_memory_efficient
        self.flash_attention_version = flash_attention_version
        self.xformers_version = xformers_version
        self.enable_math_sdp = enable_math_sdp
        self.enable_flash_sdp = enable_flash_sdp
        self.enable_mem_efficient_sdp = enable_mem_efficient_sdp
        self.dropout = dropout
        self.causal = causal
        self.scale = scale


class AttentionWrapper:
    """Wrapper for attention computation with optional
    FlashAttention/xFormers support.

    This class provides a unified interface for attention computation that
    automatically selects the best available backend and falls back gracefully.
    """

    def __init__(self, config: AttentionConfig | None = None):
        """__init__.

        Args:
            config (Optional[AttentionConfig]): Description.
        """
        self.config = config or AttentionConfig()
        self._backends_available = self._check_available_backends()
        self._selected_backend = self._select_best_backend()
        self.training = True  # Default to training mode

    def _check_available_backends(self) -> dict[str, bool]:
        """Check which attention backends are available"""
        backends = {
            AttentionBackend.PYTORCH: True,  # Always available
            AttentionBackend.MEMORY_EFFICIENT: True,  # PyTorch built-in
        }

        # Check for xFormers
        try:
            import xformers

            backends[AttentionBackend.XFORMERS] = True
            logger.info(f"xFormers available (version: {xformers.__version__})")
        except ImportError:
            backends[AttentionBackend.XFORMERS] = False
            logger.debug("xFormers not available")

        # Check for FlashAttention
        try:
            import flash_attn

            backends[AttentionBackend.FLASH_ATTENTION] = True
            logger.info(f"FlashAttention available (version: {flash_attn.__version__})")
        except ImportError:
            backends[AttentionBackend.FLASH_ATTENTION] = False
            logger.debug("FlashAttention not available")

        return backends

    def _select_best_backend(self) -> str:
        """Select the best available backend based on configuration
        and availability
        """
        # Priority order: configured preference > availability > performance

        # Validate backend exists in available backends
        if self.config.backend not in self._backends_available:
            logger.warning(
                f"Invalid backend '{self.config.backend}', "
                f"falling back to {AttentionBackend.PYTORCH}",
            )
            return AttentionBackend.PYTORCH

        if self.config.backend != AttentionBackend.PYTORCH:
            if self._backends_available[self.config.backend]:
                return self.config.backend

        # Fallback to best available
        if (
            self._backends_available[AttentionBackend.FLASH_ATTENTION]
            and self.config.use_flash_attention
        ):
            return AttentionBackend.FLASH_ATTENTION
        if self._backends_available[AttentionBackend.XFORMERS] and self.config.use_xformers:
            return AttentionBackend.XFORMERS
        if (
            self._backends_available[AttentionBackend.MEMORY_EFFICIENT]
            and self.config.use_memory_efficient
        ):
            return AttentionBackend.MEMORY_EFFICIENT
        return AttentionBackend.PYTORCH

    def compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute attention using the selected backend.

        Args:
            query: Query tensor [batch, heads, seq_len, head_dim]
            key: Key tensor [batch, heads, seq_len, head_dim]
            value: Value tensor [batch, heads, seq_len, head_dim]
            attn_mask: Attention mask [batch, 1, seq_len, seq_len]
            **kwargs: Additional backend-specific arguments

        Returns:
            Attention output tensor

        """
        # Validate input tensor shapes
        if query.dim() != 4:
            raise RuntimeError(
                f"Query tensor must have 4 dimensions, got {query.dim()}",
            )
        if key.dim() != 4:
            raise RuntimeError(f"Key tensor must have 4 dimensions, got {key.dim()}")
        if value.dim() != 4:
            raise RuntimeError(
                f"Value tensor must have 4 dimensions, got {value.dim()}",
            )

        # Check that all tensors have the same batch size and number of heads
        if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
            raise RuntimeError(
                f"Batch size mismatch: query {query.shape[0]}, "
                f"key {key.shape[0]}, value {value.shape[0]}",
            )
        if query.shape[1] != key.shape[1] or query.shape[1] != value.shape[1]:
            raise RuntimeError(
                f"Number of heads mismatch: query {query.shape[1]}, "
                f"key {key.shape[1]}, value {value.shape[1]}",
            )

        # Check sequence length consistency
        if query.shape[2] != value.shape[2]:
            raise RuntimeError(
                f"Sequence length mismatch: query {query.shape[2]}, value {value.shape[2]}",
            )
        if key.shape[2] != query.shape[2] and key.shape[2] != value.shape[2]:
            raise RuntimeError(
                f"Key sequence length {key.shape[2]} doesn't match "
                f"query {query.shape[2]} or value {value.shape[2]}",
            )

        # Check head dimension consistency
        if query.shape[3] != key.shape[3] or query.shape[3] != value.shape[3]:
            raise RuntimeError(
                f"Head dimension mismatch: query {query.shape[3]}, "
                f"key {key.shape[3]}, value {value.shape[3]}",
            )
        if self._selected_backend == AttentionBackend.FLASH_ATTENTION:
            try:
                return self._compute_flash_attention(
                    query,
                    key,
                    value,
                    attn_mask,
                    **kwargs,
                )
            except ImportError:
                logger.warning(
                    "FlashAttention not available during computation, "
                    "falling back to PyTorch attention",
                )
                return self._compute_pytorch_attention(
                    query,
                    key,
                    value,
                    attn_mask,
                    **kwargs,
                )
        elif self._selected_backend == AttentionBackend.XFORMERS:
            try:
                return self._compute_xformers_attention(
                    query,
                    key,
                    value,
                    attn_mask,
                    **kwargs,
                )
            except ImportError:
                logger.warning(
                    "xFormers not available during computation, falling back to PyTorch attention",
                )
                return self._compute_pytorch_attention(
                    query,
                    key,
                    value,
                    attn_mask,
                    **kwargs,
                )
        elif self._selected_backend == AttentionBackend.MEMORY_EFFICIENT:
            return self._compute_memory_efficient_attention(
                query,
                key,
                value,
                attn_mask,
                **kwargs,
            )
        else:
            return self._compute_pytorch_attention(
                query,
                key,
                value,
                attn_mask,
                **kwargs,
            )

    def _compute_pytorch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute attention using standard PyTorch"""
        scale = self.config.scale or (query.size(-1) ** -0.5)

        if attn_mask is not None:
            # The mask is additive (it was added to the scores); SDPA would read a bool as keep/drop.
            attn_mask = attn_mask.to(query.dtype)
            # Ensure mask is broadcastable to attention scores shape
            if attn_mask.dim() == 4:
                # Handle [batch, 1, seq_len, seq_len] or
                # [batch, heads, seq_len, seq_len]
                if attn_mask.shape[1] == 1:
                    # Broadcast single-head mask to all heads
                    attn_mask = attn_mask.expand(-1, query.shape[1], -1, -1)
                elif attn_mask.shape[1] == query.shape[1]:
                    # Already correct shape
                    pass
                else:
                    raise ValueError(
                        f"Mask head dimension {attn_mask.shape[1]} "
                        f"doesn't match query heads {query.shape[1]}",
                    )
            elif attn_mask.dim() == 3:
                # Broadcast [batch, seq_len, seq_len] to
                # [batch, heads, seq_len, seq_len]
                attn_mask = attn_mask.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
            elif attn_mask.dim() == 2:
                # Broadcast [seq_len, seq_len] to
                # [batch, heads, seq_len, seq_len]
                attn_mask = (
                    attn_mask.unsqueeze(0)
                    .unsqueeze(1)
                    .expand(query.shape[0], query.shape[1], -1, -1)
                )

        is_causal = False
        if self.config.causal:
            if attn_mask is None:
                is_causal = True
            else:
                # SDPA rejects ``is_causal`` together with a mask: fold it in.
                causal = torch.ones(
                    query.shape[-2], key.shape[-2], dtype=torch.bool, device=query.device
                ).triu(diagonal=1)
                attn_mask = attn_mask.masked_fill(causal, float("-inf"))

        dropout_p = self.config.dropout if self.training else 0.0
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

        return output

    def _compute_memory_efficient_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute attention using PyTorch's memory efficient SDP"""
        # Use PyTorch's scaled_dot_product_attention
        # with memory efficient backend
        scale = self.config.scale or None

        # Convert mask format if needed
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                # Handle [batch, 1, seq_len, seq_len] or
                # [batch, heads, seq_len, seq_len]
                if attn_mask.shape[1] == 1:
                    # Broadcast single-head mask to all heads
                    attn_mask = attn_mask.expand(-1, query.shape[1], -1, -1)
                elif attn_mask.shape[1] == query.shape[1]:
                    # Already correct shape
                    pass
                else:
                    raise ValueError(
                        f"Mask head dimension {attn_mask.shape[1]} "
                        f"doesn't match query heads {query.shape[1]}",
                    )
            elif attn_mask.dim() == 3:
                # Broadcast [batch, seq_len, seq_len] to
                # [batch, heads, seq_len, seq_len]
                attn_mask = attn_mask.unsqueeze(1).expand(-1, query.shape[1], -1, -1)
            elif attn_mask.dim() == 2:
                # Broadcast [seq_len, seq_len] to
                # [batch, heads, seq_len, seq_len]
                attn_mask = (
                    attn_mask.unsqueeze(0)
                    .unsqueeze(1)
                    .expand(query.shape[0], query.shape[1], -1, -1)
                )

        # Build list of enabled backends
        enabled_backends = []
        if self.config.enable_math_sdp:
            enabled_backends.append(torch.nn.attention.SDPBackend.MATH)
        if self.config.enable_flash_sdp:
            enabled_backends.append(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
        if self.config.enable_mem_efficient_sdp:
            enabled_backends.append(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)

        with torch.nn.attention.sdpa_kernel(backends=enabled_backends):
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=self.config.causal,
                scale=scale,
            )

        return output

    def _compute_xformers_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute attention using xFormers"""
        try:
            import xformers.ops as xops

            # xFormers expects different tensor shapes
            # [batch, seq_len, heads, head_dim] ->
            # [batch, heads, seq_len, head_dim]
            if query.dim() == 4:  # [batch, heads, seq_len, head_dim]
                query = query.transpose(1, 2)
                # [batch, seq_len, heads, head_dim]
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

            output = xops.memory_efficient_attention(
                query,
                key,
                value,
                attn_bias=attn_mask,
                p=self.config.dropout if self.training else 0.0,
                scale=self.config.scale,
            )

            # Transpose back to expected shape
            output = output.transpose(1, 2)
            # [batch, heads, seq_len, head_dim]

            return output

        except ImportError:
            # Let the caller handle the ImportError for fallback
            raise
        except Exception as e:
            logger.warning(f"xFormers attention failed: {e}, falling back to PyTorch")
            return self._compute_pytorch_attention(
                query,
                key,
                value,
                attn_mask,
                **kwargs,
            )

    def _compute_flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute attention using FlashAttention"""
        try:
            from flash_attn import flash_attn_func

            # FlashAttention expects [batch, seq_len, heads, head_dim]
            if query.dim() == 4:  # [batch, heads, seq_len, head_dim]
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

            # FlashAttention doesn't support all mask types
            if attn_mask is not None:
                logger.warning(
                    "FlashAttention doesn't support attention masks, ignoring mask",
                )

            output = flash_attn_func(
                query,
                key,
                value,
                dropout_p=self.config.dropout if self.training else 0.0,
                causal=self.config.causal,
                softmax_scale=self.config.scale,
            )

            # Transpose back to expected shape
            output = output.transpose(1, 2)

            return output

        except ImportError:
            # Let the caller handle the ImportError for fallback
            raise
        except Exception as e:
            logger.warning(f"FlashAttention failed: {e}, falling back to PyTorch")
            return self._compute_pytorch_attention(
                query,
                key,
                value,
                attn_mask,
                **kwargs,
            )

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about available and selected backends"""
        return {
            "selected_backend": self._selected_backend,
            "available_backends": self._backends_available,
            "config": {
                "backend": self.config.backend,
                "use_flash_attention": self.config.use_flash_attention,
                "use_xformers": self.config.use_xformers,
                "use_memory_efficient": self.config.use_memory_efficient,
            },
        }


class EfficientAttentionLayer(nn.Module):
    """Attention layer with optional FlashAttention/xFormers support.

    This layer automatically selects the best available attention backend
    and provides a drop-in replacement for standard attention layers.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        attention_config: AttentionConfig | None = None,
    ):
        """__init__.

        Args:
            embed_dim (int): Description.
            num_heads (int): Description.
            dropout (float): Description.
            bias (bool): Description.
            attention_config (Optional[AttentionConfig]): Description.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Attention weights
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Attention wrapper
        self.attention = AttentionWrapper(attention_config or AttentionConfig())

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with efficient attention.

        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            attn_mask: Attention mask
            **kwargs: Additional arguments

        Returns:
            Output tensor [batch, seq_len, embed_dim]

        forward method for EfficientAttentionLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            attn_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch_size, seq_len, _ = x.shape

        # Project to queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, seq_len, heads, head_dim]
        # -> [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        attn_output = self.attention.compute_attention(q, k, v, attn_mask, **kwargs)

        # Transpose back: [batch, heads, seq_len, head_dim]
        # -> [batch, seq_len, heads, head_dim]
        attn_output = attn_output.transpose(1, 2)

        # Reshape and project
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_output)

        return output


# Global attention wrapper instance
_attention_wrapper = None


def get_attention_wrapper(config: AttentionConfig | None = None) -> AttentionWrapper:
    """Get or create global attention wrapper instance"""
    global _attention_wrapper
    if _attention_wrapper is None or config is not None:
        _attention_wrapper = AttentionWrapper(config)
    return _attention_wrapper


def configure_attention_backend(
    backend: str = AttentionBackend.PYTORCH,
    use_flash_attention: bool = False,
    use_xformers: bool = False,
    **kwargs,
) -> AttentionWrapper:
    """Configure the attention backend with specified parameters.

    Args:
        backend: Preferred backend
        use_flash_attention: Whether to use FlashAttention if available
        use_xformers: Whether to use xFormers if available
        **kwargs: Additional configuration parameters

    Returns:
        Configured AttentionWrapper instance

    """
    config = AttentionConfig(
        backend=backend,
        use_flash_attention=use_flash_attention,
        use_xformers=use_xformers,
        **kwargs,
    )
    return get_attention_wrapper(config)


@contextmanager
def attention_backend_context(backend: str, **config_kwargs):
    """Context manager for temporary attention backend configuration.

    Args:
        backend: Backend to use in this context
        **config_kwargs: Additional configuration parameters

    """
    original_wrapper = get_attention_wrapper()
    temp_config = AttentionConfig(backend=backend, **config_kwargs)
    temp_wrapper = AttentionWrapper(temp_config)

    global _attention_wrapper
    _attention_wrapper = temp_wrapper

    try:
        yield
    finally:
        _attention_wrapper = original_wrapper


def create_efficient_attention_layer(
    embed_dim: int,
    num_heads: int,
    backend: str = AttentionBackend.PYTORCH,
    **kwargs,
) -> EfficientAttentionLayer:
    """Create an efficient attention layer with the specified backend.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        backend: Attention backend to use
        **kwargs: Additional arguments

    Returns:
        EfficientAttentionLayer instance

    """
    config = AttentionConfig(backend=backend, **kwargs)
    return EfficientAttentionLayer(
        embed_dim,
        num_heads,
        attention_config=config,
        **kwargs,
    )


if __name__ == "__main__":
    # Example usage
    logger.debug("Testing FlashAttention/xFormers Integration...")

    # Test basic functionality
    wrapper = get_attention_wrapper()
    logger.debug("Available backends: %s", wrapper.get_backend_info())

    # Test attention computation
    batch_size, heads, seq_len, head_dim = 2, 8, 32, 64

    query = torch.randn(batch_size, heads, seq_len, head_dim)
    key = torch.randn(batch_size, heads, seq_len, head_dim)
    value = torch.randn(batch_size, heads, seq_len, head_dim)

    output = wrapper.compute_attention(query, key, value)
    logger.debug(f"Attention output shape: {output.shape}")

    # Test efficient attention layer
    embed_dim = heads * head_dim
    layer = EfficientAttentionLayer(embed_dim, heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    layer_output = layer(x)
    logger.debug(f"Layer output shape: {layer_output.shape}")

    logger.debug("FlashAttention/xFormers integration test complete!")
