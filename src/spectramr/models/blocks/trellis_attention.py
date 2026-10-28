"""TRELLIS attention building blocks.

Migrated from src/implementations/blocks/attention.py during consolidation.
"""

from __future__ import annotations


import torch
from torch import nn
from torch.nn import functional as F

try:  # pragma: no cover - PyTorch >= 2.0 provides SDPA
    from torch.nn.functional import scaled_dot_product_attention as torch_sdpa
except ImportError:  # pragma: no cover - fallback for older PyTorch

    def torch_sdpa(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """torch_sdpa.

        Args:
            q (torch.Tensor): Description.
            k (torch.Tensor): Description.
            v (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return F.scaled_dot_product_attention(q, k, v)


class MultiHeadRMSNorm(nn.Module):
    """Per-head RMS normalisation used by TRELLIS attention."""

    def __init__(self, dim: int, heads: int) -> None:
        """__init__.

        Args:
            dim (int): Description.
            heads (int): Description.
        """
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for MultiHeadRMSNorm.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        normed = F.normalize(x.float(), dim=-1, eps=1e-6)
        gamma = self.gamma.unsqueeze(0).unsqueeze(0)
        return (normed * gamma * self.scale).to(dtype=x.dtype)


class RotaryPositionEmbedder(nn.Module):
    """Rotary position embedding for TRELLIS attention."""

    def __init__(self, hidden_size: int, in_channels: int = 3) -> None:
        """__init__.

        Args:
            hidden_size (int): Description.
            in_channels (int): Description.
        """
        super().__init__()
        if hidden_size % 2 != 0:
            raise ValueError("hidden_size must be divisible by 2")
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        freq_dim = hidden_size // in_channels // 2
        freqs = torch.arange(freq_dim, dtype=torch.float32) / float(freq_dim)
        self.register_buffer("freqs", 1.0 / (10000**freqs), persistent=False)

    def _rotary_embedding(
        self,
        x: torch.Tensor,
        phases: torch.Tensor,
    ) -> torch.Tensor:
        """_rotary_embedding.

        Args:
            x (torch.Tensor): Description.
            phases (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(reshaped)
        x_rotated = x_complex * phases
        real = torch.view_as_real(x_rotated).reshape(*x.shape[:-1], -1)
        return real.to(x.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            q (torch.Tensor): Description.
            k (torch.Tensor): Description.
            indices (torch.Tensor | None): Description.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Description.

        forward method for RotaryPositionEmbedder.

        Executes PyTorch tensor operations.

        Args:
            q (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            k (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            indices (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))

        freqs: torch.Tensor = self.freqs.to(indices.device)
        phases = torch.outer(indices.reshape(-1), freqs)
        phases = phases.reshape(*indices.shape, -1)
        phases = torch.polar(torch.ones_like(phases), phases)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed


class TrellisMultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE and RMS normalisation."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: int | None = None,
        attn_type: str = "self",
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ) -> None:
        """__init__.

        Args:
            channels (int): Description.
            num_heads (int): Description.
            ctx_channels (int | None): Description.
            attn_type (str): Description.
            qkv_bias (bool): Description.
            use_rope (bool): Description.
            qk_rms_norm (bool): Description.
        """
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        if attn_type not in {"self", "cross"}:
            raise ValueError(f"Invalid attention type: {attn_type}")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels or channels
        self.attn_type = attn_type
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if attn_type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(
                self.ctx_channels,
                channels * 2,
                bias=qkv_bias,
            )

        if self.qk_rms_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        else:
            self.q_norm = None
            self.k_norm = None

        self.to_out = nn.Linear(channels, channels)
        self.rope = RotaryPositionEmbedder(channels) if use_rope else None

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """_reshape_heads.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """_apply_rope.

        Args:
            q (torch.Tensor): Description.
            k (torch.Tensor): Description.
            indices (torch.Tensor | None): Description.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Description.
        """
        if self.rope is None:
            return q, k
        return self.rope(q, k, indices)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            context (torch.Tensor | None): Description.
            indices (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TrellisMultiHeadAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            indices (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        bsz, seq_len, _ = x.shape
        if self.attn_type == "self":
            qkv = self.to_qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            if context is None:
                raise ValueError(
                    "context tensor must be provided for cross attention",
                )
            q = self.to_q(x)
            kv = self.to_kv(context)
            k, v = kv.chunk(2, dim=-1)

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        if self.use_rope:
            q, k = self._apply_rope(q, k, indices)

        if self.qk_rms_norm:
            assert self.q_norm is not None and self.k_norm is not None
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch_sdpa(q, k, v)
        attn = attn.permute(0, 2, 1, 3).contiguous()
        attn = attn.view(bsz, seq_len, self.channels)
        return self.to_out(attn)


class TrellisSelfAttention2D(nn.Module):
    """2D self-attention block built on top of Trellis attention."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        use_rope: bool = False,
        qk_rms_norm: bool = True,
        ff_multiplier: int = 4,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TrellisSelfAttention2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        bsz, channels, height, width = x.shape
        seq = x.view(bsz, channels, height * width).transpose(1, 2)
        seq_norm = self.norm(seq)
        attn_out = self.attn(seq_norm)
        seq = seq + attn_out
        seq = seq + self.ff(seq)
        return seq.transpose(1, 2).view(bsz, channels, height, width)


class MultiHeadAttention(nn.Module):
    """Minimal multi-head attention with optional rotary embeddings."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: int | None = None,
        kind: str = "self",
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
    ) -> None:
        """__init__.

        Args:
            channels (int): Description.
            num_heads (int): Description.
            ctx_channels (int | None): Description.
            kind (str): Description.
            use_rope (bool): Description.
            qk_rms_norm (bool): Description.
            qkv_bias (bool): Description.
        """
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.ctx_channels = ctx_channels or channels
        self.num_heads = num_heads
        self.kind = kind
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm
        head_dim = channels // num_heads

        if kind == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(
                self.ctx_channels,
                channels * 2,
                bias=qkv_bias,
            )

        if qk_rms_norm:
            self.q_norm = MultiHeadRMSNorm(head_dim, num_heads)
            self.k_norm = MultiHeadRMSNorm(head_dim, num_heads)
        else:
            self.q_norm = None
            self.k_norm = None

        self.to_out = nn.Linear(channels, channels)
        self.rope = RotaryPositionEmbedder(channels) if use_rope else None

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """_split_heads.

        Args:
            tensor (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return tensor.reshape(
            tensor.shape[0],
            tensor.shape[1],
            self.num_heads,
            -1,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:  # type: ignore[override]
        """forward.

        Args:
            x (torch.Tensor): Description.
            context (torch.Tensor | None): Description.
            indices (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for MultiHeadAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            indices (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.kind == "self":
            qkv = self.to_qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            if context is None:
                raise ValueError(
                    "context must be provided for cross attention",
                )
            q = self.to_q(x)
            k, v = self.to_kv(context).chunk(2, dim=-1)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rope is not None:
            q, k = self.rope(q, k, indices)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch_sdpa(q, k, v)
        attn = attn.permute(0, 2, 1, 3).contiguous()
        attn = attn.view(attn.shape[0], attn.shape[1], -1)
        return self.to_out(attn)


class FeedForwardNet(nn.Module):
    """Two-layer MLP used inside Transformer blocks."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0) -> None:
        """__init__.

        Args:
            channels (int): Description.
            mlp_ratio (float): Description.
        """
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, channels),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for FeedForwardNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


class ModulatedTransformerCrossBlock(nn.Module):
    """Cross-attention transformer block with adaptive modulation."""

    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ) -> None:
        """__init__.

        Args:
            channels (int): Description.
            ctx_channels (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            use_rope (bool): Description.
            qk_rms_norm (bool): Description.
            qk_rms_norm_cross (bool): Description.
            qkv_bias (bool): Description.
            share_mod (bool): Description.
        """
        super().__init__()
        self.share_mod = share_mod
        from .trellis_norm import LayerNorm32

        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            kind="self",
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
            qkv_bias=qkv_bias,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            kind="cross",
            use_rope=False,
            qk_rms_norm=qk_rms_norm_cross,
            qkv_bias=qkv_bias,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True),
            )
        else:
            self.register_parameter("adaLN_modulation", None)

    def _unpack_modulation(
        self,
        modulation: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """_unpack_modulation.

        Args:
            modulation (torch.Tensor): Description.
        Returns:
            tuple[torch.Tensor, ...]: Description.
        """
        if self.share_mod:
            return modulation.chunk(6, dim=1)
        return self.adaLN_modulation(modulation).chunk(6, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        modulation: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """forward.

        Args:
            x (torch.Tensor): Description.
            modulation (torch.Tensor): Description.
            context (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ModulatedTransformerCrossBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            modulation (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._unpack_modulation(modulation)

        residual = x
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.self_attn(h)
        x = residual + h * gate_msa.unsqueeze(1)

        residual = x
        h = self.norm2(x)
        h = self.cross_attn(h, context)
        x = residual + h

        residual = x
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        return residual + h * gate_mlp.unsqueeze(1)


__all__ = [
    "FeedForwardNet",
    "ModulatedTransformerCrossBlock",
    "MultiHeadAttention",
    "MultiHeadRMSNorm",
    "RotaryPositionEmbedder",
    "TrellisMultiHeadAttention",
    "TrellisSelfAttention2D",
]
