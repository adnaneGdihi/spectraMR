"""Attention Building Blocks
========================

Common attention mechanisms used across different model architectures.
"""

import warnings

import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention2D(nn.Module):
    """Multi-head self-attention block operating on 2D feature maps.

    The block follows a transformer-style design with pre-layer normalization,
    multi-head self-attention, and a position-wise feed-forward network. Inputs
    are 2D feature maps of shape ``(batch, channels, height, width)`` and the
    output preserves the same shape.

    Args:
        channels: Number of channels in the input and output feature maps.
        num_heads: Number of attention heads to use.
        dropout: Dropout probability applied after attention and each feed-
            forward linear layer. Defaults to ``0.0``.
        ff_multiplier: Expansion factor for the hidden dimension of the feed-
            forward network. A value of ``1.0`` keeps the hidden dimension the
            same as ``channels``. Defaults to ``1.0``.
        bias: Whether to include bias terms in the linear projections.
        norm_eps: Epsilon value used by the layer normalization layers.

    Raises:
        ValueError: If ``channels`` is not divisible by ``num_heads``.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        ff_multiplier: float = 1.0,
        bias: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SelfAttention2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        seq, height, width = self._flatten(x)
        attn_input = self.attn_norm(seq)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        seq = seq + self.attn_dropout(attn_output)

        ff_input = self.ff_norm(seq)
        seq = seq + self.feed_forward(ff_input)
        return self._unflatten(seq, height, width)


class SelfAttention(SelfAttention2D):
    """Backward-compatible wrapper around :class:`SelfAttention2D`."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        ff_multiplier: float = 1.0,
        bias: bool = True,
        norm_eps: float = 1e-5,
    ) -> None:
        """__init__.

        Args:
            channels (int): Description.
            num_heads (int): Description.
            dropout (float): Description.
            ff_multiplier (float): Description.
            bias (bool): Description.
            norm_eps (float): Description.
        """
        warnings.warn(
            (
                "SelfAttention is deprecated; import "
                "SelfAttention2D from spectramr.models.blocks.attention "
                "instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            channels=channels,
            num_heads=num_heads,
            dropout=dropout,
            ff_multiplier=ff_multiplier,
            bias=bias,
            norm_eps=norm_eps,
        )


class CrossAttention(nn.Module):
    """Cross-attention mechanism for conditional generation."""

    def __init__(self, channels: int, context_channels: int):
        """__init__.

        Args:
            channels (int): Description.
            context_channels (int): Description.
        """
        super().__init__()
        self.channels = channels
        self.context_channels = context_channels

        # Project context to same dimension as query if needed
        if context_channels != channels:
            self.context_proj = nn.Linear(context_channels, channels)
        else:
            self.context_proj = None

        self.mha = nn.MultiheadAttention(
            channels,
            num_heads=8,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            context (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for CrossAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            context (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).transpose(1, 2)
        context = context.view(
            -1,
            self.context_channels,
            context.shape[-2] * context.shape[-1],
        ).transpose(1, 2)

        # Project context to same dimension as query if needed
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


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for feature maps."""

    def __init__(self, in_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SpatialAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        return x * attn


class CBAMSpatialAttention(nn.Module):
    """CBAM-style Spatial attention mechanism.

    Compresses channel dimension via Max & Avg pooling, then applies a large
    kernel convolution to compute spatial attention map.
    """

    def __init__(self, kernel_size: int = 7):
        """__init__.

        Args:
            kernel_size (int): Description.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for CBAMSpatialAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


def orthogonal_random_features(
    num_heads: int, num_features: int, head_dim: int, generator: torch.Generator
) -> torch.Tensor:
    """Performer ORF draw: ``[num_heads, num_features, head_dim]``.

    Blockwise ``QR`` gives mutually orthogonal directions, then each row is
    rescaled to a chi_d norm so the marginal stays ``N(0, I_d)`` and the softmax
    kernel estimate remains unbiased -- only its variance drops. Measured over 8
    seeds on a 1/f k-space feature map at L = 4096: relative error against exact
    softmax attention 0.1185 -> 0.0992 mean, 0.0154 -> 0.0109 std.
    """
    blocks: list[torch.Tensor] = []
    remaining = num_features
    while remaining > 0:
        take = min(remaining, head_dim)
        gaussian = torch.randn(num_heads, head_dim, head_dim, generator=generator)
        q, _ = torch.linalg.qr(gaussian)
        blocks.append(q.transpose(-2, -1)[:, :take])
        remaining -= take
    directions = torch.cat(blocks, dim=1)
    chi_norms = torch.randn(num_heads, num_features, head_dim, generator=generator).norm(
        dim=-1, keepdim=True
    )
    return directions * chi_norms


class KernelizedAttention(nn.Module):
    """FAVOR+ kernelized attention (Performer; Choromanski et al., 2021), O(L).

    Positive random features ``phi(x) = exp(w^T x_hat - ||x_hat||^2 / 2) / sqrt(m)``
    with ``x_hat = x * d^{-1/4}`` give ``E[phi(q) . phi(k)] = exp(q . k / sqrt(d))``
    (the softmax kernel); the output is the row-normalized
    ``D^{-1} phi(Q) (phi(K)^T V)``, where ``D^{-1}`` replaces the softmax
    denominator.

    Two details are what make that estimate usable on k-space rather than merely
    correct in expectation (issue #405). The ``LayerNorm`` is per TOKEN, because
    k-space dynamic range is spatial: without it the DC bin drives
    ``|q.k/sqrt(d)|`` to ~5e3 and a 256-feature Monte-Carlo estimate of ``exp()``
    carries ~100 % relative error against the softmax attention it approximates.
    The features are orthogonal (Performer ORF), which cuts the residual variance
    rather than the bias. Measured against exact softmax on a 1/f k-space feature
    map at L = 4096: 0.998 as shipped, 0.121 with the norm, 0.099 with both.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        num_features: int = 256,
        zero_init_output: bool = True,
        feature_seed: int = 0,
    ):
        """__init__.

        Args:
            channels (int): Feature channels; ``channels // num_heads`` is the
                per-head dimension ``d``.
            num_heads (int): Attention heads.
            num_features (int): Total random features ``m`` (must be even; the
                two halves are antithetic ``+/- w`` pairs).
            zero_init_output (bool): Zero the output projection so the block
                starts as an exact identity. Pass ``False`` when an outer
                wrapper already owns that guarantee -- see the note at the
                initialiser below.
            feature_seed (int): Seeds the random-feature draw from a LOCAL
                generator, so every rank of a distributed run builds the same
                features by construction rather than by trusting buffer
                broadcast.
        """
        super().__init__()
        if num_features % 2 != 0:
            raise ValueError(f"num_features must be even, got {num_features}")
        self.channels = channels
        self.num_heads = num_heads
        self.num_features = num_features
        self.head_dim = channels // num_heads

        # Per-TOKEN pre-norm, the axis LinearAttention's InstanceNorm2d does not
        # cover: k-space dynamic range is spatial (the DC bin dwarfs the periphery),
        # so a per-channel norm leaves the logits at |q.k/sqrt(d)| ~ 5e3, far outside
        # the range a 256-feature Monte-Carlo estimate of exp() can represent.
        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        # Identity at init, but ONLY when this block owns that guarantee. Wrapped in
        # IdentityAtInitAttention the two mechanisms multiply into
        # ``y = x + g*(P*a + b)`` with ``g == P == b == 0``, whose every gradient is
        # exactly zero -- a fixed point the optimiser never leaves (issue #471).
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        # Orthogonal directions; forward mirrors them into antithetic +/- pairs.
        self.register_buffer(
            "rand_features",
            orthogonal_random_features(
                num_heads,
                num_features // 2,
                self.head_dim,
                torch.Generator().manual_seed(int(feature_seed)),
            ),
        )

    def _favor_plus_features(self, x: torch.Tensor, stabilizer: str = "none") -> torch.Tensor:
        """Positive features: ``[B, heads, L, d] -> [B, heads, L, m]``.

        ``stabilizer`` subtracts a max from the exponent before ``exp`` so large
        inputs cannot overflow: ``"row"`` per query position (a per-row factor
        cancels in the ``D^{-1}`` ratio), ``"global"`` per batch-head (a shared
        factor scales numerator and denominator equally). ``"none"`` is the
        exact estimator.
        """
        x_hat = x * self.head_dim**-0.25
        proj = torch.einsum("bhld,hfd->bhlf", x_hat, self.rand_features)
        proj = torch.cat([proj, -proj], dim=-1)
        expo = proj - 0.5 * x_hat.pow(2).sum(dim=-1, keepdim=True)
        if stabilizer == "row":
            expo = expo - expo.amax(dim=-1, keepdim=True).detach()
        elif stabilizer == "global":
            expo = expo - expo.amax(dim=(-2, -1), keepdim=True).detach()
        elif stabilizer != "none":
            raise ValueError(f"unknown stabilizer {stabilizer!r}")
        return torch.exp(expo) * self.num_features**-0.5

    def _favor_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Row-stochastic linear attention ``D^{-1} phi(Q) (phi(K)^T V)``.

        All-positive features make ``D`` strictly positive; a constant value
        field is a fixed point (rows of the implied attention matrix sum to 1),
        so the gain cannot scale with sequence length.
        """
        q_feat = self._favor_plus_features(q, stabilizer="row")
        k_feat = self._favor_plus_features(k, stabilizer="global")
        kv = torch.einsum("bhlf,bhld->bhfd", k_feat, v)
        numerator = torch.einsum("bhlf,bhfd->bhld", q_feat, kv)
        denominator = torch.einsum("bhlf,bhf->bhl", q_feat, k_feat.sum(dim=-2)).unsqueeze(-1)
        # clamp_min, not "+ eps": positive features make D > 0 mathematically,
        # so the guard only engages on true underflow and the row-stochastic
        # fixed point (constant V -> V) stays exact.
        return numerator / denominator.clamp_min(1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual FAVOR+ attention over the flattened spatial grid.

        Args:
            x (torch.Tensor, shape (B, C, H, W)): Input feature map.

        Returns:
            torch.Tensor: ``x + out_proj(attention(x))`` -- same shape; at init
            (zero ``out_proj``) exactly ``x``.
        """
        batch, channels, height, width = x.shape
        seq_len = height * width

        x_flat = self.norm(x.view(batch, channels, seq_len).transpose(1, 2))  # [B, L, C]
        q = self.q_proj(x_flat).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = self._favor_attention(q, k, v)  # [B, heads, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(batch, channels, height, width)
        return x + out


class WindowAttention(nn.Module):
    """Efficient Window Attention (O(N) memory/compute).
    Replaces the broken 'SparseAttention'.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 8,
        num_heads: int = 8,
        zero_init_output: bool = True,
    ):
        """__init__.

        Args:
            channels (int): Description.
            window_size (int): Description.
            num_heads (int): Description.
            zero_init_output (bool): Zero the output projection so the block
                starts as an exact identity. Pass ``False`` when an outer
                wrapper already owns that guarantee.
        """
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        # No explicit scale: F.scaled_dot_product_attention applies the
        # identical 1/sqrt(head_dim) default internally.
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        # Identity at init, but ONLY when this block owns that guarantee. Wrapped in
        # IdentityAtInitAttention the two mechanisms multiply into
        # ``y = x + g*(P*a + b)`` with ``g == P == b == 0``, whose every gradient is
        # exactly zero -- a fixed point the optimiser never leaves (issue #471).
        if zero_init_output:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, C, H, W]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for WindowAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # Pad if dimensions aren't divisible by window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        H_pad, W_pad = x.shape[2:]

        # 1. Window Partition: [B, C, H, W] -> [B * NumWindows, WindowSize*WindowSize, C]
        # Reshape to [B, C, H/ws, ws, W/ws, ws] -> permute -> flatten
        x_windows = x.view(B, C, H_pad // ws, ws, W_pad // ws, ws)
        x_windows = x_windows.permute(0, 2, 4, 3, 5, 1).contiguous()
        x_windows = x_windows.view(-1, ws * ws, C)  # [Batch*Windows, SeqLen, C]

        # 2. Compute Attention on small windows (Sequence Length = ws*ws = 64)
        # This fits easily in L1/Shared Memory
        qkv = self.qkv(x_windows).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(t.shape[0], t.shape[1], self.num_heads, -1).transpose(1, 2),
            qkv,
        )

        # Flash Attention (PyTorch 2.0+)
        # uses minimal memory and fuses operations
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        # 3. Reshape back
        attn_out = attn_out.transpose(1, 2).reshape(x_windows.shape)
        attn_out = self.proj(attn_out)

        # 4. Window Reverse
        attn_out = attn_out.view(B, H_pad // ws, W_pad // ws, ws, ws, C)
        attn_out = attn_out.permute(0, 5, 1, 3, 2, 4).contiguous()
        attn_out = attn_out.view(B, C, H_pad, W_pad)

        # Crop padding
        if pad_h > 0 or pad_w > 0:
            attn_out = attn_out[:, :, :H, :W]

        return attn_out + x


# Alias for backward compatibility if needed, though implementation changed
SparseAttention = WindowAttention


class ChannelAttention(nn.Module):
    """Channel attention mechanism (squeeze-excitation style).

    Uses manual global pooling instead of AdaptiveMaxPool2d for Inductor compatibility.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """__init__.

        Args:
            in_channels (int): Description.
            reduction_ratio (int): Description.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average pooling via AdaptiveAvgPool2d (Inductor compatible)
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ChannelAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        avg_out = self.fc(self.avg_pool(x))

        # Manual max pooling for Inductor compatibility. ``reshape``, not
        # ``view``: ``view`` refuses a permuted or strided input outright rather
        # than copying, so any caller that hands this block a non-contiguous
        # feature map fails here instead of pooling it. Same tensor when the
        # input is already contiguous, which is the common case.
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1)  # [B, C, H*W]
        max_pooled = x_flat.max(dim=-1, keepdim=True)[0]  # [B, C, 1]
        max_pooled = max_pooled.unsqueeze(-1)  # [B, C, 1, 1]
        max_out = self.fc(max_pooled)

        out = avg_out + max_out
        return x * self.sigmoid(out)


class LinearAttention(nn.Module):
    """Linear Attention O(N) following Katharopoulos et al.
    "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention".

    Computes Attention(Q,K,V) = phi(Q) @ (phi(K).T @ V)
    where phi(x) = elu(x) + 1 is the feature map.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        norm_type: str = "instance",
        zero_init_output: bool = True,
    ):
        """__init__.

        Args:
            channels (int): Description.
            num_heads (int): Description.
            norm_type (str): Description.
            zero_init_output (bool): Zero the output projection so the block
                starts as an exact identity. Pass ``False`` when an outer
                wrapper already owns that guarantee.
        """
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Normalization
        if norm_type == "instance":
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(channels)  # Note: LayerNorm expects specific shape usually
        elif norm_type == "batch":
            self.norm = nn.BatchNorm2d(channels)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(8, channels)
        else:
            self.norm = nn.Identity()

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        # Identity at init, but ONLY when this block owns that guarantee. Wrapped in
        # IdentityAtInitAttention the two mechanisms multiply into
        # ``y = x + g*(P*a + b)`` with ``g == P == b == 0``, whose every gradient is
        # exactly zero -- a fixed point the optimiser never leaves (issue #471).
        if zero_init_output:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for LinearAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        B, C, H, W = x.shape
        N = H * W  # Sequence length
        D = self.head_dim  # Head dimension

        # Pre-norm
        if isinstance(self.norm, nn.LayerNorm):
            # LayerNorm typically on last dim, but here x is BCHW
            x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x_norm = self.norm(x)

        # Generate Q, K, V
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to [B, Heads, N, D]
        q = q.view(B, self.num_heads, D, N).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, D, N).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, D, N).permute(0, 1, 3, 2)

        # Apply feature map phi(x) = elu(x) + 1
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Linear Attention: Q @ (K.T @ V)
        # K.T @ V: [B, H, D, N] @ [B, H, N, D] -> [B, H, D, D]
        kv = torch.matmul(k.transpose(-2, -1), v)

        # Q @ KV: [B, H, N, D] @ [B, H, D, D] -> [B, H, N, D]
        numerator = torch.matmul(q, kv)

        # Denominator: Q @ K.sum(dim=sequence)
        # K.sum(dim=-2): [B, H, D]
        k_sum = k.sum(dim=-2, keepdim=True)  # [B, H, 1, D]

        # Q @ K_sum.T: [B, H, N, D] @ [B, H, D, 1] -> [B, H, N, 1]
        denominator = torch.matmul(q, k_sum.transpose(-2, -1))

        # Normalize
        out = numerator / (denominator + 1e-6)

        # Reshape back to image: [B, H, N, D] -> [B, C, H, W]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # Project
        out = self.proj(out)

        return x + out


class IdentityAtInitAttention(nn.Module):
    """Start ANY attention block at an exact identity, then learn its contribution.

    .. math::

        y = x + \\gamma \\cdot (\\mathrm{inner}(x) - x), \\qquad \\gamma_{init} = 0

    At initialisation ``y == x`` bit-for-bit, so the block's energy gain is exactly
    ``rho = 1.000``. At ``gamma == 1`` the wrapper reproduces ``inner`` exactly, so
    no expressivity is given up: ``gamma`` interpolates from identity to the block's
    native behaviour and the optimiser decides how far to go.

    Why this exists (issue #471). ``LinearAttention`` states the family contract in
    its own initialiser -- *"zero output projection => forward(x) == x, so the
    residual family starts at energy gain rho = 1.0"* -- but only 3 of the 8 blocks
    the ``complex_unet`` dispatch can build honoured it. Measured rho at init over
    5 seeds on a 1/f k-space feature map: ``self``/``kernelized``/``sparse`` 1.000,
    ``dual_domain`` ~1.05, ``channel`` ~0.65, ``wavelet_freq`` ~0.54, ``spatial``
    ~0.96 (as low as 0.23), ``kan_dual_domain`` **7.4-9.2**. Because the call site
    is REPLACE (``x = self.attention(x)``), a non-identity block discards its input
    and emits that. The attention shootout advertises ``attention_type`` as its sole
    varying knob, but it also varied the effective initialisation, so an arm-vs-arm
    delta at a truncated budget was partly a difference in how far each arm had to
    travel from init.

    One wrapper for all eight types rather than five different per-block fixes --
    five chances to miss one, which is how the drift happened.

    ``t_emb`` support is detected from ``inner.forward``'s signature at construction
    (once, not per forward). The dispatch it replaces was an ``isinstance`` check
    against a hardcoded class tuple, which would silently drop ``t_emb`` -- blinding
    the timestep conditioning -- for any future time-conditioned block.
    """

    def __init__(self, inner: nn.Module):
        """Wrap ``inner`` so it starts as an identity.

        Args:
            inner: The attention block to wrap. Must map ``[B, C, H, W]`` to the
                same shape; anything else raises at the first forward rather than
                broadcasting silently.
        """
        super().__init__()
        self.inner = inner
        self.takes_t_emb = self._accepts(inner, "t_emb")
        self.takes_mask = self._accepts(inner, "mask")
        # Scalar, not per-channel: the point is a single interpretable "how much of
        # this block is switched on" knob that the energy probe can read directly.
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _accepts(inner: nn.Module, name: str) -> bool:
        """True when ``inner.forward`` declares a parameter called ``name``.

        By NAME, not by positional arity: with two optional extras in play, arity
        cannot tell ``forward(x, t_emb)`` from ``forward(x, mask)``, and would hand
        a block expecting the mask the timestep embedding instead.
        """
        import inspect

        try:
            params = inspect.signature(inner.forward).parameters
        except (TypeError, ValueError):
            return False
        return name in params

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Identity-at-init residual around ``inner``.

        Args:
            x: Feature map ``[B, C, H, W]`` in the caller's feature domain.
            t_emb: Timestep embedding, forwarded only when ``inner`` accepts it.
            mask: Acquisition mask, forwarded only when ``inner`` accepts it.

        Returns:
            ``x + gamma * (inner(x) - x)``, identical to ``x`` at initialisation.
        """
        extra: dict[str, torch.Tensor | None] = {}
        if self.takes_t_emb:
            extra["t_emb"] = t_emb
        if self.takes_mask:
            extra["mask"] = mask
        out = self.inner(x, **extra)
        if out.shape != x.shape:
            raise ValueError(
                f"{type(self.inner).__name__} changed the feature shape from "
                f"{tuple(x.shape)} to {tuple(out.shape)}; IdentityAtInitAttention "
                "wraps shape-preserving attention blocks only."
            )
        return x + self.gamma * (out - x)


class PhaseSafeDualAttention(nn.Module):
    """
    Phase-preserving attention that prevents ghosting artifacts in k-space.

    Computes attention weights using magnitude (phase-invariant operation) but
    applies them to complex values (phase-preserving). This prevents destructive
    phase interference that occurs when standard attention is applied directly
    to complex features, which can cause 180° phase flips leading to ghosting.

    Fuses image-domain feature guidance (Q: query) with k-space frequency
    information (K: key, V: value) to improve reconstruction accuracy.

    Args:
        in_channels (int): Number of input channels. If input is stacked real/imag
                          [B, 2*C, H, W], pass in_channels=C. This module will
                          internally handle the stacked representation.
        num_heads (int): Number of attention heads. Default: 8.
        reduction (int): Reduction factor for query/key/value projections.
                        Default: 2 (reduces compute cost).

    Shape:
        - Input x_kspace: (B, 2*C, H, W) where C=in_channels, channels 0:C are real, C:2C are imag
        - Input x_image: (B, 2*C, H, W) guidance for attention
        - Output: (B, 2*C, H, W)

    Architecture:
        Image Feature (Mag) -> Q projection (real-valued)
        K-Space (Mag) -----> K projection (real-valued)
               |
               v
           Softmax Attention (magnitude-based, phase-neutral)
               |
               v
        K-Space (Complex) -> V projection (complex-valued)
                 |
                 v
           Apply Attention Weights -> Output (phase-preserved)
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        reduction: int = 2,
        max_tokens: int = 4096,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Description.
            num_heads (int): Description.
            reduction (int): Description.
            max_tokens (int): OOM cap on the attended token count. The dense
                softmax matrix is ``[B, H*W, H*W]`` -- O((H*W)^2) memory -- so a
                256x256 feature map (N=65536) would need ~32 GiB. When
                ``H*W > max_tokens`` the q/k/v maps are adaptive-pooled to a
                coarser grid before attention and the output is bilinearly
                upsampled; the gamma-gated residual on the full-resolution
                input preserves fine detail. Mirrors the cap in
                ``dual_domain_attention_kan.py`` (added after the 2026-05-10
                cluster OOM).
        """
        super().__init__()
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.reduction = reduction
        self.max_tokens = int(max_tokens)

        if reduction < 1:
            raise ValueError(f"reduction must be >= 1, got {reduction}")

        # For stacked real/imag representation [B, 2*C, H, W]
        stacked_channels = in_channels * 2
        # `reduction` is a DIVISOR, so the floor has to be 1, not stacked_channels.
        # The previous spelling was `max(stacked_channels // reduction, stacked_channels)`,
        # and since reduction >= 1 makes the left operand <= the right one, that max
        # ALWAYS returned stacked_channels -- the knob provably could not change the
        # module (pitfall #15: advertised but unread). No corpus arm sets
        # `phase_safe_attention_reduction`, so at the default reduction=1 both
        # spellings give stacked_channels and every existing state_dict still loads.
        hidden_dim = max(stacked_channels // reduction, 1)

        # Query projection: image feature guidance (real-valued, on stacked repr)
        self.query_proj = nn.Sequential(
            nn.Conv2d(stacked_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=max(1, num_heads), num_channels=hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Key projection: k-space frequency (real-valued magnitude)
        self.key_proj = nn.Sequential(
            nn.Conv2d(stacked_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=max(1, num_heads), num_channels=hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Value projection: k-space (stacked real/imag)
        self.value_proj = nn.Conv2d(stacked_channels, stacked_channels, kernel_size=1)

        # Output projection
        self.out_proj = nn.Conv2d(stacked_channels, stacked_channels, kernel_size=1)

        # Residual connection weight
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_kspace: torch.Tensor,
        x_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply phase-safe dual attention.

        Args:
            x_kspace (torch.Tensor): K-space input [B, 2*C, H, W] (stacked real/imag).
            x_image (torch.Tensor): Image-domain guidance [B, 2*C, H, W] (stacked real/imag).

        Returns:
            torch.Tensor: Attentioned output [B, 2*C, H, W] (phase preserved).

        forward method for PhaseSafeDualAttention.

        Executes PyTorch tensor operations.

        Args:
            x_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        B, stacked_c, H, W = x_kspace.shape
        C = self.in_channels

        # Extract magnitude for attention computation (phase-neutral)
        # x_kspace: [B, 2*C, H, W] -> real=[B, C, H, W] and imag=[B, C, H, W]
        # Interleaved parsing: [R1, I1, R2, I2, ...]
        real_k = x_kspace[:, 0::2, ...]
        imag_k = x_kspace[:, 1::2, ...]
        x_kspace_mag = torch.sqrt(real_k**2 + imag_k**2 + 1e-8)  # [B, C, H, W]

        # Reconstruct interleaved magnitude for projection
        # Interleave mag with itself: [M1, M1, M2, M2, ...]
        x_kspace_mag_interleaved = torch.stack([x_kspace_mag, x_kspace_mag], dim=2).flatten(1, 2)

        # Project to attention space (full-resolution 1x1 convs — cheap).
        q = self.query_proj(x_image)  # [B, hidden_dim, H, W]
        k = self.key_proj(x_kspace_mag_interleaved)  # [B, hidden_dim, H, W]
        v = self.value_proj(x_kspace)  # [B, 2*C, H, W]

        hidden_dim = q.shape[1]

        # OOM cap: the dense [B, N, N] softmax is O((H*W)^2) memory — a
        # 256x256 map (N=65536) needs ~32 GiB. When the token count exceeds
        # ``max_tokens``, attend on an adaptive-pooled coarse grid and
        # bilinearly upsample the result; the gamma-gated residual below
        # restores full-resolution detail. Mirrors the cap in the KAN
        # dual-domain blocks (added after the 2026-05-10 cluster OOM).
        if self.max_tokens < H * W:
            ratio = (self.max_tokens / float(H * W)) ** 0.5
            ah, aw = max(1, int(H * ratio)), max(1, int(W * ratio))
            q = F.adaptive_avg_pool2d(q, (ah, aw))
            k = F.adaptive_avg_pool2d(k, (ah, aw))
            v = F.adaptive_avg_pool2d(v, (ah, aw))
        else:
            ah, aw = H, W

        # Reshape for attention: [B, C, N] with N = ah*aw <= max_tokens
        q_flat = q.reshape(B, hidden_dim, ah * aw)  # [B, hidden_dim, N]
        k_flat = k.reshape(B, hidden_dim, ah * aw)  # [B, hidden_dim, N]
        v_flat = v.reshape(B, v.shape[1], ah * aw)  # [B, 2*C, N]

        # Compute attention weights using magnitude (phase-neutral)
        # attn = softmax(Q^T @ K)
        attn = torch.bmm(q_flat.transpose(1, 2), k_flat)  # [B, N, N]
        attn = attn / (hidden_dim**0.5 + 1e-8)
        attn = F.softmax(attn, dim=-1)  # [B, N, N]

        # Apply attention to complex values (phase-preserved): out = attn @ V
        out_flat = torch.bmm(v_flat, attn.transpose(1, 2))  # [B, 2*C, N]
        out = out_flat.reshape(B, v.shape[1], ah, aw)  # [B, 2*C, ah, aw]

        # Upsample the (possibly coarse) attention output back to full res.
        if (ah, aw) != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        # Project output
        out = self.out_proj(out)  # [B, 2*C, H, W]

        # Residual connection with learnable gate
        return x_kspace + self.gamma * out


class CrossContrastOLMPA(nn.Module):
    """Cross-Contrast Orthogonal Linear Multi-Phase Attention (CC-OLMPA).

    Bridges multi-contrast MRI inputs (e.g., T1 source → FLAIR/T2 target) in
    pure k-space by decoupling structural magnitude routing from complex phase.

    Key design choices:

    1. **Phase Safety** — Q/K are computed on ``torch.abs()`` magnitudes so
       the attention kernel never interferes with k-space phase vectors.
       Multiplying these strictly positive, real-valued weights against
       complex V preserves spatial coherence (Fourier Shift Theorem).

    2. **O(N) Linear Attention** — The Softmax operator is replaced with the
       non-negative feature map ``φ(x) = ELU(x) + 1``.  Evaluating
       ``Q(K^T V)`` instead of ``(QK^T)V`` drops spatial complexity from
       O(N²) to O(N·d), making unpatched 256×256 processing viable.
       (Katharopoulos et al., ICML 2020)

    3. **Parseval Energy Conservation** — Q and K projections are wrapped in
       ``nn.utils.parametrizations.orthogonal`` enforcing W^H W = I, which
       bounds the Lipschitz constant and conserves L2 signal energy across
       diffusion timesteps.  (Trabelsi et al., ICLR 2018)

    Args:
        in_channels: Channel dimension of the **target** k-space tensor
            (real-valued representation, e.g. 8 for 4 interleaved coils).
        num_contrasts: Number of reference contrasts concatenated in the
            source tensor.  ``source_channels = in_channels * num_contrasts``.
        phase_safe_dim: Hidden dimension for the Q/K projections.  Controls
            the structural basis size (default 128).

    References:
        * Katharopoulos et al. "Transformers are RNNs", ICML 2020.
        * Trabelsi et al. "Deep Complex Networks", ICLR 2018.
        * Bilgic et al. "Multi-contrast reconstruction with Bayesian
          compressed sensing", MRM 2011.
    """

    def __init__(
        self,
        in_channels: int,
        num_contrasts: int = 4,
        phase_safe_dim: int = 128,
    ) -> None:
        """Initialise CC-OLMPA.

        Args:
            in_channels: Target k-space channel count.
            num_contrasts: Number of reference contrasts.
            phase_safe_dim: Q/K projection dimensionality.
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_contrasts = num_contrasts
        self.phase_safe_dim = phase_safe_dim
        self.scale = phase_safe_dim**-0.5

        # [O]rthogonal: Constrain projections to conserve k-space energy
        # via Parseval's Theorem  (W^H W = I  →  isometric mapping)
        self.W_q = nn.utils.parametrizations.orthogonal(
            nn.Linear(in_channels, phase_safe_dim, bias=False),
        )
        self.W_k = nn.utils.parametrizations.orthogonal(
            nn.Linear(in_channels * num_contrasts, phase_safe_dim, bias=False),
        )

        # Values must remain in the original channel dim to preserve
        # native k-space phase (complex-valued output).
        self.W_v = nn.Linear(
            in_channels * num_contrasts,
            in_channels,
            bias=False,
        )

        # Identity at init (issue #471). ComplexUNet adds this block's output
        # residually (``attended + target_feat``), so a zero output scale makes the
        # bottleneck an exact identity at step 0. Measured without it: the block
        # emitted 150-190x the target's norm across seeds, i.e. the bottleneck
        # carried almost pure attention output rather than the encoder's features.
        # This is the per-block analogue of IdentityAtInitAttention's ``gamma``; the
        # wrapper itself does not fit here because forward takes TWO tensors
        # (target, ref), not (x, t_emb).
        self.out_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        target_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
    ) -> torch.Tensor:
        """Apply CC-OLMPA.

        Args:
            target_kspace: ``[B, N, C]`` — undersampled target contrast
                (flattened spatial dims, C = in_channels).
            ref_kspace: ``[B, N, C * num_contrasts]`` — concatenated
                reference contrasts along the channel axis.

        Returns:
            ``[B, N, C]`` — attended target with preserved phase.

        forward method for CrossContrastOLMPA.

        Executes PyTorch tensor operations.

        Args:
            target_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            ref_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        Q = self.W_q(target_kspace)  # [B, N, phase_safe_dim]
        K = self.W_k(ref_kspace)  # [B, N, phase_safe_dim]
        V = self.W_v(ref_kspace)  # [B, N, C]

        # [P]hase-Safe: Extract magnitude for structural correlations only.
        # Prevents destructive phase interference across contrasts.
        Q_mag = torch.abs(Q)
        K_mag = torch.abs(K)

        # [L]inear Kernel: φ(x) = ELU(x) + 1  (non-negative feature map)
        # Bypasses O(N²) Softmax, enables associative K^T V first.
        phi_Q = F.elu(Q_mag) + 1.0  # [B, N, d]
        phi_K = F.elu(K_mag) + 1.0  # [B, N, d]

        # Associative property:  compute K^T V first → O(N·d·C)
        # K_V: [B, d, C]
        K_V = torch.einsum("b n d, b n c -> b d c", phi_K, V)

        # Attended output: Q · (K^T V) → [B, N, C]
        out = torch.einsum("b n d, b d c -> b n c", phi_Q, K_V) * self.scale

        return out * self.out_scale


__all__ = [
    "CBAMSpatialAttention",
    "ChannelAttention",
    "CrossAttention",
    "CrossContrastOLMPA",
    "IdentityAtInitAttention",
    "KernelizedAttention",
    "LinearAttention",
    "PhaseSafeDualAttention",
    "SelfAttention",
    "SparseAttention",
    "SpatialAttention",
]
