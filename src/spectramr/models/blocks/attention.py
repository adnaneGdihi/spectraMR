"""Attention Building Blocks
========================

Common attention mechanisms used across different model architectures.
"""

import math
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.layers.complex_conv import ComplexConv2d


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

    The norm feeds ``q`` and ``k`` only. ``V`` is projected (bias-free) from the
    raw tokens, so the attended values keep the power-law magnitude of the bins
    they come from; values drawn from whitened tokens are flat in radius and
    write O(1) energy into outer bins whose true magnitude is orders lower.
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
        # Its mean subtraction over interleaved Re/Im is not phase-equivariant,
        # which is acceptable because it only shapes the attention weights.
        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        # No bias: a row-stochastic attention maps a constant value field to itself,
        # so a value bias would paint one constant over every k-space bin.
        self.v_proj = nn.Linear(channels, channels, bias=False)
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

        tokens = x.reshape(batch, channels, seq_len).transpose(1, 2)  # [B, L, C]
        x_flat = self.norm(tokens)
        q = self.q_proj(x_flat).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = self._favor_attention(q, k, v)  # [B, heads, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(batch, channels, height, width)
        return x + out


#: Upper clamp on the learned QK-norm temperature (Swin-v2's ``log(100)``).
_MAX_LOGIT_SCALE = math.log(100.0)


def qk_normalize(t: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """L2-normalise ``t`` over its last (head) dim and scale by ``exp(log_scale)``.

    k-space magnitude follows a power law with a dominant DC peak, so raw
    dot-product logits grow with magnitude squared and the softmax collapses onto
    the DC/low-frequency tokens. On unit vectors a logit is bounded by the learned
    temperature whatever the token energy. ``log_scale`` broadcasts against ``t``.
    """
    return F.normalize(t, dim=-1) * log_scale.clamp(max=_MAX_LOGIT_SCALE).exp()


class WindowAttention(nn.Module):
    """PE-axial window attention with QK-normalised logits and a PE-distance bias.

    Every window spans the FULL width ``W`` -- the phase-encode axis, along which
    the mask acquires whole columns -- and ``window_size`` readout rows (``H``).
    Square 8x8 windows were narrower than the gap between acquired columns at
    R >= 16 (at R = 32 the median missing-to-acquired gap is ~30 columns, max
    ~103 at W = 256), so 86 % of windows held no acquired column and had nothing
    to propagate. A learned bias indexed by PE distance (``2 * max_width - 1``
    entries per head) gives the logits the position information the windows used
    to lack. ``H`` not divisible by ``window_size`` is zero-padded, the padded
    tokens are masked out of the keys, and the output is cropped back.

    The class name and the ``sparse`` dispatch key (via ``SparseAttention``) are
    kept for config compatibility; the window geometry is what changed.
    """

    def __init__(
        self,
        channels: int,
        window_size: int = 2,
        num_heads: int = 8,
        zero_init_output: bool = True,
        max_width: int = 512,
    ):
        """__init__.

        Args:
            channels (int): Token channels; must divide by ``num_heads``.
            window_size (int): Readout rows per window. Tokens per window are
                ``window_size * W``, so this is the memory knob.
            num_heads (int): Attention heads.
            zero_init_output (bool): Zero the output projection so the block
                starts as an exact identity. Pass ``False`` when an outer
                wrapper already owns that guarantee.
            max_width (int): Largest PE width the bias table covers; a wider
                input raises rather than reusing clipped distances.
        """
        super().__init__()
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if max_width < 1:
            raise ValueError(f"max_width must be >= 1, got {max_width}")
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must divide by num_heads={num_heads}")
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.max_width = max_width
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.logit_scale = nn.Parameter(torch.full((num_heads, 1, 1), math.log(10.0)))
        self.relative_position_bias = nn.Parameter(torch.zeros(num_heads, 2 * max_width - 1))
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)
        # Identity at init, but ONLY when this block owns that guarantee. Wrapped in
        # IdentityAtInitAttention the two mechanisms multiply into
        # ``y = x + g*(P*a + b)`` with ``g == P == b == 0``, whose every gradient is
        # exactly zero -- a fixed point the optimiser never leaves (issue #471).
        if zero_init_output:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def _partition(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """``[B, C, H, W] -> [B * nW, window_size * W, C]`` plus the row padding."""
        batch, channels, height, width = x.shape
        if width > self.max_width:
            raise ValueError(
                f"WindowAttention: PE width {width} exceeds max_width={self.max_width}; "
                "construct the block with a larger max_width"
            )
        ws = self.window_size
        pad_h = (-height) % ws
        if pad_h:
            x = F.pad(x, (0, 0, 0, pad_h))
        n_win = (height + pad_h) // ws
        tokens = x.reshape(batch, channels, n_win, ws, width).permute(0, 2, 3, 4, 1)
        return tokens.reshape(batch * n_win, ws * width, channels), pad_h

    def _reverse(self, tokens: torch.Tensor, batch: int, height: int, width: int) -> torch.Tensor:
        """Inverse of :meth:`_partition`, cropping the padded rows."""
        channels = tokens.shape[-1]
        rows = tokens.shape[0] // batch * self.window_size
        out = tokens.reshape(batch, -1, self.window_size, width, channels)
        out = out.permute(0, 4, 1, 2, 3).reshape(batch, channels, rows, width)
        return out[:, :, :height]

    def _qkv(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-head ``q`` (normalised and temperature-scaled), unit ``k`` and ``v``."""
        n, length, _ = tokens.shape
        q, k, v = (
            t.reshape(n, length, self.num_heads, self.head_dim).transpose(1, 2)
            for t in self.qkv(tokens).chunk(3, dim=-1)
        )
        return qk_normalize(q, self.logit_scale), F.normalize(k, dim=-1), v

    def _attn_bias(
        self, width: int, pad_h: int, ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Additive logits ``[1, heads, L, L]`` for interior windows and for the last one.

        Only the last window row-block can hold padded keys, so the padding mask is
        folded into a second single-window bias instead of being repeated over every
        window and batch item; neither tensor scales with ``B`` or ``nW``.
        """
        cols = torch.arange(width, device=ref.device)
        index = cols[:, None] - cols[None, :] + self.max_width - 1
        ws = self.window_size
        # Distance depends on the column only, so one [W, W] tile repeats over the rows.
        bias = self.relative_position_bias[:, index].repeat(1, ws, ws).unsqueeze(0).to(ref.dtype)
        if not pad_h:
            return bias, None
        padded = torch.zeros(ws, width, dtype=torch.bool, device=ref.device)
        padded[ws - pad_h :] = True
        key_mask = torch.zeros(padded.shape, dtype=ref.dtype, device=ref.device)
        key_mask = key_mask.masked_fill(padded, float("-inf")).reshape(1, 1, 1, ws * width)
        return bias, bias + key_mask

    def _split_windows(self, t: torch.Tensor, batch: int) -> torch.Tensor:
        """``[B * nW, heads, L, d] -> [B, nW, heads, L, d]`` as a view."""
        return t.reshape(batch, -1, *t.shape[1:])

    def attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax weights ``[B * nW, heads, L, L]`` -- a diagnostic, materialised densely."""
        batch = x.shape[0]
        tokens, pad_h = self._partition(x)
        q, k, _ = self._qkv(tokens)
        bias, last_bias = self._attn_bias(x.shape[-1], pad_h, q)
        logits = self._split_windows(q @ k.transpose(-2, -1), batch) + bias
        if last_bias is not None:
            logits[:, -1] += last_bias - bias
        return logits.flatten(0, 1).softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual PE-axial attention: ``[B, C, H, W] -> [B, C, H, W]``."""
        batch, _, height, width = x.shape
        tokens, pad_h = self._partition(x)
        q, k, v = self._qkv(tokens)
        bias, last_bias = self._attn_bias(width, pad_h, q)
        # scale=1.0: the learned temperature is already folded into q.
        if last_bias is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
        else:
            # SDPA takes the window dim as a batch dim, so each [1, heads, L, L] bias
            # broadcasts over B and nW instead of being materialised per window.
            q, k, v = (self._split_windows(t, batch) for t in (q, k, v))
            head = F.scaled_dot_product_attention(
                q[:, :-1], k[:, :-1], v[:, :-1], attn_mask=bias, scale=1.0
            )
            tail = F.scaled_dot_product_attention(
                q[:, -1:], k[:, -1:], v[:, -1:], attn_mask=last_bias, scale=1.0
            )
            out = torch.cat((head, tail), dim=1).flatten(0, 1)
        out = self.proj(out.transpose(1, 2).reshape(tokens.shape))
        return x + self._reverse(out, batch, height, width)


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
        # Floor of 1: below reduction_ratio channels a zero-width bottleneck made
        # the gate a constant sigmoid(0) = 0.5 that no parameter could move.
        hidden = max(1, in_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, in_channels, 1, bias=False),
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


def _complex_channels(channels: int, owner: str) -> int:
    """Complex channel count of an interleaved ``[R1, I1, R2, I2, ...]`` map; raise on odd."""
    if channels % 2 != 0:
        raise ValueError(f"{owner} expects interleaved Re/Im channels (even count), got {channels}")
    return channels // 2


class ComplexChannelAttention(nn.Module):
    """Squeeze-excitation gate with ONE gain per complex channel, shared by Re and Im.

    :class:`ChannelAttention` gates every real channel separately, so the Re and
    Im halves of one complex channel get different sigmoid gains -- a phase
    rotation -- and its signed avg/max pools are driven by the DC bin. Here the
    squeeze is ``log1p(mean |z|)`` per complex channel: ``|z|`` is invariant to a
    global phase, so ``gate(x * e^{i theta}) = gate(x) * e^{i theta}``, and the
    log keeps the power-law magnitude range out of the sigmoid's saturated tails.
    For interleaved k-space features; real image features keep ChannelAttention.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """__init__.

        Args:
            in_channels (int): Interleaved real channel count (even).
            reduction_ratio (int): Bottleneck divisor applied to the complex
                channel count (floor 1).
        """
        super().__init__()
        complex_channels = _complex_channels(in_channels, type(self).__name__)
        hidden = max(1, complex_channels // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(complex_channels, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, complex_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gate ``[B, 2C, H, W]`` interleaved features by a per-complex-channel gain."""
        batch, channels, height, width = x.shape
        z = x.reshape(batch, channels // 2, 2, height * width)
        squeeze = torch.log1p(torch.linalg.vector_norm(z, dim=2).mean(dim=-1))  # [B, C]
        gain = torch.sigmoid(self.fc(squeeze))
        return (z * gain[:, :, None, None]).reshape(batch, channels, height, width)


#: q/k pre-norms LinearAttention accepts; an unknown name raises (non-negotiable 3).
_LINEAR_ATTENTION_NORMS: dict[str, Callable[[int], nn.Module]] = {
    "instance": lambda c: nn.InstanceNorm2d(c, affine=True),
    "layer": nn.LayerNorm,
    "batch": nn.BatchNorm2d,
    "group": lambda c: nn.GroupNorm(8, c),
    "none": lambda c: nn.Identity(),
}


class LinearAttention(nn.Module):
    """Linear Attention O(N) following Katharopoulos et al.
    "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention".

    Computes Attention(Q,K,V) = phi(Q) @ (phi(K).T @ V)
    where phi(x) = elu(x) + 1 is the feature map.

    Input is interleaved ``[R1, I1, R2, I2, ...]``. Only ``q``/``k`` see the
    pre-norm, since they produce real weights. The value and output projections
    are complex-linear 1x1 maps (:class:`ComplexConv2d`, no bias) applied to the
    raw input: a real 1x1 conv over interleaved channels mixes Re and Im across
    coils arbitrarily, and values from normalised tokens lose the k-space
    magnitude profile. Each head holds whole Re/Im pairs, so for fixed weights
    the attended output is complex-linear in the input.
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
            channels (int): Interleaved real channel count; ``channels //
                num_heads`` must be even so no head splits a Re/Im pair.
            num_heads (int): Attention heads.
            norm_type (str): Pre-norm for the q/k path.
            zero_init_output (bool): Zero the output projection so the block
                starts as an exact identity. Pass ``False`` when an outer
                wrapper already owns that guarantee.
        """
        super().__init__()
        complex_channels = _complex_channels(channels, type(self).__name__)
        if channels % num_heads != 0 or (channels // num_heads) % 2 != 0:
            raise ValueError(
                f"LinearAttention: channels={channels} / num_heads={num_heads} must give an "
                "even head dim so each head holds whole Re/Im pairs"
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        if norm_type not in _LINEAR_ATTENTION_NORMS:
            raise ValueError(
                f"LinearAttention: unknown norm_type {norm_type!r}; "
                f"expected one of {sorted(_LINEAR_ATTENTION_NORMS)}"
            )
        self.norm = _LINEAR_ATTENTION_NORMS[norm_type](channels)

        self.qk = nn.Conv2d(channels, channels * 2, 1)
        self.v_proj = ComplexConv2d(complex_channels, complex_channels, 1, bias=False)
        self.proj = ComplexConv2d(complex_channels, complex_channels, 1, bias=False)
        # Identity at init, but ONLY when this block owns that guarantee. Wrapped in
        # IdentityAtInitAttention the two mechanisms multiply into
        # ``y = x + g*(P*a + b)`` with ``g == P == b == 0``, whose every gradient is
        # exactly zero -- a fixed point the optimiser never leaves (issue #471).
        if zero_init_output:
            nn.init.zeros_(self.proj.weight_real)
            nn.init.zeros_(self.proj.weight_imag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual linear attention: ``[B, C, H, W] -> [B, C, H, W]``."""
        B, C, H, W = x.shape
        N = H * W
        D = self.head_dim

        if isinstance(self.norm, nn.LayerNorm):
            x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            x_norm = self.norm(x)

        q, k = self.qk(x_norm).chunk(2, dim=1)
        v = self.v_proj(x)

        # [B, Heads, N, D]; contiguous channel groups keep each Re/Im pair in one head.
        q = F.elu(q.reshape(B, self.num_heads, D, N).transpose(-2, -1)) + 1.0
        k = F.elu(k.reshape(B, self.num_heads, D, N).transpose(-2, -1)) + 1.0
        v = v.reshape(B, self.num_heads, D, N).transpose(-2, -1)

        kv = torch.matmul(k.transpose(-2, -1), v)  # [B, H, D, D]
        numerator = torch.matmul(q, kv)  # [B, H, N, D]
        denominator = torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1))
        out = numerator / (denominator + 1e-6)

        out = out.transpose(-2, -1).reshape(B, C, H, W)
        return x + self.proj(out)


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

    Attention is linear (Katharopoulos et al., 2020), O(N) at the native
    resolution: pooling complex k-space to a coarse grid and upsampling the
    result averaged acquired with missing columns. ``q``/``k`` are L2-normalised
    per token and head and scaled by a learned per-head temperature before the
    ``elu + 1`` feature map, so the DC peak cannot claim the attention mass.
    Weights are real and row-stochastic, and the value and output projections
    are complex-linear (:class:`ComplexConv2d`, no bias), so the attended output
    is a phase-preserving combination of complex values.

    Args:
        in_channels (int): Number of COMPLEX input channels ``C``; tensors carry
            ``2*C`` interleaved real channels.
        num_heads (int): Attention heads; must divide the q/k width. Default: 8.
        reduction (int): Divisor on the q/k projection width. Default: 2.

    Shape:
        - Input x_kspace: (B, 2*C, H, W), interleaved [R1, I1, R2, I2, ...]
        - Input x_image: (B, 2*C, H, W) guidance for attention, same layout
        - Output: (B, 2*C, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        reduction: int = 2,
    ) -> None:
        """__init__.

        Args:
            in_channels (int): Complex channel count ``C``.
            num_heads (int): Attention heads.
            reduction (int): q/k projection width divisor (>= 1).
        """
        super().__init__()
        if reduction < 1:
            raise ValueError(f"reduction must be >= 1, got {reduction}")
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.reduction = reduction

        stacked_channels = in_channels * 2
        # `reduction` is a divisor, so the floor is 1 channel (Conv2d rejects 0).
        hidden_dim = max(stacked_channels // reduction, 1)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"PhaseSafeDualAttention: q/k width {hidden_dim} must divide by num_heads={num_heads}"
            )
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Sequential(
            nn.Conv2d(stacked_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=num_heads, num_channels=hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.key_proj = nn.Sequential(
            nn.Conv2d(stacked_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(num_groups=num_heads, num_channels=hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.qk_log_scale = nn.Parameter(
            torch.full((1, num_heads, 1, 1), 0.5 * math.log(self.head_dim))
        )
        self.value_proj = ComplexConv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.out_proj = ComplexConv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def _features(self, t: torch.Tensor) -> torch.Tensor:
        """``[B, hidden, H, W] -> [B, heads, head_dim, N]``: QK-norm, then ``elu + 1``."""
        b = t.shape[0]
        t = t.reshape(b, self.num_heads, self.head_dim, -1).transpose(-2, -1)
        return (F.elu(qk_normalize(t, self.qk_log_scale)) + 1.0).transpose(-2, -1)

    def kernel_features(
        self, x_kspace: torch.Tensor, x_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positive feature maps ``phi(q), phi(k)``, each ``[B, heads, head_dim, N]``."""
        real_k = x_kspace[:, 0::2]
        imag_k = x_kspace[:, 1::2]
        mag = torch.sqrt(real_k**2 + imag_k**2 + 1e-8)
        mag_interleaved = torch.stack([mag, mag], dim=2).flatten(1, 2)
        return self._features(self.query_proj(x_image)), self._features(self.key_proj(mag_interleaved))

    def forward(self, x_kspace: torch.Tensor, x_image: torch.Tensor) -> torch.Tensor:
        """Apply phase-safe dual attention.

        Args:
            x_kspace (torch.Tensor): ``[B, 2*C, H, W]`` interleaved k-space;
                value source and residual base.
            x_image (torch.Tensor): ``[B, 2*C, H, W]`` image-domain guidance;
                steers the weights only.

        Returns:
            torch.Tensor: ``x_kspace + gamma * out_proj(attention)``, same shape.
        """
        B, stacked_c, H, W = x_kspace.shape
        if stacked_c != 2 * self.in_channels:
            raise ValueError(
                f"PhaseSafeDualAttention(in_channels={self.in_channels}) expects "
                f"{2 * self.in_channels} interleaved channels, got {stacked_c}"
            )
        phi_q, phi_k = self.kernel_features(x_kspace, x_image)
        v = self.value_proj(x_kspace).reshape(B, stacked_c, H * W)
        # fp32 reductions: a sum of N = H*W k-space values overflows fp16 under AMP.
        with torch.autocast(device_type=x_kspace.device.type, enabled=False):
            phi_q, phi_k, v32 = phi_q.float(), phi_k.float(), v.float()
            kv = torch.einsum("bhdn,bcn->bhdc", phi_k, v32)
            numerator = torch.einsum("bhdn,bhdc->bhcn", phi_q, kv)
            denominator = torch.einsum("bhdn,bhd->bhn", phi_q, phi_k.sum(dim=-1))
            # Positive features make D > 0; heads are averaged, so rows stay stochastic.
            out = (numerator / denominator.clamp_min(1e-6).unsqueeze(2)).mean(dim=1)
        out = self.out_proj(out.to(v.dtype).reshape(B, stacked_c, H, W))
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
    "ComplexChannelAttention",
    "CrossAttention",
    "CrossContrastOLMPA",
    "IdentityAtInitAttention",
    "KernelizedAttention",
    "LinearAttention",
    "PhaseSafeDualAttention",
    "SelfAttention",
    "SparseAttention",
    "SpatialAttention",
    "WindowAttention",
    "qk_normalize",
]
