"""Shared Swin Transformer Blocks (2D).

Common components for Swin Transformer and SwinIR. The windowing primitives
live in :mod:`spectramr.models.blocks.swin_windows`; ``window_partition`` and
``window_reverse`` are re-exported here so existing imports keep working.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.blocks.swin_windows import (
    ShiftedWindowMaskCache,
    padded_resolution,
    window_count,
)
from spectramr.models.blocks.swin_windows import window_partition as window_partition
from spectramr.models.blocks.swin_windows import window_reverse as window_reverse


class MLP(nn.Module):
    """Multilayer Perceptron."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        dropout: float = 0.0,
    ):
        """__init__.

        Args:
            in_features (int): Description.
            hidden_features (Optional[int]): Description.
            out_features (Optional[int]): Description.
            dropout (float): Description.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for MLP.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        """__init__.

        Args:
            drop_prob (float): Description.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DropPath.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return drop_path(x, self.drop_prob, self.training)


class WindowAttention(nn.Module):
    """Window-based Multi-head Self Attention (W-MSA) with relative position bias."""

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
            qk_scale (Optional[float]): Description.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        num_windows: int | None = None,
    ) -> torch.Tensor:
        """forward.

        Args:
            num_windows: How many windows the caller partitioned into. Used
                only to validate ``mask``; the shape checks cannot catch a
                count that merely *divides* the row count (#1345).
            x (torch.Tensor): Description.
            mask (Optional[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.

        forward method for WindowAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B_, N, C = x.shape
        # qkv(x): [B, N, 3*C] -> reshape -> [B, N, 3, nH, C//nH] -> permute -> [3, B, nH, N, C//nH]
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # (1, nH, N, N)

        if mask is not None:
            # ``nW`` is read off the mask, so a mask from another resolution
            # reinterprets the window axis instead of failing. Raise rather
            # than attend unmasked (non-negotiable 3).
            nW = mask.shape[0]
            if num_windows is not None and nW != num_windows:
                raise ValueError(
                    f"attention mask describes {nW} windows but {num_windows} were "
                    "partitioned; the mask was built for a different resolution"
                )
            if mask.shape[-2:] != (N, N):
                raise ValueError(
                    f"attention mask has window size {tuple(mask.shape[-2:])} but the "
                    f"windows carry {N} tokens each; the mask was built for a "
                    "different window_size"
                )
            if B_ % nW != 0:
                raise ValueError(
                    f"attention mask describes {nW} windows but {B_} window-rows were "
                    f"partitioned; the mask was built for a different resolution"
                )
            # Fold the window axis into the head axis so one 4-D fused SDPA call sees a
            # (1, nW*nH, N, N) bias instead of a (B_, nH, N, N) score tensor.
            bias = (bias + mask.unsqueeze(1)).reshape(1, nW * self.num_heads, N, N)
            q, k, v = (
                t.reshape(B_ // nW, nW * self.num_heads, N, C // self.num_heads) for t in (q, k, v)
            )

        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.to(q.dtype), dropout_p=dropout_p, scale=self.scale
        )
        x = x.reshape(B_, self.num_heads, N, C // self.num_heads).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    """Swin Transformer Block."""

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
        norm_layer: nn.Module = nn.LayerNorm,
        input_resolution: tuple[int, int] | None = None,
        qk_scale: float | None = None,
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
            norm_layer (nn.Module): Description.
            input_resolution (Optional[tuple[int, int]]): Description.
            qk_scale (Optional[float]): Description.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Adjust window size if input resolution is provided and smaller
        if input_resolution is not None:
            if min(input_resolution) <= self.window_size:
                self.shift_size = 0
                self.window_size = min(input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_dropout,
            proj_drop=dropout,
            qk_scale=qk_scale,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dropout=dropout,
        )

        self.input_resolution = input_resolution
        # Derived state, not parameters: see ShiftedWindowMaskCache for why the
        # mask is deliberately absent from ``state_dict``.
        self._mask_cache = ShiftedWindowMaskCache()
        if self.input_resolution is not None:
            self._init_attn_mask()

    @property
    def attn_mask(self) -> torch.Tensor | None:
        """The mask for the *declared* ``input_resolution``, or ``None``.

        Introspection only, and a read of the cache rather than a second copy:
        ``forward`` resolves the mask for the resolution it is actually given
        (non-negotiable 17 -- one owner).
        """
        if self.shift_size <= 0 or not self.input_resolution:
            return None
        reference = self.attn.qkv.weight
        return self._resolve_attn_mask(
            *self.input_resolution, device=reference.device, dtype=reference.dtype
        )

    def _init_attn_mask(self):
        """Build the declared resolution's mask once, at construction time."""
        _ = self.attn_mask

    def _resolve_attn_mask(self, height: int, width: int, *, device, dtype) -> torch.Tensor:
        """The mask for the resolution actually being processed (#1345)."""
        return self._mask_cache.get(
            height,
            width,
            self.window_size,
            self.shift_size,
            device=device,
            dtype=dtype,
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Drop the ``attn_mask`` key written by checkpoints predating #1345.

        It was a persistent buffer sized for that run's resolution, so it
        resolution-locked the weights. Derived state, rebuilt on demand.
        """
        state_dict.pop(prefix + "attn_mask", None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self, x: torch.Tensor, input_resolution: tuple[int, int] | None = None
    ) -> torch.Tensor:
        # Support dynamic resolution if not set in init
        """forward.

        Args:
            x (torch.Tensor): Description.
            input_resolution (Optional[tuple[int, int]]): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SwinBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            input_resolution (tuple[int, int] | None): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = (
            input_resolution
            if input_resolution
            else (int(x.shape[1] ** 0.5), int(x.shape[1] ** 0.5))
        )
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad for window partition -- through the same rule the mask is built
        # on, so the two cannot drift apart.
        padded_h, padded_w = padded_resolution(H, W, self.window_size)
        pad_r = padded_w - W
        pad_b = padded_h - H
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # The resolution in hand -- not the constructed one, and not None
            # when none was declared. Both used to happen (#1345).
            attn_mask = self._resolve_attn_mask(H, W, device=x.device, dtype=x.dtype)
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA. The count is passed so the mask is validated against
        # the partition it is applied to, not a shape that happens to divide.
        attn_windows = self.attn(
            x_windows,
            mask=attn_mask,
            num_windows=window_count(H, W, self.window_size),
        )

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # Residual + FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
