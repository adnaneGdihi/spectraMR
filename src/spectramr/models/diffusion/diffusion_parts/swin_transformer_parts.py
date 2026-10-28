import torch
import torch.nn.functional as F
from torch import nn


class EnhancedLayerNorm(nn.Module):
    """Enhanced Layer Normalization with configurable strategies for Swin Transformer models.

    Supports:
    - Pre-normalization: LayerNorm → Attention/FFN → Residual
    - Post-normalization: Attention/FFN → LayerNorm → Residual
    - Sandwich normalization: LayerNorm → Attention/FFN → LayerNorm → Residual
    """

    def __init__(self, dim, eps=1e-6, elementwise_affine=True, strategy="pre"):
        """__init__.

        Args:
            dim (Any): Description.
            eps (Any): Description.
            elementwise_affine (Any): Description.
            strategy (Any): Description.
        """
        super().__init__()
        self.strategy = strategy
        self.eps = eps

        if strategy == "sandwich":
            self.norm_pre = nn.LayerNorm(
                dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
            self.norm_post = nn.LayerNorm(
                dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )
        else:
            self.norm = nn.LayerNorm(
                dim,
                eps=eps,
                elementwise_affine=elementwise_affine,
            )

    def forward(self, x, layer_fn=None):
        """Apply layer normalization with the specified strategy.

        Args:
            x: Input tensor
            layer_fn: Layer function (attention or FFN) to apply

        Returns:
            Normalized output

        forward method for EnhancedLayerNorm.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            layer_fn (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if layer_fn is None:
            # Just apply normalization
            if self.strategy == "sandwich":
                return self.norm_post(self.norm_pre(x))
            return self.norm(x)

        if self.strategy == "pre":
            # Pre-normalization: LayerNorm → Layer → Residual
            return x + layer_fn(self.norm(x))
        if self.strategy == "post":
            # Post-normalization: Layer → LayerNorm → Residual
            return x + self.norm(layer_fn(x))
        if self.strategy == "sandwich":
            # Sandwich: LayerNorm → Layer → LayerNorm → Residual
            normalized = self.norm_pre(x)
            output = layer_fn(normalized)
            return x + self.norm_post(output)
        raise ValueError(f"Unknown normalization strategy: {self.strategy}")


def window_partition(x, window_size):
    """Partition input tensor into non-overlapping windows with padding info.

    Args:
        x: (B, H, W, C)
        window_size: Window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
        padding_info: dict with padding information

    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # Store original dimensions for reverse operation
    padding_info = {"original_H": H, "original_W": W, "pad_h": pad_h, "pad_w": pad_w}

    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, H, W, _ = x.shape
    # Always reshape into window grid before permuting
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, padding_info


def window_reverse(windows, window_size, H, W, padding_info=None):
    """Reverse window partition with padding info support.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
        padding_info: Optional dict with padding information

    Returns:
        x: (B, H, W, C)

    """
    if padding_info is not None:
        # Use provided padding info
        H_pad = padding_info["original_H"] + padding_info["pad_h"]
        W_pad = padding_info["original_W"] + padding_info["pad_w"]
        num_windows = (H_pad // window_size) * (W_pad // window_size)
        B = windows.shape[0] // num_windows if num_windows > 0 else 0
        try:
            x = windows.view(
                B,
                H_pad // window_size,
                W_pad // window_size,
                window_size,
                window_size,
                -1,
            )
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
            if padding_info["pad_h"] > 0 or padding_info["pad_w"] > 0:
                x = x[
                    :,
                    : padding_info["original_H"],
                    : padding_info["original_W"],
                    :,
                ].contiguous()
            return x
        except RuntimeError:
            return windows.view(B, H, W, -1)
    else:
        # Fallback to original logic
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        H_pad, W_pad = H + pad_h, W + pad_w
        num_windows = (H_pad // window_size) * (W_pad // window_size)
        B = windows.shape[0] // num_windows if num_windows > 0 else 0
        try:
            x = windows.view(
                B,
                H_pad // window_size,
                W_pad // window_size,
                window_size,
                window_size,
                -1,
            )
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_pad, W_pad, -1)
            if pad_h > 0 or pad_w > 0:
                x = x[:, :H, :W, :].contiguous()
            return x
        except RuntimeError:
            return windows.view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window-based multi-head self attention (W-MSA) module with relative position bias.

    .. deprecated::
        For new code, prefer importing from the canonical location:
        ``from spectramr.models.blocks.swin import WindowAttention``

        This version is retained for backward compatibility as it uses tuple window_size
        and has padding-aware window_partition/window_reverse functions.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        """__init__.

        Args:
            dim (Any): Description.
            window_size (Any): Description.
            num_heads (Any): Description.
            qkv_bias (Any): Description.
            attn_drop (Any): Description.
            proj_drop (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads),
        )

        # Get pair-wise relative position index for each token inside the
        # window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: torch.Tensor | None = None):
        """forward.

        Args:
            x (Any): Description.
            mask (Optional[torch.Tensor]): Description.
        Returns:
            Any: Description.

        forward method for WindowAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # (1, nH, N, N)

        if mask is not None:
            nW = mask.shape[0]
            # Fold the window axis into the head axis: one 4-D fused SDPA call, bias (1, nW*nH, N, N).
            bias = (bias + mask.unsqueeze(1)).reshape(1, nW * self.num_heads, N, N)
            q, k, v = (t.reshape(B_ // nW, nW * self.num_heads, N, -1) for t in (q, k, v))

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.to(q.dtype),
            dropout_p=getattr(self.attn_drop, "p", 0.0) if self.training else 0.0,
            scale=self.scale,
        )
        x = x.reshape(B_, self.num_heads, N, -1).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with window-based and shifted window-based self-attention."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_norm_strategy="pre",
        layer_norm_eps=1e-6,
        use_enhanced_layer_norm=True,
    ):
        """__init__.

        Args:
            dim (Any): Description.
            input_resolution (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
            shift_size (Any): Description.
            mlp_ratio (Any): Description.
            qkv_bias (Any): Description.
            drop (Any): Description.
            attn_drop (Any): Description.
            drop_path (Any): Description.
            norm_layer (Any): Description.
            layer_norm_strategy (Any): Description.
            layer_norm_eps (Any): Description.
            use_enhanced_layer_norm (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # Enhanced layer normalization
        if use_enhanced_layer_norm:
            self.norm1 = EnhancedLayerNorm(
                dim,
                eps=layer_norm_eps,
                strategy=layer_norm_strategy,
            )
            self.norm2 = EnhancedLayerNorm(
                dim,
                eps=layer_norm_eps,
                strategy=layer_norm_strategy,
            )
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)

        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows, mask_padding_info = window_partition(
                img_mask,
                self.window_size,
            )
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
                attn_mask == 0,
                0.0,
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for SwinTransformerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        # Partition windows
        x_windows, x_padding_info = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, x_padding_info)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer for hierarchical feature maps."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        """__init__.

        Args:
            input_resolution (Any): Description.
            dim (Any): Description.
            norm_layer (Any): Description.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for PatchMerging.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x
