"""Hybrid Transformer Diffusion Generator
=====================================

This module implements a hybrid Vision Transformer (ViT) and Swin Transformer
diffusion model for medical image super-resolution.

Key Features:
- Combines global attention (ViT) and local windowed attention (Swin)
- U-Net backbone with skip connections
- Time-conditioned diffusion process
- Optimized for medical imaging applications
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from spectramr.models.blocks.block_registry import create_block
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.reconstruction.unet import SinusoidalPositionEmbeddings
from spectramr.models.registry import register_model

# --- Utility Functions ---


def window_partition(x, window_size):
    """Partition tensor into non-overlapping windows with padding info."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # Store original dimensions for reverse operation
    padding_info = {"original_H": H, "original_W": W, "pad_h": pad_h, "pad_w": pad_w}

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    _, H_pad, W_pad, _ = x.shape
    x = x.view(
        B,
        H_pad // window_size,
        window_size,
        W_pad // window_size,
        window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, padding_info


def window_reverse(windows, window_size, H, W, padding_info=None):
    """Reverse window partition with padding info support."""
    if padding_info is not None:
        # Use provided padding info
        H_pad = padding_info["original_H"] + padding_info["pad_h"]
        W_pad = padding_info["original_W"] + padding_info["pad_w"]
        num_windows = (H_pad // window_size) * (W_pad // window_size)
        B = windows.shape[0] // num_windows
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
    # Fallback to original logic
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    H_pad, W_pad = H + pad_h, W + pad_w
    num_windows = (H_pad // window_size) * (W_pad // window_size)
    B = windows.shape[0] // num_windows
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


# --- Component Classes ---


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention for Swin Transformers."""

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
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                num_heads,
            ),
        )

        # Generate relative position coordinates
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        # Attention layers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """forward.

        Args:
            x (Any): Description.
            mask (Any): Description.
        Returns:
            Any: Description.

        forward method for WindowAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

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
        q, k, v = qkv.unbind(0)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
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
        return self.proj_drop(self.proj(x))


# --- Transformer Blocks ---


class ViTBlock(nn.Module):
    """Vision Transformer block with multi-head attention and feed-forward."""

    def __init__(
        self,
        dim,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.1,
        layer_norm_strategy="pre",
        layer_norm_eps=1e-6,
    ):
        """__init__.

        Args:
            dim (Any): Description.
            heads (Any): Description.
            dim_head (Any): Description.
            mlp_dim (Any): Description.
            dropout (Any): Description.
            layer_norm_strategy (Any): Description.
            layer_norm_eps (Any): Description.
        """
        super().__init__()
        # Import attention and feed-forward from centralized components
        from ..transformer.components import FeedForward, MultiHeadSelfAttention

        self.attention = MultiHeadSelfAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
        )
        self.ff = FeedForward(embed_dim=dim, hidden_dim=mlp_dim, dropout=dropout)
        self.norm1 = create_block(
            "layer_norm",
            channels=dim,
            eps=layer_norm_eps,
            strategy=layer_norm_strategy,
        )
        self.norm2 = create_block(
            "layer_norm",
            channels=dim,
            eps=layer_norm_eps,
            strategy=layer_norm_strategy,
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for ViTBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.norm1(x, self.attention)
        x = self.norm2(x, self.ff)
        return x


class HybridSwinBlock(nn.Module):
    """Swin Transformer block with windowed attention."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        dropout=0.0,
        layer_norm_strategy="pre",
        layer_norm_eps=1e-6,
    ):
        """__init__.

        Args:
            dim (Any): Description.
            input_resolution (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
            shift_size (Any): Description.
            mlp_ratio (Any): Description.
            dropout (Any): Description.
            layer_norm_strategy (Any): Description.
            layer_norm_eps (Any): Description.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # Import feed-forward from centralized components
        from ..transformer.components import FeedForward

        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            attn_drop=dropout,
        )
        self.ffn = FeedForward(
            embed_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=dropout,
        )
        self.norm1 = create_block(
            "layer_norm",
            channels=dim,
            eps=layer_norm_eps,
            strategy=layer_norm_strategy,
        )
        self.norm2 = create_block(
            "layer_norm",
            channels=dim,
            eps=layer_norm_eps,
            strategy=layer_norm_strategy,
        )

        # Create attention mask for shifted windows
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
            self.register_buffer(
                "attn_mask",
                attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
                    attn_mask == 0,
                    0.0,
                ),
            )
        else:
            self.attn_mask = None

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for HybridSwinBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x

        x = self.norm1.norm(x)
        x = x.view(B, H, W, C)

        # Apply window shift if needed
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        # Window partition and attention
        x_windows, x_padding_info = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse(
            attn_windows.view(-1, self.window_size, self.window_size, C),
            self.window_size,
            H,
            W,
            x_padding_info,
        )

        # Reverse window shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x

        x = shortcut + x.view(B, L, C)
        x = x + self.ffn(self.norm2.norm(x))
        return x


# --- Main Generator Model ---


@register_model(name="hybrid_transformer_diffusion", training_mode="diffusion")
class HybridTransformerDiffusionGenerator(nn.Module, IGenerator):
    """Hybrid ViT-Swin Diffusion Generator.
    Combines global attention (ViT) and windowed attention (Swin) in a
    diffusion framework.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_dim=256,
        image_size=256,
        patch_size=16,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        window_size=8,
        dropout=0.1,
        bilinear=False,
        layer_norm_strategy="pre",
        layer_norm_eps=1e-6,
        T=1000,
        beta_start=1e-4,
        beta_end=2e-2,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            time_dim (Any): Description.
            image_size (Any): Description.
            patch_size (Any): Description.
            dim (Any): Description.
            depth (Any): Description.
            heads (Any): Description.
            mlp_dim (Any): Description.
            window_size (Any): Description.
            dropout (Any): Description.
            bilinear (Any): Description.
            layer_norm_strategy (Any): Description.
            layer_norm_eps (Any): Description.
            T (Any): Description.
            beta_start (Any): Description.
            beta_end (Any): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.T = T

        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size**2
        patches_resolution = (image_size // patch_size, image_size // patch_size)

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, dim),
        )

        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim, eps=layer_norm_eps),
        )

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Hybrid Transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for i in range(depth):
            if i % 2 == 0:
                # ViT block for global attention
                self.transformer_blocks.append(
                    ViTBlock(
                        dim,
                        heads,
                        dim // heads,
                        mlp_dim,
                        dropout,
                        layer_norm_strategy=layer_norm_strategy,
                        layer_norm_eps=layer_norm_eps,
                    ),
                )
            else:
                # Swin block for local, windowed attention
                self.transformer_blocks.append(
                    HybridSwinBlock(
                        dim,
                        input_resolution=patches_resolution,
                        num_heads=heads,
                        window_size=window_size,
                        shift_size=window_size // 2 if (i // 2) % 2 == 1 else 0,
                        mlp_ratio=mlp_dim / dim,
                        dropout=dropout,
                        layer_norm_strategy=layer_norm_strategy,
                        layer_norm_eps=layer_norm_eps,
                    ),
                )

        # Score prediction network
        self.score_net = nn.Sequential(
            nn.LayerNorm(dim, eps=layer_norm_eps),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, patch_dim),
        )

        # U-Net components
        self.inc = create_block("unet_double_conv", in_channels=in_channels, out_channels=64)
        self.down1 = create_block("unet_down", in_channels=64, out_channels=128)
        self.down2 = create_block("unet_down", in_channels=128, out_channels=256)
        self.down3 = create_block("unet_down", in_channels=256, out_channels=512)
        self.up1 = create_block("unet_up", in_channels=512, out_channels=256, bilinear=bilinear)
        self.up2 = create_block("unet_up", in_channels=256, out_channels=128, bilinear=bilinear)
        self.up3 = create_block("unet_up", in_channels=128, out_channels=64, bilinear=bilinear)
        self.outc = create_block("unet_out_conv", in_channels=64, out_channels=out_channels)
        self.output_activation = nn.Identity()

        # Diffusion parameters
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    @property
    def name(self) -> str:
        """Return the model name."""
        return "HybridTransformerDiffusionGenerator"

    def get_parameter_count(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return the output shape for given input shape."""
        return input_shape

    def generate(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generate output using the diffusion model."""
        return self.forward(x, t)

    def chi_square_noise(self, shape, degrees_of_freedom=2, device=None):
        """Generate chi-square noise."""
        if device is None:
            device = self.betas.device
        gamma_noise = (
            torch.distributions.Gamma(degrees_of_freedom / 2, 1.0).sample(shape).to(device)
        )
        return (gamma_noise - degrees_of_freedom) / math.sqrt(2 * degrees_of_freedom)

    def forward(self, x, t):
        """forward.

        Args:
            x (Any): Description.
            t (Any): Description.
        Returns:
            Any: Description.

        forward method for HybridTransformerDiffusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        b, c, h, w = x.shape

        # Time embedding
        time_emb = self.time_embed(t)

        # U-Net encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Hybrid Transformer processing
        scores = torch.zeros_like(x)
        if h >= self.patch_size and w >= self.patch_size:
            x_vit = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            patches = self.to_patch_embedding(x_vit)
            b_patch, n, _ = patches.shape

            cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b_patch)
            patches_with_cls = torch.cat((cls_tokens, patches), dim=1)
            patches_with_cls += self.pos_embedding[:, : (n + 1)]
            patches_with_cls = self.dropout(patches_with_cls)

            time_emb_expanded = time_emb.unsqueeze(1).expand(
                -1,
                patches_with_cls.size(1),
                -1,
            )
            patches_with_cls += time_emb_expanded

            # Process through transformer blocks
            processed_patches = patches_with_cls
            for block in self.transformer_blocks:
                if isinstance(block, HybridSwinBlock):
                    # Swin block operates on flattened patches
                    # without CLS token
                    cls_token, patch_tokens = (
                        processed_patches[:, :1],
                        processed_patches[:, 1:],
                    )
                    patch_tokens = block(patch_tokens)
                    processed_patches = torch.cat((cls_token, patch_tokens), dim=1)
                else:  # ViTBlock
                    processed_patches = block(processed_patches)

            patch_features = processed_patches[:, 1:]
            scores = self.score_net(patch_features)
            scores = rearrange(
                scores,
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=self.image_size // self.patch_size,
                w=self.image_size // self.patch_size,
                p1=self.patch_size,
                p2=self.patch_size,
                c=c,
            )
            scores = F.interpolate(
                scores,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        # U-Net decoder path
        x_up = self.up1(x4, x3)
        x_up = self.up2(x_up, x2)
        x_up = self.up3(x_up, x1)
        unet_output = self.outc(x_up)

        # Combine scores and return
        final_output = scores + unet_output
        return self.output_activation(final_output)

    def sample(self, shape, device, steps=None, eta=1.0):
        """Generate samples using Langevin dynamics."""
        if steps is None:
            steps = self.T
        x = self.chi_square_noise(shape).to(device)
        step_size = 1.0 / steps

        for i in reversed(range(steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            with torch.no_grad():
                score = self.forward(x, t)
            if i > 0:
                noise = self.chi_square_noise(x.shape).to(device)
                x = x + step_size * score + eta * math.sqrt(step_size) * noise
            else:
                x = x + step_size * score
        return x


# --- Factory Function ---


def create_hybrid_transformer_diffusion_generator(
    in_channels=1,
    out_channels=1,
    time_dim=256,
    layer_norm_strategy="pre",
    layer_norm_eps=1e-6,
    **kwargs,
):
    """Factory function to create the Hybrid Transformer Diffusion Generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        time_dim: Time embedding dimension
        layer_norm_strategy: Layer normalization strategy ("pre" or "post")
        layer_norm_eps: Layer normalization epsilon
        **kwargs: Additional arguments for the generator

    Returns:
        HybridTransformerDiffusionGenerator: The configured generator

    """
    return HybridTransformerDiffusionGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        time_dim=time_dim,
        layer_norm_strategy=layer_norm_strategy,
        layer_norm_eps=layer_norm_eps,
        **kwargs,
    )


__all__ = [
    "HybridSwinBlock",
    "HybridTransformerDiffusionGenerator",
    "ViTBlock",
    "WindowAttention",
    "create_hybrid_transformer_diffusion_generator",
]
