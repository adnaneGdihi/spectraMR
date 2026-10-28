#!/usr/bin/env python3
"""Transformer Components Module
====================

This module contains transformer-related components following SOLID principles:
- PatchEmbedding: Patch embedding for ViT
- MultiHeadSelfAttention: Multi-head self-attention mechanism
- FeedForward: Feed-forward network for transformer
- KANFeedForward: KAN-based feed-forward network
- TransformerBlock: Standard transformer block
- TransformerBlockWithKAN: Transformer block with KAN
- StandardViT: Vision Transformer implementation
- StandardViTWithKAN: ViT with KAN layers
"""

import torch
import torch.nn.functional as F
from torch import nn

# Import the KAN layer
from spectramr.models.layers.kan.kan_convs.kans import FastKANLayer

FastKANLinearLayer = FastKANLayer  # Alias for backward compatibility


from spectramr.models.blocks.embeddings import PatchEmbedding


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        """__init__.

        Args:
            embed_dim (int): Description.
            num_heads (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for MultiHeadSelfAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=getattr(self.attn_dropout, "p", 0.0) if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class FeedForward(nn.Module):
    """Feed-forward network for transformer"""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 3072,
        dropout: float = 0.0,
    ):
        """__init__.

        Args:
            embed_dim (int): Description.
            hidden_dim (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for FeedForward.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class KANFeedForward(nn.Module):
    """Feed-forward network for transformers using KAN layers."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        """__init__.

        Args:
            embed_dim (int): Description.
            hidden_dim (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        # NOTE: Using FastKANLinearLayer, which is assumed to exist.
        self.fc1 = FastKANLinearLayer(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = FastKANLinearLayer(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KANFeedForward.

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


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward"""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """__init__.

        Args:
            embed_dim (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm design
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TransformerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBlockWithKAN(nn.Module):
    """Transformer block with a KAN-based feed-forward network."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        """__init__.

        Args:
            embed_dim (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = KANFeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm design
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TransformerBlockWithKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class StandardViT(nn.Module):
    """Standard Vision Transformer implementation"""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """__init__.

        Args:
            img_size (int): Description.
            patch_size (int): Description.
            in_channels (int): Description.
            out_channels (int): Description.
            embed_dim (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            dropout (float): Description.
        """
        super().__init__()
        if out_channels != 1:
            raise NotImplementedError(
                f"StandardViT only supports out_channels=1; got {out_channels}. "
                "The decoder head and forward reshape are single-channel; "
                "multi-channel output is not yet wired."
            )
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)],
        )

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder head for image reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, img_size * img_size),
        )
        self.output_activation = nn.Tanh()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize positional embeddings
        """init_weights.

        Returns:
            Any: Description.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for StandardViT.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Interpolate positional embedding to match input size
        pos_embed = self.pos_embed
        if pos_embed.size(1) != x.size(1):
            pos_embed = F.interpolate(
                pos_embed.transpose(1, 2),
                size=x.size(1),
                mode="linear",
            ).transpose(1, 2)

        # Add positional embedding
        x = x + pos_embed
        x = self.pos_dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.norm(x)

        # Use class token for reconstruction
        cls_output = x[:, 0]

        # Decode to image
        output = self.decoder(cls_output)
        # The decoder outputs img_size * img_size, but we need H * W
        # Reshape to img_size x img_size first, then interpolate to H x W
        output = output.view(B, 1, self.img_size, self.img_size)

        # Interpolate to match input dimensions if necessary
        if output.shape[2] != H or output.shape[3] != W:
            output = torch.nn.functional.interpolate(
                output,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        # Apply tanh activation to ensure output is in [-1, 1] range
        return self.output_activation(output)

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "StandardViT"

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def to_device(self, device: torch.device) -> "StandardViT":
        """Moves the model to the specified device."""
        return self.to(device)


class StandardViTWithKAN(nn.Module):
    """Standard Vision Transformer with KAN layers in the encoder."""

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 1,
        out_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """__init__.

        Args:
            img_size (int): Description.
            patch_size (int): Description.
            in_channels (int): Description.
            out_channels (int): Description.
            embed_dim (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            dropout (float): Description.
        """
        super().__init__()
        if out_channels != 1:
            raise NotImplementedError(
                f"StandardViTWithKAN only supports out_channels=1; got {out_channels}. "
                "The decoder head and forward reshape are single-channel; "
                "multi-channel output is not yet wired."
            )
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Use Transformer blocks with KAN
        self.blocks = nn.ModuleList(
            [
                TransformerBlockWithKAN(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ],
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, img_size * img_size),
        )
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for StandardViTWithKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Interpolate positional embedding to match input size
        pos_embed = self.pos_embed
        if pos_embed.size(1) != x.size(1):
            pos_embed = F.interpolate(
                pos_embed.transpose(1, 2),
                size=x.size(1),
                mode="linear",
            ).transpose(1, 2)

        x = x + pos_embed
        x = self.pos_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]
        output = self.decoder(cls_output)
        # The decoder outputs img_size * img_size, but we need H * W
        # Reshape to img_size x img_size first, then interpolate to H x W
        output = output.view(B, 1, self.img_size, self.img_size)

        # Interpolate to match input dimensions if necessary
        if output.shape[2] != H or output.shape[3] != W:
            output = torch.nn.functional.interpolate(
                output,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        return self.output_activation(output)

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "StandardViTWithKAN"

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def to_device(self, device: torch.device) -> "StandardViTWithKAN":
        """Moves the model to the specified device."""
        return self.to(device)
