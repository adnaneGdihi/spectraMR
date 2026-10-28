"""Enhanced Transformer Models with Architectural Stability Improvements
Addresses the stability issues found in the architectural analysis.
"""

import torch
import torch.nn.functional as F
from torch import nn


class EnhancedLayerNorm(nn.Module):
    """Enhanced LayerNorm with better initialization and stability."""

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        """__init__.

        Args:
            normalized_shape (Any): Description.
            eps (Any): Description.
            elementwise_affine (Any): Description.
        """
        super().__init__()

        # Ensure normalized_shape is a tuple/list
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for EnhancedLayerNorm.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class StableConvPatchEmbedding(nn.Module):
    """Patch embedding with proper normalization for stability."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            embed_dim (int): Description.
            patch_size (int): Description.
            use_batch_norm (bool): Description.
            use_layer_norm (bool): Description.
        """
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolutional patch embedding
        self.conv = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Add BatchNorm after Conv2d for stability (addresses stability issue
        # #1)
        self.batch_norm = nn.BatchNorm2d(embed_dim) if use_batch_norm else nn.Identity()

        # Additional LayerNorm for transformer compatibility
        self.layer_norm = EnhancedLayerNorm(embed_dim) if use_layer_norm else nn.Identity()

        # Activation
        self.activation = nn.GELU()

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # B, C, H, W -> B, embed_dim, H//patch_size, W//patch_size
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for StableConvPatchEmbedding.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.conv(x)

        # Apply BatchNorm for training stability
        x = self.batch_norm(x)
        x = self.activation(x)

        # Flatten spatial dimensions: B, embed_dim, H', W' -> B, H'*W',
        # embed_dim
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # B, N, embed_dim

        # Apply LayerNorm
        x = self.layer_norm(x)
        x = self.dropout(x)

        return x


class MultiHeadAttentionWithSkip(nn.Module):
    """Multi-head attention with skip connections and normalization."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """__init__.

        Args:
            embed_dim (int): Description.
            num_heads (int): Description.
            dropout (float): Description.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Normalization layers for stability
        self.qkv_norm = EnhancedLayerNorm(embed_dim * 3)
        self.output_norm = EnhancedLayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for MultiHeadAttentionWithSkip.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x)
        qkv = self.qkv_norm(qkv)  # Normalize QKV for stability
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention (Fused Flash Attention)
        x_attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.head_dim**-0.5,
        )

        # Apply attention to values
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)

        # Apply output normalization
        x_attn = self.output_norm(x_attn)

        return x_attn


class StableTransformerBlock(nn.Module):
    """Enhanced transformer block with skip connections and comprehensive normalization."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        """__init__.

        Args:
            embed_dim (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            dropout (float): Description.
            drop_path (float): Description.
        """
        super().__init__()

        # Pre-norm architecture
        self.norm1 = EnhancedLayerNorm(embed_dim)
        self.attn = MultiHeadAttentionWithSkip(embed_dim, num_heads, dropout)

        # Skip connection scaling (for very deep networks)
        self.skip_scale1 = nn.Parameter(torch.ones(1) * 0.1)

        # MLP block
        self.norm2 = EnhancedLayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Skip connection scaling
        self.skip_scale2 = nn.Parameter(torch.ones(1) * 0.1)

        # Drop path for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Additional stability normalization
        self.final_norm = EnhancedLayerNorm(embed_dim)

    def forward(self, x):
        # Attention with skip connection and scaling
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for StableTransformerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        attn_out = self.attn(self.norm1(x))
        x = x + self.drop_path(attn_out * self.skip_scale1)

        # MLP with skip connection and scaling
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.drop_path(mlp_out * self.skip_scale2)

        # Final normalization for stability
        x = self.final_norm(x)

        return x


class DropPath(nn.Module):
    """Drop path (Stochastic Depth) regularization."""

    def __init__(self, drop_prob: float = 0.0):
        """__init__.

        Args:
            drop_prob (float): Description.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for DropPath.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor

        return output


class StableViTWithSkipConnections(nn.Module):
    """Enhanced Vision Transformer with architectural stability improvements:
    - BatchNorm after Conv2d patch embedding
    - Skip connections throughout the network
    - Proper weight initialization
    - Enhanced normalization strategies
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        image_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            image_size (int): Description.
            patch_size (int): Description.
            embed_dim (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            dropout (float): Description.
            drop_path_rate (float): Description.
        """
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.depth = depth

        # Enhanced patch embedding with BatchNorm (fixes stability issue #1)
        self.patch_embed = StableConvPatchEmbedding(
            in_channels,
            embed_dim,
            patch_size,
            use_batch_norm=True,
            use_layer_norm=True,
        )

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        # Dropout
        self.pos_drop = nn.Dropout(dropout)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Enhanced transformer blocks with skip connections (fixes stability
        # issue #2)
        self.blocks = nn.ModuleList(
            [
                StableTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
                for i in range(depth)
            ],
        )

        # Global skip connections for very deep networks
        self.mid_skip_layers = nn.ModuleList(
            [
                nn.Sequential(
                    EnhancedLayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(depth // 4)  # Skip connections every 4 layers
            ],
        )

        # Final normalization
        self.norm = EnhancedLayerNorm(embed_dim)

        # Enhanced decoder with skip connections
        self.decoder = self._build_decoder(
            embed_dim,
            image_size,
            patch_size,
            out_channels,
            dropout,
        )
        self.output_activation = nn.Tanh()

        # Initialize weights properly (fixes stability issue #3)
        # Weight initialization will be handled by WeightInitializationManager

        # self._init_weights() - REMOVED to prevent over-initialization

    def _build_decoder(self, embed_dim, image_size, patch_size, out_channels, dropout):
        """Build decoder with skip connections."""
        decoder_dim = embed_dim * 2

        return nn.Sequential(
            # Expand features
            nn.Linear(embed_dim, decoder_dim),
            nn.GELU(),
            EnhancedLayerNorm(decoder_dim),
            nn.Dropout(dropout),
            # Intermediate layer with skip-like structure
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            EnhancedLayerNorm(decoder_dim),
            nn.Dropout(dropout),
            # Final projection
            nn.Linear(decoder_dim, patch_size * patch_size * out_channels),
        )

    def _init_weights(self):
        """Proper weight initialization to fix stability issue #3."""
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.conv.weight)
        if self.patch_embed.conv.bias is not None:
            nn.init.constant_(self.patch_embed.conv.bias, 0)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize transformer blocks
        for block in self.blocks:
            # Initialize attention layers
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.constant_(block.attn.proj.bias, 0)

            # Initialize MLP layers
            for layer in block.mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)

        # Initialize decoder
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for StableViTWithSkipConnections.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # B, N, embed_dim
        # N = x.shape[1]  # Actual number of patches

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # CRITICAL FIX: Adaptive positional embedding
        # Handle different input sizes by interpolating or truncating pos_embed
        expected_seq_len = x.shape[1]  # N + 1 (including class token)
        pos_embed_seq_len = self.pos_embed.shape[1]

        if expected_seq_len != pos_embed_seq_len:
            # Interpolate positional embeddings for different input sizes
            if expected_seq_len > pos_embed_seq_len:
                # Need to extend pos_embed - use last embedding repeated
                extra_embed = self.pos_embed[:, -1:, :].expand(
                    1,
                    expected_seq_len - pos_embed_seq_len,
                    -1,
                )
                pos_embed = torch.cat([self.pos_embed, extra_embed], dim=1)
            else:
                # Need to truncate pos_embed
                pos_embed = self.pos_embed[:, :expected_seq_len, :]
        else:
            pos_embed = self.pos_embed

        # Add positional embedding (now guaranteed to match)
        x = x + pos_embed
        x = self.pos_drop(x)

        # Store intermediate features for skip connections
        skip_connections = []

        # Pass through transformer blocks with global skip connections
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Add global skip connections every 4 layers
            if i % 4 == 3 and len(skip_connections) < len(self.mid_skip_layers):
                skip_idx = len(skip_connections)
                skip_feat = self.mid_skip_layers[skip_idx](x)
                skip_connections.append(skip_feat)

                # Add skip connection from earlier layer
                if len(skip_connections) > 1:
                    x = x + skip_connections[-2]

        # Final normalization
        x = self.norm(x)

        # Remove class token for reconstruction
        x = x[:, 1:]  # Remove CLS token

        # Decode to image
        x = self.decoder(x)

        # CRITICAL FIX: Calculate actual number of patches dynamically
        # x is already without class token after decoder
        actual_n_patches = x.shape[1]  # Actual number of patches

        # Calculate the original input dimensions from the patch embedding
        H_patches = int(actual_n_patches**0.5)
        W_patches = actual_n_patches // H_patches

        # Ensure we have a valid square grid (fallback for non-square inputs)
        if H_patches * W_patches != actual_n_patches:
            # For non-perfect squares, use the closest square dimensions
            H_patches = W_patches = int(actual_n_patches**0.5)
            actual_n_patches = H_patches * W_patches
            # Truncate to valid patches
            x = x[:, :actual_n_patches, :]

        # Get output channels from decoder
        decoder_output_dim = x.shape[-1]
        n_output_channels = decoder_output_dim // (self.patch_size * self.patch_size)

        # Reshape to image using actual patch count
        x = x.view(
            B,
            actual_n_patches,
            self.patch_size,
            self.patch_size,
            n_output_channels,
        )
        x = x.permute(0, 4, 1, 2, 3)  # B, C, N, P, P

        # Unpatchify with calculated grid size
        x = x.view(
            B,
            n_output_channels,
            H_patches,
            self.patch_size,
            W_patches,
            self.patch_size,
        )
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(
            B,
            n_output_channels,
            H_patches * self.patch_size,
            W_patches * self.patch_size,
        )

        # CRITICAL FIX: Apply Tanh to ensure output is in [-1, 1] range
        x = self.output_activation(x)

        return x


class StableSwinTransformerWithSkipConnections(nn.Module):
    """Enhanced Swin Transformer with architectural stability improvements:
    - BatchNorm after Conv2d patch embedding
    - Skip connections throughout the network
    - Proper weight initialization
    - Enhanced normalization strategies
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        image_size: int = 256,
        patch_size: int = 4,
        embed_dim: int = 96,
        depth: int = 6,
        num_heads: int = 3,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            image_size (int): Description.
            patch_size (int): Description.
            embed_dim (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            window_size (int): Description.
            mlp_ratio (float): Description.
            dropout (float): Description.
            drop_path_rate (float): Description.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = window_size
        self.depth = depth

        # Enhanced patch embedding with BatchNorm (fixes stability issue #1)
        self.patch_embed = StableConvPatchEmbedding(
            in_channels,
            embed_dim,
            patch_size,
            use_batch_norm=True,
            use_layer_norm=True,
        )

        # Patch resolution
        self.patch_resolution = [image_size // patch_size, image_size // patch_size]

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.patch_resolution[0] * self.patch_resolution[1],
                embed_dim,
            ),
        )
        self.pos_drop = nn.Dropout(dropout)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Enhanced Swin blocks with skip connections
        self.blocks = nn.ModuleList(
            [
                StableSwinTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ],
        )

        # Global skip connections (fixes stability issue #2)
        self.global_skip_connections = nn.ModuleList(
            [
                nn.Sequential(
                    EnhancedLayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(depth // 3)  # Skip connections every 3 layers
            ],
        )

        # Final normalization
        self.norm = EnhancedLayerNorm(embed_dim)

        # Enhanced decoder
        self.decoder = self._build_decoder(
            embed_dim,
            image_size,
            patch_size,
            out_channels,
            dropout,
        )
        self.output_activation = nn.Tanh()

        # Initialize weights (fixes stability issue #3)
        # Weight initialization will be handled by WeightInitializationManager

        # self._init_weights() - REMOVED to prevent over-initialization

    def _build_decoder(self, embed_dim, image_size, patch_size, out_channels, dropout):
        """Build enhanced decoder."""
        return nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            EnhancedLayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * 4),
            nn.GELU(),
            EnhancedLayerNorm(embed_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, patch_size * patch_size * out_channels),
        )

    def _init_weights(self):
        """Proper weight initialization."""
        # Initialize patch embedding
        nn.init.xavier_uniform_(self.patch_embed.conv.weight)
        if self.patch_embed.conv.bias is not None:
            nn.init.constant_(self.patch_embed.conv.bias, 0)

        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize blocks
        for _block in self.blocks:
            # Weight initialization will be handled by
            # WeightInitializationManager
            pass  # block._init_weights() - REMOVED to prevent over-initialization

        # Initialize decoder
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for StableSwinTransformerWithSkipConnections.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # B, N, C

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Store skip connections
        skip_features = []

        # Pass through Swin blocks with global skip connections
        for i, block in enumerate(self.blocks):
            x = block(x, self.patch_resolution)

            # Add global skip connections every 3 layers
            if i % 3 == 2 and len(skip_features) < len(self.global_skip_connections):
                skip_idx = len(skip_features)
                skip_feat = self.global_skip_connections[skip_idx](x)
                skip_features.append(skip_feat)

                # Add skip connection from earlier layer
                if len(skip_features) > 1:
                    x = x + skip_features[-2]

        # Final normalization
        x = self.norm(x)

        # Decode
        x = self.decoder(x)

        # Reshape to image
        H, W = self.patch_resolution
        x = x.view(B, H * W, self.patch_size, self.patch_size, -1)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, N, P, P

        # Unpatchify
        x = x.view(B, -1, H, self.patch_size, W, self.patch_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(B, -1, H * self.patch_size, W * self.patch_size)

        # CRITICAL FIX: Apply Tanh to ensure output is in [-1, 1] range
        x = self.output_activation(x)

        return x


class StableSwinTransformerBlock(nn.Module):
    """Enhanced Swin Transformer Block with stability improvements."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path=0.0,
    ):
        """__init__.

        Args:
            dim (Any): Description.
            num_heads (Any): Description.
            window_size (Any): Description.
            shift_size (Any): Description.
            mlp_ratio (Any): Description.
            dropout (Any): Description.
            drop_path (Any): Description.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Normalization layers
        self.norm1 = EnhancedLayerNorm(dim)
        self.norm2 = EnhancedLayerNorm(dim)

        # Window-based multi-head self-attention
        self.attn = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            dropout=dropout,
        )

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Skip connection scaling
        self.skip_scale1 = nn.Parameter(torch.ones(1) * 0.1)
        self.skip_scale2 = nn.Parameter(torch.ones(1) * 0.1)

    def _init_weights(self):
        """Initialize weights for stability."""
        # Initialize attention
        nn.init.xavier_uniform_(self.attn.qkv.weight)
        nn.init.xavier_uniform_(self.attn.proj.weight)
        nn.init.constant_(self.attn.proj.bias, 0)

        # Initialize MLP
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, patch_resolution):
        """forward.

        Args:
            x (Any): Description.
            patch_resolution (Any): Description.
        Returns:
            Any: Description.

        forward method for StableSwinTransformerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            patch_resolution (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = patch_resolution
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
        x_windows, x_padding_info = window_partition(
            shifted_x,
            self.window_size,
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1,
            self.window_size * self.window_size,
            C,
        )  # nW*B, window_size*window_size, C

        # Window attention
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            H,
            W,
            x_padding_info,
        )  # B H' W' C

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

        # First residual connection with scaling
        x = shortcut + self.drop_path(x * self.skip_scale1)

        # MLP with second residual connection
        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut2 + self.drop_path(x * self.skip_scale2)

        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention."""

    def __init__(self, dim, window_size, num_heads, dropout=0.1):
        """__init__.

        Args:
            dim (Any): Description.
            window_size (Any): Description.
            num_heads (Any): Description.
            dropout (Any): Description.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Enhanced normalization
        self.qkv_norm = EnhancedLayerNorm(dim * 3)

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for WindowAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = self.qkv_norm(qkv)  # Normalize for stability
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(
            2,
            0,
            3,
            1,
            4,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=getattr(self.dropout, "p", 0.0) if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def window_partition(x, window_size):
    """Partition into non-overlapping windows with padding info."""
    B, H, W, C = x.shape
    # Pad H/W if needed for divisibility
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # Store original dimensions for reverse operation
    padding_info = {"original_H": H, "original_W": W, "pad_h": pad_h, "pad_w": pad_w}

    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        _, H, W, _ = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, padding_info


def window_reverse(windows, window_size, H, W, padding_info=None):
    """Reverse window partition with padding info support."""
    if padding_info is not None:
        # Use provided padding info
        H_pad = padding_info["original_H"] + padding_info["pad_h"]
        W_pad = padding_info["original_W"] + padding_info["pad_w"]
        num_windows = (H_pad // window_size) * (W_pad // window_size)
        B = windows.shape[0] // num_windows if num_windows > 0 else 0

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
    else:
        # Fallback to original logic
        # Fallback to original logic
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        H_pad, W_pad = H + pad_h, W + pad_w
        num_windows = (H_pad // window_size) * (W_pad // window_size)
        B = windows.shape[0] // num_windows if num_windows > 0 else 0

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
