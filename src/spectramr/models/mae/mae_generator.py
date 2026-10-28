"""Masked Autoencoder (MAE) for MRI Super-Resolution
=================================================

Implementation of Masked Autoencoder for self-supervised pretraining
on MRI volumes. Adapted for 3D medical imaging with patch-based
masking strategy.

Architecture:
- 3D patch embedding for volumetric data
- ViT-based encoder processing visible patches only
- Asymmetric decoder reconstructing full volume
- Random patch masking during pretraining
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding for volumetric data."""

    def __init__(
        self,
        img_size: tuple[int, int, int] = (64, 64, 64),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        """__init__.

        Args:
            img_size (tuple[int, int, int]): Description.
            patch_size (tuple[int, int, int]): Description.
            in_channels (int): Description.
            embed_dim (int): Description.
        """
        super().__init__()
        # Normalize scalars/lists to 3-tuples
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            img_size = (img_size[0], img_size[1], 1)
        img_size = tuple(img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
            patch_size = (patch_size[0], patch_size[1], 1)
        patch_size = tuple(patch_size)
        # [FIX] Clamp depth patch to img depth when img is 2D (depth=1)
        # Prevents Conv3d kernel exceeding input depth dimension
        if img_size[2] == 1 and patch_size[2] > 1:
            patch_size = (patch_size[0], patch_size[1], 1)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[i] // patch_size[i] for i in range(3)]
        self.num_patches = (
            self.patches_resolution[0] * self.patches_resolution[1] * self.patches_resolution[2]
        )

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed 3D patches. Handles both 4D [B,C,H,W] and 5D [B,C,H,W,D] inputs.

        forward method for PatchEmbed3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Auto-expand 2D [B,C,H,W] → 3D [B,C,H,W,1] for Conv3d compatibility
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        B, C, H, W, D = x.shape
        x = self.proj(x)  # (B, embed_dim, H//p, W//p, D//p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention for 3D patches."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            qkv_bias (bool): Description.
            qk_scale (Optional[float]): Description.
            attn_drop (float): Description.
            proj_drop (float): Description.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for Attention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=getattr(self.attn_drop, "p", 0.0) if self.training else 0.0,
            scale=self.scale,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth / Drop Connect per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        """__init__.

        Args:
            drop_prob (float): Description.
            scale_by_keep (bool): Description.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

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
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor


class MaeTransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            qkv_bias (bool): Description.
            qk_scale (Optional[float]): Description.
            drop (float): Description.
            attn_drop (float): Description.
            drop_path (float): Description.
            act_layer (nn.Module): Description.
            norm_layer (nn.Module): Description.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for MaeTransformerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder for visible patches."""

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_patches: int = 512,
    ):
        """__init__.

        Args:
            embed_dim (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            num_patches (int): Description.
        """
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        self.blocks = nn.ModuleList(
            [
                MaeTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                )
                for _ in range(depth)
            ],
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Weight initialization handled by centralized weight_initialization module
        """initialize_weights.

        Returns:
            Any: Description.
        """
        pass

    def _init_weights(self, m):
        # Weight initialization handled by centralized weight_initialization module
        """_init_weights.

        Args:
            m (Any): Description.
        Returns:
            Any: Description.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ViTEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, C = x.shape

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add position embedding (slice to match sequence length)
        seq_len = x.shape[1]
        pos_embed = self.pos_embed[:, :seq_len, :]
        x = x + pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class MAEEncoder(nn.Module):
    """MAE Encoder that processes only visible patches."""

    def __init__(
        self,
        img_size: tuple[int, int, int] = (64, 64, 64),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        """__init__.

        Args:
            img_size (tuple[int, int, int]): Description.
            patch_size (tuple[int, int, int]): Description.
            in_channels (int): Description.
            embed_dim (int): Description.
            depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
        """
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.encoder = ViTEncoder(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_patches=self.patch_embed.num_patches,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with masking.

        forward method for MAEEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Embed all patches
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Keep only visible patches
        x = x[~mask]  # Remove masked patches
        x = x.view(x.shape[0] // (mask.shape[0] // x.shape[0]), -1, x.shape[-1])

        # Encode visible patches
        x = self.encoder(x)
        return x


@register_model(
    name="mae_mri",
    training_mode="ssl",
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class MAEGenerator(nn.Module, IGenerator):
    """Masked Autoencoder Generator for MRI volumes.

    During pretraining: Takes masked input, reconstructs full volume
    During finetuning: Can be used as feature extractor for SR tasks
    """

    def __init__(
        self,
        img_size: tuple[int, int, int] = (64, 64, 64),
        patch_size: tuple[int, int, int] = (16, 16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        decoder_depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        **kwargs,
    ):
        """__init__.

        Args:
            img_size (tuple[int, int, int]): Description.
            patch_size (tuple[int, int, int]): Description.
            in_channels (int): Description.
            embed_dim (int): Description.
            encoder_depth (int): Description.
            decoder_depth (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
            mask_ratio (float): Description.
        """
        super().__init__()
        # Normalize scalars/lists to 3-tuples
        if isinstance(img_size, int):
            img_size = (img_size, img_size, img_size)
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            img_size = (img_size[0], img_size[1], 1)
        img_size = tuple(img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        elif isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
            patch_size = (patch_size[0], patch_size[1], 1)
        patch_size = tuple(patch_size)
        # [FIX] Clamp depth patch to img depth when img is 2D (depth=1)
        if img_size[2] == 1 and patch_size[2] > 1:
            patch_size = (patch_size[0], patch_size[1], 1)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # Encoder
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.encoder.patch_embed.num_patches + 1, embed_dim),
        )

        self.decoder_blocks = nn.ModuleList(
            [
                MaeTransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0,
                )
                for _ in range(decoder_depth)
            ],
        )

        self.decoder_norm = nn.LayerNorm(embed_dim)
        patch_size = self.encoder.patch_embed.patch_size
        self.decoder_pred = nn.Linear(
            embed_dim,
            patch_size[0] * patch_size[1] * patch_size[2] * in_channels,
            bias=True,
        )
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Weight initialization handled by centralized weight_initialization module
        """initialize_weights.

        Returns:
            Any: Description.
        """
        pass

    def _init_weights(self, m):
        # Weight initialization handled by centralized weight_initialization module
        """_init_weights.

        Args:
            m (Any): Description.
        Returns:
            Any: Description.
        """
        pass

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches."""
        pH, pW, pD = self.encoder.patch_embed.patch_size
        h, w, d = self.img_size
        assert h % pH == 0 and w % pW == 0 and d % pD == 0

        x = imgs.reshape(shape=(imgs.shape[0], self.in_channels, h, w, d))
        x = x.reshape(
            shape=(x.shape[0], self.in_channels, h // pH, pH, w // pW, pW, d // pD, pD),
        )
        x = torch.einsum("nchpwqdr->nhwdpqrc", x)
        x = x.reshape(
            shape=(
                x.shape[0],
                (h // pH) * (w // pW) * (d // pD),
                self.in_channels * pH * pW * pD,
            ),
        )
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images."""
        pH, pW, pD = self.encoder.patch_embed.patch_size
        h, w, d = self.img_size
        c = self.in_channels

        x = x.reshape(shape=(x.shape[0], h // pH, w // pW, d // pD, pH, pW, pD, c))
        x = torch.einsum("nhwdpqrc->nchpwqdr", x)
        imgs = x.reshape(shape=(x.shape[0], c, h, w, d))
        return imgs

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random masking to patches."""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # Sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through encoder with masking."""
        # Embed patches
        x = self.encoder.patch_embed(x)

        # Apply masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Encode visible patches
        x = self.encoder.encoder(x)

        return x, mask, ids_restore

    def forward_decoder(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through decoder."""
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0],
            ids_restore.shape[1] + 1 - x.shape[1],
            1,
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # unshuffle
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add position embedding
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict pixel values for all patches
        x = self.decoder_pred(x)

        # Remove cls token
        x = x[:, 1:, :]

        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Full forward pass for reconstruction.

        Args:
            x: Input tensor [B, C, H, W, D] or [B, C, H, W]
            **kwargs: Extra keyword arguments (batch, disable_spatial_masking, etc.)
                      are accepted but ignored for forward compatibility with
                      training strategy dispatch.

        forward method for MAEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        latent, mask, ids_restore = self.forward_encoder(x, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return self.unpatchify(pred)

    def generate(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate reconstruction from input."""
        mask_ratio = mask_ratio if mask_ratio is not None else self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return self.unpatchify(pred)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        return input_shape

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self) -> str:
        """Get model name."""
        return "mae_mri"

    def __repr__(self) -> str:
        """__repr__.

        Returns:
            str: Description.
        """
        return (
            f"MAEGenerator(name={self.name}, "
            f"img_size={self.img_size}, "
            f"patch_size={self.patch_size}, "
            f"mask_ratio={self.mask_ratio}, "
            f"params={self.get_parameter_count():,})"
        )
