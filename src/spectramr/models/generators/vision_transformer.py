"""Vision Transformer (ViT) Generator
==================================

A standard Vision Transformer adapted for image generation/reconstruction tasks.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


def _conv_stem(in_channels: int, embed_dim: int, patch_size: int) -> nn.Sequential:
    """Xiao et al. (2021) ViT_C stem: log2(patch_size) stride-2 3x3 convs, then a 1x1 projection."""
    num_down = patch_size.bit_length() - 1
    if patch_size < 2 or (1 << num_down) != patch_size:
        raise ValueError(
            f"VisionTransformer: use_conv_stem needs a power-of-two patch_size >= 2, "
            f"got {patch_size}."
        )
    layers: list[nn.Module] = []
    channels = in_channels
    for i in range(num_down):
        out = max(embed_dim >> (num_down - i), 8)
        layers += [
            nn.Conv2d(channels, out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
        ]
        channels = out
    layers.append(nn.Conv2d(channels, embed_dim, 1))
    return nn.Sequential(*layers)


@register_model(
    name="vision_transformer",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class VisionTransformer(nn.Module, IGenerator):
    """Vision Transformer for Image Reconstruction.

    This implementation uses a standard ViT architecture with a decoder
    to map back to image space.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: int = 256,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_conv_stem: bool = False,
        **kwargs,
    ):
        """Initialize Vision Transformer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            img_size: Input image size
            patch_size: Patch size
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            use_conv_stem: Replace the patchify conv with a stride-2 3x3 conv stem
                (Xiao et al., 2021) whose total stride equals ``patch_size``.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.use_conv_stem = bool(use_conv_stem)

        self.num_patches = (img_size // patch_size) ** 2

        if img_size % patch_size != 0:
            raise ValueError(
                f"VisionTransformer: img_size={img_size} is not divisible by "
                f"patch_size={patch_size}."
            )
        self.patch_embed = (
            _conv_stem(in_channels, embed_dim, patch_size)
            if use_conv_stem
            else nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        )

        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Decoder (Simple projection back to patches)
        self.decoder_embed = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        self._init_weights()

    def _init_weights(self):
        """_init_weights.

        Returns:
            Any: Description.
        """
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if isinstance(self.patch_embed, nn.Sequential):
            for m in self.patch_embed:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
        else:
            nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.constant_(self.decoder_embed.bias, 0)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [N, C, H, W]

        Returns:
            Reconstructed tensor [N, C_out, H, W]

        forward method for VisionTransformer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        if tuple(x.shape[-2:]) != (self.img_size, self.img_size):
            raise ValueError(
                f"VisionTransformer: input {H}x{W} does not match img_size="
                f"{self.img_size}; the positional embedding is fixed to that grid."
            )

        # Patch Embedding
        x = self.patch_embed(x)  # [B, E, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]

        # Add Positional Embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        x = self.transformer(x)

        # Decoder
        x = self.decoder_embed(x)  # [B, N, P*P*C_out]

        # Reshape back to image
        x = x.transpose(1, 2)  # [B, P*P*C_out, N]
        x = x.reshape(
            B,
            self.out_channels,
            self.patch_size,
            self.patch_size,
            H // self.patch_size,
            W // self.patch_size,
        )
        x = x.permute(0, 1, 4, 2, 5, 3).reshape(B, self.out_channels, H, W)

        return x

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "vision_transformer"
