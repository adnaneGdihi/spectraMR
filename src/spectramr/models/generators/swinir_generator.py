"""SwinIR Generator Implementation
==============================

SOLID-compliant implementation of SwinIR (Swin Transformer for
Image Restoration) generator.
Based on the paper: "SwinIR: Image Restoration Using Swin Transformer"
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.blocks.swin import SwinBlock
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB)."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        img_size: int = 224,
        patch_norm: bool = True,
    ):
        """__init__.

        Args:
            dim (int): Description.
            input_resolution (tuple[int, int]): Description.
            depth (int): Description.
            num_heads (int): Description.
            window_size (int): Description.
            mlp_ratio (float): Description.
            qkv_bias (bool): Description.
            qk_scale (Optional[float]): Description.
            drop (float): Description.
            attn_drop (float): Description.
            drop_path (float): Description.
            norm_layer (nn.Module): Description.
            img_size (int): Description.
            patch_norm (bool): Description.
        """
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = nn.ModuleList(
            [
                SwinBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    dropout=drop,
                    attn_dropout=attn_drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ],
        )

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for RSTB.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, H*W, C)): Token sequence.
            x_size (tuple[int, int]): The (H, W) the tokens tile.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        res = x
        for block in self.residual_group:
            res = block(res, input_resolution=x_size)
        b, _, c = res.shape
        res = self.conv(res.transpose(1, 2).reshape(b, c, *x_size))
        return x + res.flatten(2).transpose(1, 2)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        """__init__.

        Args:
            img_size (int): Description.
            patch_size (int): Description.
            in_chans (int): Description.
            embed_dim (int): Description.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for PatchEmbed.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        x = self.conv(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    """Image to Patch Unembedding"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        """__init__.

        Args:
            img_size (int): Description.
            patch_size (int): Description.
            in_chans (int): Description.
            embed_dim (int): Description.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            x_size (tuple[int, int]): Description.
        Returns:
            torch.Tensor: Description.

        forward method for PatchUnEmbed.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_size (tuple[int, int]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


@register_model(name="swinir", training_mode="reconstruction")
class SwinIRGenerator(IGenerator, nn.Module):
    """SwinIR (Swin Transformer for Image Restoration) generator
    implementation. Based on the paper: "SwinIR: Image Restoration
    Using Swin Transformer"
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        upscale: int = 2,
        window_size: int = 8,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 1,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        patch_norm: bool = True,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            upscale (int): Description.
            window_size (int): Description.
            img_size (int): Description.
            patch_size (int): Description.
            in_chans (int): Description.
            embed_dim (int): Description.
            depths (tuple[int, ...]): Description.
            num_heads (tuple[int, ...]): Description.
            mlp_ratio (float): Description.
            qkv_bias (bool): Description.
            qk_scale (Optional[float]): Description.
            drop_rate (float): Description.
            attn_drop_rate (float): Description.
            drop_path_rate (float): Description.
            norm_layer (nn.Module): Description.
            patch_norm (bool): Description.
        """
        super().__init__()
        if patch_size != 1:
            raise ValueError(
                f"SwinIR: patch_size={patch_size} is unsupported; the upsampler restores "
                "only `upscale`, so tokens must tile the input at patch_size=1"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale = upscale
        self.window_size = window_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # absolute position embedding
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, embed_dim),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # stochastic depth decay rule

        if len(num_heads) != len(depths):
            raise ValueError(
                f"SwinIR: num_heads has {len(num_heads)} entries but depths has {len(depths)}"
            )

        # Canonical SwinIR: one RSTB per ``depths`` entry, all at the input resolution
        # and embed_dim, so the output is input * upscale whatever the depth config.
        self.layers = nn.ModuleList(
            RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                img_size=img_size,
                patch_norm=patch_norm,
            )
            for i_layer in range(len(depths))
        )

        self.norm = norm_layer(embed_dim)

        # build the last conv layer in deep feature extraction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # upsample
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * upscale * upscale, 1, 1, 0),
            nn.PixelShuffle(upscale),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, 1, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Weight initialization handled by centralized weight_initialization module
        """_init_weights.

        Args:
            m (Any): Description.
        Returns:
            Any: Description.
        """
        pass

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "SwinIR"

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Check and adjust image size to be divisible by
        window_size * patch_size.
        """
        _, _, h, w = x.size()
        mod_pad_h = (
            self.window_size * self.patch_size - h % (self.window_size * self.patch_size)
        ) % (self.window_size * self.patch_size)
        mod_pad_w = (
            self.window_size * self.patch_size - w % (self.window_size * self.patch_size)
        ) % (self.window_size * self.patch_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        return x

    def _get_pos_embed(self, num_patches: int) -> torch.Tensor:
        """Get positional embeddings, resampling if necessary for variable input shapes.

        Args:
            num_patches: Number of patches in the input

        Returns:
            Positional embeddings with shape (1, num_patches, embed_dim)
        """
        if self.absolute_pos_embed.shape[1] == num_patches:
            return self.absolute_pos_embed

        # Resample pos_embed for new patch resolution
        # This handles variable input sizes gracefully
        pos_embed = self.absolute_pos_embed
        if pos_embed.shape[1] != num_patches:
            # Reshape to spatial coordinates for interpolation
            orig_h = int(pos_embed.shape[1] ** 0.5 + 0.5)
            orig_w = int(pos_embed.shape[1] ** 0.5 + 0.5)
            new_h = int(num_patches**0.5 + 0.5)
            new_w = int(num_patches**0.5 + 0.5)

            pos_embed_2d = pos_embed.reshape(1, orig_h, orig_w, -1).permute(0, 3, 1, 2)
            pos_embed_2d = F.interpolate(
                pos_embed_2d, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            pos_embed = pos_embed_2d.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1)

        return pos_embed

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through the SwinIR model.

        forward method for SwinIRGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        # shallow feature extraction
        shallow = self.patch_embed.conv(x)  # B, C, Hp, Wp
        x_size = (shallow.shape[2], shallow.shape[3])
        x = self.patch_embed.norm(shallow.flatten(2).transpose(1, 2))  # B, Hp*Wp, C
        x = x + self._get_pos_embed(x.shape[1])
        x = self.pos_drop(x)

        # deep feature extraction
        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B, Hp*Wp, C
        x = self.patch_unembed(x, x_size)
        x = self.conv_after_body(x) + shallow

        # upsample
        x = self.conv_before_upsample(x)
        x = self.upsample(x)

        return x[:, :, : H * self.upscale, : W * self.upscale]

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples using the generator interface.

        This delegates directly to ``forward`` and mirrors other generator
        implementations in the codebase.
        """
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Compute the output shape for a given input shape.

        The method accepts both (N, C, H, W) and (C, H, W) style tuples and
        returns the output shape including the batch dimension if present.
        """
        if len(input_shape) == 4:
            n, c, h, w = input_shape
            return (n, self.out_channels, h * self.upscale, w * self.upscale)
        if len(input_shape) == 3:
            c, h, w = input_shape
            return (self.out_channels, h * self.upscale, w * self.upscale)
        raise ValueError(
            "Unsupported input_shape for SwinIRGenerator: expected (N,C,H,W) or (C,H,W)"
        )

    def get_parameter_count(self) -> int:
        """Return the total number of parameters for the model."""
        return sum(p.numel() for p in self.parameters())
