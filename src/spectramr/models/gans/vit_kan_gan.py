import math

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from spectramr.models.layers.kan.kan_convs.kans.layers import FastKANLayer
from spectramr.models.registry import register_model


class PreNorm(nn.Module):
    """PreNorm class."""

    def __init__(self, dim, fn):
        """__init__.

        Args:
            dim (Any): Description.
            fn (Any): Description.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for PreNorm.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.fn(self.norm(x), **kwargs)


class FeedForwardKAN(nn.Module):
    """FeedForwardKAN class."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        """__init__.

        Args:
            dim (Any): Description.
            hidden_dim (Any): Description.
            dropout (Any): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            FastKANLayer(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            FastKANLayer(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for FeedForwardKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


class Attention(nn.Module):
    """Attention class."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        """__init__.

        Args:
            dim (Any): Description.
            heads (Any): Description.
            dim_head (Any): Description.
            dropout (Any): Description.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for Attention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerKAN(nn.Module):
    """TransformerKAN class."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        """__init__.

        Args:
            dim (Any): Description.
            depth (Any): Description.
            heads (Any): Description.
            dim_head (Any): Description.
            mlp_dim (Any): Description.
            dropout (Any): Description.
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                            ),
                        ),
                        PreNorm(dim, FeedForwardKAN(dim, mlp_dim, dropout=dropout)),
                    ],
                ),
            )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for TransformerKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


@register_model(name="vit_kan_generator", training_mode="gan")
class ViTKANGenerator(nn.Module):
    """ViTKANGenerator class."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        image_size: int = 256,
        patch_size: int = 16,
        num_classes: int | None = None,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 512,
        channels: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        **kwargs,
    ):
        """Initialize ViT-KAN Generator.

        Args:
            in_channels: Input channels (used if ``channels`` not provided).
            out_channels: Output channels (used if ``num_classes`` not provided).
            image_size: Spatial resolution.
            patch_size: Patch size for tokenization.
            num_classes: Alias for out_channels (legacy).
            dim: Transformer embedding dimension.
            depth: Number of transformer blocks.
            heads: Number of attention heads.
            mlp_dim: MLP hidden dimension.
            channels: Alias for in_channels (legacy).
            dim_head: Attention head dimension.
            dropout: Dropout rate.
            emb_dropout: Embedding dropout rate.
        """
        super().__init__()
        # Resolve legacy aliases
        channels = channels if channels is not None else in_channels
        num_classes = num_classes if num_classes is not None else out_channels

        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, (
            "Image dimensions must be divisible by the patch size."
        )

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout_layer = nn.Dropout(emb_dropout)

        self.transformer = TransformerKAN(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, patch_dim))

        # Store patch dimensions for dynamic reconstruction
        self.patch_height = patch_height
        self.patch_width = patch_width

    def forward(self, img):
        """forward.

        Args:
            img (Any): Description.
        Returns:
            Any: Description.

        forward method for ViTKANGenerator.

        Executes PyTorch tensor operations.

        Args:
            img (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout_layer(x)

        x = self.transformer(x)

        x = self.mlp_head(x)

        # Dynamically compute the spatial dimensions from the actual tensor
        # n = h * w, where h and w are the number of patches in each dimension
        # We need to find h and w such that h * w = n
        # Since we know patch_height and patch_width, we can compute:
        # total_height = sqrt(n) * patch_height (assuming square image)
        # But to be more robust, let's compute h and w assuming square patches

        # For square images, n should be a perfect square
        h_patches = int(math.sqrt(n))
        w_patches = h_patches  # Assume square

        # Verify this gives us the correct number of patches
        if h_patches * w_patches != n:
            # If not square, try to find factors
            for h in range(1, int(math.sqrt(n)) + 1):
                if n % h == 0:
                    h_patches = h
                    w_patches = n // h
                    break

        # Create dynamic to_image layer
        dynamic_to_image = Rearrange(
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h_patches,
            w=w_patches,
            p1=self.patch_height,
            p2=self.patch_width,
        )

        img_out = dynamic_to_image(x)
        img_out = torch.tanh(img_out)  # To output in [-1, 1] range

        return img_out


def get_generator(
    in_channels=1,
    out_channels=1,
    image_size=240,
    patch_size=16,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=512,
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1,
    **kwargs,
):
    """get_generator.

    Args:
        in_channels (Any): Description.
        out_channels (Any): Description.
        image_size (Any): Description.
        patch_size (Any): Description.
        dim (Any): Description.
        depth (Any): Description.
        heads (Any): Description.
        mlp_dim (Any): Description.
        dim_head (Any): Description.
        dropout (Any): Description.
        emb_dropout (Any): Description.
    Returns:
        Any: Description.
    """
    return ViTKANGenerator(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=out_channels,
        channels=in_channels,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head,
        dropout=dropout,
        emb_dropout=emb_dropout,
    )
