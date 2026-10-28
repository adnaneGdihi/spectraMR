"""UNETR Generator
=================

A generator based on the UNETR (UNet Transformer) architecture for 3D medical image synthesis.
This model uses a Vision Transformer as the encoder and a CNN-based decoder.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="unetr_generator", training_mode="reconstruction")
class UNETRGenerator(nn.Module, IGenerator):
    """A generator that uses the UNETR architecture for 3D volume generation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ):
        """Initialize UNETRGenerator.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            img_size: The size of the input image.
            feature_size: Dimension of network feature size.
            hidden_size: Dimension of hidden layer.
            mlp_dim: Dimension of feedforward layer.
            num_heads: Number of attention heads.
            proj_type: Type of projection ('conv' or 'perceptron').
            norm_name: Feature normalization type.
            conv_block: If True, use convolutional blocks.
            res_block: If True, use residual blocks.
            dropout_rate: Fraction of the input units to drop.
            spatial_dims: Number of spatial dimensions.
            qkv_bias: If True, add bias to qkv projections.
            save_attn: If True, save attention weights.

        """
        super().__init__()

        try:
            from monai.networks.nets import UNETR
        except ImportError as exc:
            raise ImportError(
                "MONAI is not installed. Please install it to use UNETRGenerator.",
            ) from exc

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size

        self.unetr = UNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            proj_type=proj_type,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )

        # Verify instantiation success
        if sum(1 for _ in self.unetr.parameters()) == 0:
            raise ValueError("UNETR initialized with 0 parameters. Check MONAI version or config.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator.

        forward method for UNETRGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # [MRI FIX] Removed Tanh activation to avoid clipping positive MRI magnitude signals.
        return self.unetr(x)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate output from input tensor."""
        return self.forward(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get the output shape for given input shape."""
        n, _, d, h, w = input_shape
        return (n, self.out_channels, d, h, w)

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self) -> str:
        """Get generator name."""
        return "UNETRGenerator"


def create_unetr_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    img_size: tuple[int, int, int] = (96, 96, 96),
    feature_size: int = 16,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
    proj_type: str = "conv",
    norm_name: str = "instance",
    conv_block: bool = True,
    res_block: bool = True,
    dropout_rate: float = 0.0,
    spatial_dims: int = 3,
    qkv_bias: bool = False,
    save_attn: bool = False,
    **kwargs,
) -> UNETRGenerator:
    """Create UNETR generator instance."""
    return UNETRGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        proj_type=proj_type,
        norm_name=norm_name,
        conv_block=conv_block,
        res_block=res_block,
        dropout_rate=dropout_rate,
        spatial_dims=spatial_dims,
        qkv_bias=qkv_bias,
        save_attn=save_attn,
        **kwargs,
    )
