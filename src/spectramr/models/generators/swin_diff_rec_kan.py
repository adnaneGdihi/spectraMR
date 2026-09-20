"""Swin-Diff-Rec-KAN Generator.
============================

Analysis: A variation of SwinDiffRec that uses Complex KAN layers in the residual blocks.
This combines the global attention of Swin Transformers with the learnable activation functions
of KANs (Kolmogorov-Arnold Networks) for the local processing steps.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spectramr.infrastructure.physics.conditioning import PhysicsInformedConditioning
from spectramr.infrastructure.physics.data_consistency_layer import MaskedReplacementDataConsistency
from spectramr.models.blocks.complex_blocks import ComplexResBlock
from spectramr.models.blocks.swin import SwinBlock
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.layers.kan.complex_kan import ComplexFastKANConvLayer
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


class KANResBlock(nn.Module):
    """Residual Block with Complex KAN convolutions and Physics Conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        dropout: float = 0.0,
        kan_type: str = "RBF",
        num_grids: int = 4,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            emb_dim (int): Description.
            dropout (float): Description.
            kan_type (str): Description.
            num_grids (int): Description.
        """
        super().__init__()

        # in_channels/out_channels are for the COMPLEX inputs (so effectively half for real/imag parts internal logic if we used ComplexConv)
        # But ComplexFastKANConvLayer takes 'channels' as the dimension of Real (or Imag) part?
        # Let's check:
        # ComplexFastKANConvLayer.__init__(in_channels, ...)
        # It expects the "component" channel count.

        # Wait, SwinDiffRec passes "in_channels" which is the total stacked channels (e.g. 64).
        # So component channels = in_channels // 2.

        self.c_in = in_channels // 2
        self.c_out = out_channels // 2

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.kan1 = ComplexFastKANConvLayer(
            self.c_in,
            self.c_out,
            kernel_size=3,
            padding="same",
            kan_type=kan_type,
            num_grids=num_grids,
        )
        self.cond1 = PhysicsInformedConditioning(emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.kan2 = ComplexFastKANConvLayer(
            self.c_out,
            self.c_out,
            kernel_size=3,
            padding="same",
            kan_type=kan_type,
            num_grids=num_grids,
        )
        self.cond2 = PhysicsInformedConditioning(emb_dim, out_channels)

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            # We can use a simple conv for shortcut or a KAN (expensive). Let's use simple KAN or Conv.
            # Use simple Conv1x1 for shortcut to save compute.
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KANResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.norm1(x)
        # KANs have their own internal activations (splines), so we often skip pre-activation like ReLU.
        # But we kept GroupNorm.
        h = self.kan1(h)
        h = self.cond1(h, emb)  # AdaGN

        h = self.norm2(h)
        h = self.dropout(h)
        h = self.kan2(h)
        h = self.cond2(h, emb)  # AdaGN

        return h + self.shortcut(x)


@register_model(name="swin_diff_rec_kan", training_mode="diffusion")
class SwinDiffRecKAN(GradCheckpointingMixin, nn.Module, IGenerator):
    """
    Swin-Diff-Rec-KAN Backbone.

    Replaces standard convolutions in SwinDiffRec with Complex KANs.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        image_size: int = 256,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        swin_depth: int = 2,
        swin_heads: int = 8,
        swin_window_size: int = 8,
        physics_emb_dim: int = 256,
        dropout: float = 0.0,
        kan_type: str = "RBF",
        kan_num_grids: int = 4,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            image_size (int): Description.
            base_channels (int): Description.
            channel_mults (Sequence[int]): Description.
            num_res_blocks (int): Description.
            swin_depth (int): Description.
            swin_heads (int): Description.
            swin_window_size (int): Description.
            physics_emb_dim (int): Description.
            dropout (float): Description.
            kan_type (str): Description.
            kan_num_grids (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]

        self.image_size = image_size

        # 1. Conditioning
        self.emb_dim = physics_emb_dim
        self.time_pos_enc = nn.Sequential(
            nn.Linear(1, physics_emb_dim // 2),
            nn.SiLU(),
            nn.Linear(physics_emb_dim // 2, physics_emb_dim),
        )

        # 2. Encoder
        self.downs = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        dims = [base_channels * m for m in channel_mults]

        # Start Conv (Regular Conv for first layer to lift channels?)
        # Or KAN? KAN is fine.
        self.start_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        curr_dim = base_channels

        for i, dim in enumerate(dims):
            layers = nn.ModuleList()
            for _ in range(num_res_blocks):
                # Encoder: Standard Complex Conv (ComplexResBlock)
                layers.append(ComplexResBlock(curr_dim, dim, physics_emb_dim, dropout))
                curr_dim = dim
            self.down_blocks.append(layers)

            if i < len(dims) - 1:
                self.downs.append(nn.Conv2d(curr_dim, curr_dim, 3, stride=2, padding=1))
            else:
                self.downs.append(nn.Identity())

        # 3. Bottleneck (Swin Transformer) - Keep Swin, it's efficient for global mixing
        self.bot_dim = curr_dim
        self.swin_blocks = nn.ModuleList(
            [
                SwinBlock(
                    dim=self.bot_dim,
                    num_heads=swin_heads,
                    window_size=swin_window_size,
                    shift_size=0 if (i % 2 == 0) else swin_window_size // 2,
                    input_resolution=(
                        image_size // (2 ** (len(dims) - 1)),
                        image_size // (2 ** (len(dims) - 1)),
                    ),
                )
            ]
        )

        # Bottleneck KAN Injection:
        # "Use Complex ConvKAN only in the Bottleneck..."
        # We add a KAN ResBlock here to enhance reasoning alongside Swin
        self.bottleneck_kan = KANResBlock(
            curr_dim, curr_dim, physics_emb_dim, dropout, kan_type, kan_num_grids
        )

        # 4. Decoder
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        rev_dims = list(reversed(dims))

        # Calculate skip dimensions: [base, d0, d1, ..., d(N-2)]
        skip_dims = [base_channels] + dims[:-1]
        rev_skip_dims = list(reversed(skip_dims))

        prev_dim = rev_dims[0] if rev_dims else base_channels
        for i, dim in enumerate(rev_dims):
            if i > 0:
                self.ups.append(nn.ConvTranspose2d(prev_dim, dim, 4, 2, 1))
            else:
                self.ups.append(nn.Identity())

            layers = nn.ModuleList()

            # Input dim = current dim (upsampled) + skip dim
            in_d = dim + rev_skip_dims[i]

            for j in range(num_res_blocks):
                # First decoder block (Deepest level -> i=0)
                if i == 0:
                    # "Use Complex ConvKAN only in the Bottleneck and the first Decoder block"
                    layers.append(
                        KANResBlock(in_d, dim, physics_emb_dim, dropout, kan_type, kan_num_grids)
                    )
                else:
                    # Standard Complex Conv for rest
                    layers.append(ComplexResBlock(in_d, dim, physics_emb_dim, dropout))

                in_d = dim
            self.up_blocks.append(layers)
            prev_dim = dim

        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        # 5. Data Consistency
        self.dc_layer = MaskedReplacementDataConsistency()


    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        acceleration: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        measured_kspace: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # [FIX] Alias for strategy compatibility
        """forward.

        Args:
            x (torch.Tensor): Description.
            timesteps (torch.Tensor): Description.
            acceleration (torch.Tensor | None): Description.
            mask (torch.Tensor | None): Description.
            measured_kspace (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SwinDiffRecKAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            acceleration (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if measured_kspace is None:
            measured_kspace = kwargs.get("kspace_measured")

        cond_input = acceleration if acceleration is not None else timesteps.float().unsqueeze(1)
        if cond_input.ndim == 1:
            cond_input = cond_input.unsqueeze(1)

        emb = self.time_pos_enc(cond_input.float())

        ckpt = self._checkpointing_active()
        h = self.start_conv(x)
        skips = [h]

        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = checkpoint(block, h, emb, use_reentrant=False) if ckpt else block(h, emb)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        B, C, H, W = h.shape
        h_flat = h.flatten(2).transpose(1, 2)

        for swin_block in self.swin_blocks:
            h_flat = (
                checkpoint(swin_block, h_flat, (H, W), use_reentrant=False)
                if ckpt
                else swin_block(h_flat, input_resolution=(H, W))
            )
        for swin_block in self.swin_blocks:
            h_flat = (
                checkpoint(swin_block, h_flat, (H, W), use_reentrant=False)
                if ckpt
                else swin_block(h_flat, input_resolution=(H, W))
            )

        h = h_flat.transpose(1, 2).view(B, C, H, W)

        # Apply Bottleneck KAN
        h = self.bottleneck_kan(h, emb)

        skips.pop()

        for i, blocks in enumerate(self.up_blocks):
            if i > 0:
                h = self.ups[i](h)

            if skips:
                skip = skips.pop()
                if skip.shape != h.shape:
                    h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
                h = torch.cat([h, skip], dim=1)

            for block in blocks:
                h = checkpoint(block, h, emb, use_reentrant=False) if ckpt else block(h, emb)

        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.final_conv(h)

        if measured_kspace is not None and mask is not None:
            out = self.dc_layer(out, measured_kspace, mask)

        return out

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "SwinDiffRecKAN"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generation wrapper to satisfy IGenerator interface."""
        timesteps = kwargs.pop("timesteps", None)
        if timesteps is None:
            device = z.device
            timesteps = torch.zeros(z.shape[0], device=device)

        return self(z, timesteps=timesteps, **kwargs)
