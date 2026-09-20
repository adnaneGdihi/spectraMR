"""Swin-Diff-Rec Generator.
========================

Implementation of the Swin-Diff-Rec backbone for Cold Diffusion MRI reconstruction.
Based on the blueprint: "Dual-Domain Swin Transformer with Mask Conditioning".

Key Features:
- U-Net Architecture with Swin Transformer Bottleneck.
- PhysicsInformedConditioning (AdaGN) modulated by Acceleration Factor (R).
- Data Consistency (DC) Layers enforcing k-space fidelity.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spectramr.infrastructure.physics.data_consistency_layer import MaskedReplacementDataConsistency
from spectramr.models.blocks.complex_blocks import ComplexResBlock
from spectramr.models.blocks.residual import ResidualBlock as ResBlock
from spectramr.models.blocks.swin import SwinBlock
from spectramr.models.blocks.timestep_embedding import sinusoidal_timestep_embedding
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="swin_diff_rec", training_mode="diffusion", spatial_dims=(2,))
class SwinDiffRec(GradCheckpointingMixin, nn.Module, IGenerator):
    """
    Swin-Diff-Rec Backbone.

    A Hybrid Dual-Domain Architecture that uses Swin Transformers for global context
    and explicitly enforces MRI physics via Data Consistency and AdaGN conditioning.
    """

    def __init__(
        self,
        in_channels: int = 2,  # Real/Imag or Complex treated as 2 channels
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
        use_complex_conv: bool = True,
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
            use_complex_conv (bool): Description.
        """
        super().__init__()
        # ``use_complex_conv=False`` selects ``ResBlock`` below, but the call
        # site is written for ``ComplexResBlock(in_channels, out_channels,
        # emb_dim, dropout)`` -- a contract ``ResidualBlock(channels,
        # kernel_size, stride, padding)`` does not share. The four arguments
        # bind positionally and silently: ``dim`` becomes the *kernel size* and
        # ``physics_emb_dim`` the *stride*, so construction allocates
        # convolutions orders of magnitude larger than the declared model,
        # before ``forward`` is ever reached (#1064). No real-valued block path
        # was ever designed for this family -- the KAN sibling
        # (``swin_diff_rec_kan.py``) uses ``ComplexResBlock`` unconditionally.
        # Raise rather than degrade to a default (non-negotiable 3).
        if not use_complex_conv:
            raise ValueError(
                "SwinDiffRec(use_complex_conv=False) is not implemented. The "
                "residual-block call site passes ComplexResBlock's "
                "(in_channels, out_channels, emb_dim, dropout) contract, which "
                "binds positionally onto ResidualBlock's (channels, "
                "kernel_size, stride, padding) -- making `dim` the kernel size "
                "and `physics_emb_dim` the stride, and allocating a model "
                "orders of magnitude larger than the one declared. This family "
                "has no real-valued block path (swin_diff_rec_kan uses "
                "ComplexResBlock unconditionally). Pass use_complex_conv=True, "
                "or choose a backbone_type that implements a real-valued path."
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_complex_conv = use_complex_conv

        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]

        self.image_size = image_size

        # 1. Conditioning Embeddings
        self.emb_dim = physics_emb_dim

        # Was ``nn.Linear(1, ...)`` on the RAW scalar — a rank-1 map of t, so
        # every timestep code sat on one line and the network could express
        # "large t" but not "which t". Now a sinusoidal basis (scaled by
        # ``max_timesteps``) feeds the same MLP width, matching ComplexUNet.
        self.time_pos_enc = nn.Sequential(
            nn.Linear(physics_emb_dim, physics_emb_dim // 2),
            nn.SiLU(),
            nn.Linear(physics_emb_dim // 2, physics_emb_dim),
        )
        # Contrast conditioning was swallowed by **kwargs; see forward().
        self.contrast_proj: nn.Module | None = None

        # 2. Encoder (Downsampling)
        self.downs = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        dims = [base_channels * m for m in channel_mults]
        in_dim = in_channels

        # Initial Conv
        if use_complex_conv:
            from spectramr.models.layers.complex_conv import ComplexConv2d

            # Ensure even channels
            if in_channels % 2 != 0:
                raise ValueError("Complex SwinDiffRec requires even input channels.")
            self.start_conv = ComplexConv2d(in_channels // 2, base_channels // 2, 3, padding=1)
        else:
            self.start_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        curr_dim = base_channels

        # The ``else`` arm is unreachable: __init__ raises on
        # use_complex_conv=False. Kept as the seam a real-valued
        # implementation would fill, not as a supported path (#1064).
        BlockClass = ComplexResBlock if use_complex_conv else ResBlock

        for i, dim in enumerate(dims):
            # ResBlocks per level
            layers = nn.ModuleList()
            for _ in range(num_res_blocks):
                layers.append(BlockClass(curr_dim, dim, physics_emb_dim, dropout))
                curr_dim = dim
            self.down_blocks.append(layers)

            # Downsample (except last)
            if i < len(dims) - 1:
                # Use complex conv for downsampling if complex mode
                if use_complex_conv:
                    self.downs.append(
                        ComplexConv2d(curr_dim // 2, curr_dim // 2, 3, stride=2, padding=1)
                    )
                else:
                    self.downs.append(nn.Conv2d(curr_dim, curr_dim, 3, stride=2, padding=1))
            else:
                self.downs.append(nn.Identity())

        # 3. Bottleneck (Swin Transformer)
        # Flatten resolution for Swin: H_bot x W_bot
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
                for i in range(swin_depth)
            ]
        )

        # 4. Decoder (Upsampling)
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        rev_dims = list(reversed(dims))  # [512, 256, 128, 64]

        skip_dims = [base_channels] + dims[:-1]
        rev_skip_dims = list(reversed(skip_dims))

        prev_dim = rev_dims[0] if rev_dims else base_channels
        for i, dim in enumerate(rev_dims):
            # Upsample (except first, effectively inverse of down)
            if i > 0:
                self.ups.append(nn.ConvTranspose2d(prev_dim, dim, 4, 2, 1))
            else:
                self.ups.append(nn.Identity())

            # ResBlocks
            layers = nn.ModuleList()

            # Input dim = current dim (upsampled) + skip dim
            in_d = dim + rev_skip_dims[i]

            for _ in range(num_res_blocks):
                layers.append(BlockClass(in_d, dim, physics_emb_dim, dropout))
                in_d = dim  # After first block, dim is reduced to 'dim'
            self.up_blocks.append(layers)
            prev_dim = dim

        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_act = nn.SiLU()

        if use_complex_conv:
            self.final_conv = ComplexConv2d(base_channels // 2, out_channels // 2, 3, padding=1)
        else:
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
        """
        Args:
            x: Input image [B, C, H, W]
            timesteps: [B]
            acceleration: [B] or [B, 1]. Acceleration factor for conditioning.
            mask: [B, 1, H, W]. Sampling mask.
            measured_kspace: [B, C, H, W]. Observed data for DC.

        forward method for SwinDiffRec.

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
        # [FIX] Alias for strategy compatibility
        if measured_kspace is None:
            measured_kspace = kwargs.get("kspace_measured")

        # 1. Embedding
        # If acceleration is provided, use it. Else fall back to timesteps (assuming t~R correlation)
        cond_input = acceleration if acceleration is not None else timesteps.float().unsqueeze(1)
        if cond_input.ndim == 1:
            cond_input = cond_input.unsqueeze(1)

        # Sinusoidal encoding of whichever scalar was chosen. ``max_timesteps``
        # is forwarded by the generator and was previously swallowed here. When
        # conditioning on ``acceleration`` the horizon is a max R, not a step
        # count, so leave that path unscaled.
        _max_t = kwargs.get("max_timesteps") if acceleration is None else None
        t_sin = sinusoidal_timestep_embedding(
            cond_input.squeeze(-1), self.emb_dim, max_timesteps=_max_t
        )
        emb = self.time_pos_enc(t_sin)

        # Contrast conditioning, mirroring ComplexUNet (complex_unet.py:333):
        # added onto the timestep embedding so it reaches every ResBlock through
        # the existing AdaGN path.
        contrast_emb = kwargs.get("contrast_emb")
        if contrast_emb is not None:
            if contrast_emb.shape[-1] != self.emb_dim:
                raise ValueError(
                    f"contrast_emb width {contrast_emb.shape[-1]} != "
                    f"physics_emb_dim {self.emb_dim}. Set "
                    "model_kwargs.physics_emb_dim to match "
                    "model_kwargs.time_embedding_dim (the generator builds "
                    "contrast_embedding at that width)."
                )
            if contrast_emb.shape[0] != emb.shape[0]:
                contrast_emb = contrast_emb[: emb.shape[0]]
            emb = emb + contrast_emb

        # 2. Encoder
        ckpt = self._checkpointing_active()
        h = self.start_conv(x)
        skips = [h]

        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = checkpoint(block, h, emb, use_reentrant=False) if ckpt else block(h, emb)
            skips.append(h)
            if i < len(self.downs):
                h = self.downs[i](h)

        # 3. Bottleneck (Swin)
        # Reshape [B, C, H, W] -> [B, L, C] for Swin
        B, C, H, W = h.shape
        h_flat = h.flatten(2).transpose(1, 2)  # [B, H*W, C]

        for swin_block in self.swin_blocks:
            h_flat = (
                checkpoint(swin_block, h_flat, (H, W), use_reentrant=False)
                if ckpt
                else swin_block(h_flat, input_resolution=(H, W))
            )

        h = h_flat.transpose(1, 2).view(B, C, H, W)

        # 4. Decoder
        skips.pop()  # Remove bottleneck input from skips (it's 'h' before bottleneck)
        # Actually standard UNet:
        # Enc1 -> Down -> Enc2 -> Down -> Bot -> Up -> Cat(Enc2) -> Dec...

        # Re-align skips
        # skips has: [Stem, Enc1_Out, Enc2_Out, ..., EncLast_Out]
        # Bottleneck processes EncLast_Out (downsampled)
        # Decoder 0: Up(Bot) -> Cat(EncLast_Out) -> Dec0

        for i, blocks in enumerate(self.up_blocks):
            # i=0: Deepest decoder block
            # Upsample
            if i > 0:  # First block is at bottleneck res? No.
                # Rev Dims: [512, 256, 128, 64]
                # Upsample from prev
                h = self.ups[i](h)

            # Skip connection
            # Depending on depth, we might have a skip
            if skips:
                skip = skips.pop()
                # Check shapes for safety (Swin might have padded)
                if skip.shape != h.shape:
                    h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
                h = torch.cat([h, skip], dim=1)

            for block in blocks:
                h = checkpoint(block, h, emb, use_reentrant=False) if ckpt else block(h, emb)

        # 5. Output
        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.final_conv(h)

        # 6. Data Consistency (if measured_kspace provided)
        # "Predictor-Corrector": We predicted 'out', now we correct it with DC
        # DC requires: image, measured_kspace, mask
        if measured_kspace is not None and mask is not None:
            out = self.dc_layer(out, measured_kspace, mask)

        return out

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "SwinDiffRec"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generation wrapper to satisfy IGenerator interface."""
        # For this component-level generator, generate is just forward pass
        # 'z' is treated as input 'x'
        # Unwrap args if they match forward signature
        timesteps = kwargs.pop("timesteps", None)
        if timesteps is None:
            # Create dummy timesteps if not provided
            device = z.device
            timesteps = torch.zeros(z.shape[0], device=device)

        return self(z, timesteps=timesteps, **kwargs)
