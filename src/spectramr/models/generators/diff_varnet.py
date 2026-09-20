"""Diff-VarNet Generator.
======================

Implementation of the Diff-VarNet (Unrolled Variational Network) backbone for
Cold Diffusion MRI reconstruction.

Key Features:
- Unrolled optimization iterations (Gradient Descent style).
- Complex-Valued Convolutional Regularization (Holomorphic).
- Explicit Data Consistency (physics constraint) at every step.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spectramr.infrastructure.physics.conditioning import PhysicsInformedConditioning
from spectramr.infrastructure.physics.data_consistency_layer import MaskedReplacementDataConsistency
from spectramr.models.blocks.timestep_embedding import sinusoidal_timestep_embedding
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.layers.complex_conv import ComplexConv2d
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


class ComplexUnrolledBlock(nn.Module):
    """
    Learned Regularization Block using Complex Convolutions.

    x_{k+1} = x_k - step_size * Grad(Regularizer(x_k))

    But typically in MoDL/VarNet we parameterize the update directly:
    x_{k+1} = DC( x_k + Conv(x_k) )
    """

    def __init__(
        self,
        channels: int,  # Number of complex channels (e.g., 1 for single-coil image)
        features: int = 64,
        num_layers: int = 3,
        physics_emb_dim: int = 256,
        kernel_size: int = 3,
    ):
        """__init__.

        Args:
            channels (int): Description.
            features (int): Description.
            num_layers (int): Description.
            physics_emb_dim (int): Description.
            kernel_size (int): Description.
        """
        super().__init__()

        # Input: [B, 2*channels, H, W] (Stacked Real/Imag)
        # ComplexConv2d expects:
        #  in_channels = channels (complex count)
        #  It splits input 2*C into Real/Imag parts.

        self.layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()

        # 1. Input Conv
        self.layers.append(ComplexConv2d(channels, features, kernel_size, padding=kernel_size // 2))
        self.cond_layers.append(
            PhysicsInformedConditioning(physics_emb_dim, features * 2)
            # Output of ComplexConv is [B, 2*features, H, W] (Stacked)
            # AdaGN needs to modulate 2*features channels
        )

        # 2. Hidden Layers
        for _ in range(num_layers - 2):
            self.layers.append(
                ComplexConv2d(features, features, kernel_size, padding=kernel_size // 2)
            )
            self.cond_layers.append(PhysicsInformedConditioning(physics_emb_dim, features * 2))

        # 3. Output Conv
        self.layers.append(ComplexConv2d(features, channels, kernel_size, padding=kernel_size // 2))
        # No conditioning on final output usually, or yes?
        # Let's add it for consistency in the "residual" path
        self.cond_layers.append(PhysicsInformedConditioning(physics_emb_dim, channels * 2))

        # Activation
        # Complex activation is tricky. ModReLU is common.
        # But here we stick to complex convolution and simple non-linearity on the magnitude/channels?
        # Standard approach: Apply ReLU on Real/Imag separately or LeakyReLU
        # ComplexConv2d returns stacked output. We can just use standard activations on that.
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # Data Consistency
        self.dc = MaskedReplacementDataConsistency()

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        measured_kspace: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input image guess [B, 2*C, H, W]
            emb: Physics embedding [B, emb_dim]
            measured_kspace: [B, 2*C, H, W] or None
            mask: [B, 1, H, W] or None

        forward method for ComplexUnrolledBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x

        h = x
        for i, (conv, cond) in enumerate(zip(self.layers, self.cond_layers, strict=False)):
            h = conv(h)  # Returns [B, 2*F, H, W]
            h = cond(h, emb)  # Modulate

            # Apply activation (except last layer typically? MoDL residual is usually non-activated at very end)
            if i < len(self.layers) - 1:
                h = self.activation(h)

        # Residual Update: x_reg = x + CNN(x)
        x_reg = residual + h

        # Data Consistency Step
        if measured_kspace is not None and mask is not None:
            x_final = self.dc(x_reg, measured_kspace, mask)
        else:
            x_final = x_reg

        return x_final


@register_model(
    name="diff_varnet",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="kspace",
    accepts_complex=True,
    expects_real_imag_interleaved=True,
)
class DiffVarNet(GradCheckpointingMixin, nn.Module, IGenerator):
    """
    Diff-VarNet Backbone.

    A Physics-Embedded Unrolled Network for Cold Diffusion.
    """

    def __init__(
        self,
        in_channels: int = 2,  # Stacked Real/Imag
        out_channels: int = 2,
        image_size: int = 256,
        num_unrolls: int = 5,
        base_channels: int = 64,  # Feature dim for regularizer
        physics_emb_dim: int = 256,
        dropout: float = 0.0,  # Kept for interface consistency
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            image_size (int): Description.
            num_unrolls (int): Description.
            base_channels (int): Description.
            physics_emb_dim (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Check channels
        if in_channels % 2 != 0:
            raise ValueError(
                f"DiffVarNet expects even input channels (Real/Imag), got {in_channels}"
            )

        self.complex_channels = in_channels // 2

        # 1. Conditioning Embedding (same as SwinDiffRec)
        #
        # Was ``nn.Linear(1, ...)`` on the RAW scalar, which is a rank-1 map of
        # t: every timestep code sits on one line, so the network could express
        # "large t" but not "which t". Now a sinusoidal basis (scaled by
        # ``max_timesteps``) feeds the same MLP width, matching what ComplexUNet
        # already does. See models/blocks/timestep_embedding.py for why the
        # scaling is load-bearing at T=28.
        self.physics_emb_dim = physics_emb_dim
        self.time_pos_enc = nn.Sequential(
            nn.Linear(physics_emb_dim, physics_emb_dim // 2),
            nn.SiLU(),
            nn.Linear(physics_emb_dim // 2, physics_emb_dim),
        )
        # Contrast conditioning: the generator emits ``contrast_emb`` at
        # ``time_embedding_dim`` width whenever num_contrasts > 0. It used to be
        # swallowed by **kwargs here, so every arm declaring num_contrasts ran
        # without it. Projected rather than assumed-compatible: the two widths
        # agree only by coincidence today (256 both), and a silent broadcast on
        # a future mismatch would be the next facade.
        self.contrast_proj: nn.Module | None = None

        # 2. Unrolled Blocks
        self.blocks = nn.ModuleList(
            [
                ComplexUnrolledBlock(
                    channels=self.complex_channels,
                    features=base_channels,
                    physics_emb_dim=physics_emb_dim,
                )
                for _ in range(num_unrolls)
            ]
        )

        # No final convolution, usually output of iterations is the result.


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
            x: Input image [B, 2*C, H, W]
            timesteps: [B]
            acceleration: [B] or [B, 1].
            mask: [B, 1, H, W]
            measured_kspace: [B, 2*C, H, W]

        forward method for DiffVarNet.

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

        # [FIX] Handle 5D volumetric input [B, C, D, H, W] → [B*D, C, H, W]
        _was_5d = False
        if x.ndim == 5:
            _was_5d = True
            B, C, D, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            if measured_kspace is not None and measured_kspace.ndim == 5:
                measured_kspace = measured_kspace.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
            if mask is not None and mask.ndim == 5:
                mask = mask.permute(0, 2, 1, 3, 4).reshape(B * D, mask.shape[1], H, W)

        # 1. Embedding
        cond_input = acceleration if acceleration is not None else timesteps.float().unsqueeze(1)
        if cond_input.ndim == 1:
            cond_input = cond_input.unsqueeze(1)

        # Handle batch dimension for 5D inputs
        if _was_5d and cond_input.shape[0] != x.shape[0]:
            cond_input = cond_input.repeat_interleave(D, dim=0)

        # Sinusoidal encoding of whichever scalar was chosen. ``max_timesteps``
        # is already forwarded by the generator (kspace_cold_diffusion_generator
        # sets filtered_kwargs["max_timesteps"]); it was previously swallowed
        # here. When conditioning on ``acceleration`` instead of ``timesteps``
        # the horizon is the max acceleration, so fall back to the raw value
        # rather than dividing an R by a step count.
        _max_t = kwargs.get("max_timesteps") if acceleration is None else None
        t_sin = sinusoidal_timestep_embedding(
            cond_input.squeeze(-1), self.physics_emb_dim, max_timesteps=_max_t
        )
        emb = self.time_pos_enc(t_sin)

        # Contrast conditioning, mirroring ComplexUNet (complex_unet.py:333):
        # added onto the timestep embedding so it reaches every unrolled block
        # through the existing AdaGN path.
        contrast_emb = kwargs.get("contrast_emb")
        if contrast_emb is not None:
            if contrast_emb.shape[-1] != self.physics_emb_dim:
                if self.contrast_proj is None:
                    raise ValueError(
                        f"contrast_emb width {contrast_emb.shape[-1]} != "
                        f"physics_emb_dim {self.physics_emb_dim}. Set "
                        "model_kwargs.physics_emb_dim to match "
                        "model_kwargs.time_embedding_dim (the generator builds "
                        "contrast_embedding at that width)."
                    )
                contrast_emb = self.contrast_proj(contrast_emb)
            if _was_5d and contrast_emb.shape[0] != emb.shape[0]:
                contrast_emb = contrast_emb.repeat_interleave(D, dim=0)
            if contrast_emb.shape[0] != emb.shape[0]:
                contrast_emb = contrast_emb[: emb.shape[0]]
            emb = emb + contrast_emb

        # 2. Iterative Unrolling
        curr_x = x

        ckpt = self._checkpointing_active()
        for block in self.blocks:
            curr_x = (
                checkpoint(block, curr_x, emb, measured_kspace, mask, use_reentrant=False)
                if ckpt
                else block(curr_x, emb, measured_kspace, mask)
            )

        # [FIX] Restore 5D shape if input was volumetric
        if _was_5d:
            curr_x = curr_x.reshape(B, D, C, *curr_x.shape[2:]).permute(0, 2, 1, 3, 4)

        return curr_x

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "DiffVarNet"

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
