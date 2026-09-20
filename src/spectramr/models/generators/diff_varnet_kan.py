"""Diff-VarNet-KAN Generator.
==========================

Implementation of the Diff-VarNet-KAN (Unrolled Variational Network with KANs).
Combines the physics-embedded unrolling of DiffVarNet with the expressive power of Complex KANs.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from spectramr.infrastructure.physics.conditioning import PhysicsInformedConditioning
from spectramr.infrastructure.physics.data_consistency_layer import MaskedReplacementDataConsistency
from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.layers.kan.complex_kan import ComplexFastKANConvLayer
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


class ComplexKANUnrolledBlock(nn.Module):
    """
    Learned Regularization Block using Complex KAN Convolutions.
    """

    def __init__(
        self,
        channels: int,  # Number of complex channels (e.g., 1 for single-coil image)
        features: int = 64,
        num_layers: int = 3,
        physics_emb_dim: int = 256,
        kan_type: str = "RBF",
        num_grids: int = 4,
    ):
        """__init__.

        Args:
            channels (int): Description.
            features (int): Description.
            num_layers (int): Description.
            physics_emb_dim (int): Description.
            kan_type (str): Description.
            num_grids (int): Description.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.cond_layers = nn.ModuleList()

        # 1. Input KAN
        self.layers.append(
            ComplexFastKANConvLayer(
                channels,
                features,
                kernel_size=3,
                padding="same",
                kan_type=kan_type,
                num_grids=num_grids,
            )
        )
        self.cond_layers.append(PhysicsInformedConditioning(physics_emb_dim, features * 2))

        # 2. Hidden KANs
        for _ in range(num_layers - 2):
            self.layers.append(
                ComplexFastKANConvLayer(
                    features,
                    features,
                    kernel_size=3,
                    padding="same",
                    kan_type=kan_type,
                    num_grids=num_grids,
                )
            )
            self.cond_layers.append(PhysicsInformedConditioning(physics_emb_dim, features * 2))

        # 3. Output KAN
        self.layers.append(
            ComplexFastKANConvLayer(
                features,
                channels,
                kernel_size=3,
                padding="same",
                kan_type=kan_type,
                num_grids=num_grids,
            )
        )
        self.cond_layers.append(PhysicsInformedConditioning(physics_emb_dim, channels * 2))

        # KANs usually don't need intermediate activations like ReLU because they are splines.
        # But we can add it if needed. For now, rely on KAN non-linearity.

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

        forward method for ComplexKANUnrolledBlock.

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
            h = conv(h)
            h = cond(h, emb)

        x_reg = residual + h

        if measured_kspace is not None and mask is not None:
            x_final = self.dc(x_reg, measured_kspace, mask)
        else:
            x_final = x_reg

        return x_final


@register_model(name="diff_varnet_kan", training_mode="diffusion", spatial_dims=(2,))
class DiffVarNetKAN(GradCheckpointingMixin, nn.Module, IGenerator):
    """
    Diff-VarNet-KAN Backbone.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        image_size: int = 256,
        num_unrolls: int = 5,
        base_channels: int = 64,
        physics_emb_dim: int = 256,
        kan_type: str = "RBF",
        kan_num_grids: int = 4,
        dropout: float = 0.0,
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
            kan_type (str): Description.
            kan_num_grids (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels % 2 != 0:
            raise ValueError(f"DiffVarNetKAN expects even input channels, got {in_channels}")

        self.complex_channels = in_channels // 2

        self.time_pos_enc = nn.Sequential(
            nn.Linear(1, physics_emb_dim // 2),
            nn.SiLU(),
            nn.Linear(physics_emb_dim // 2, physics_emb_dim),
        )

        self.blocks = nn.ModuleList(
            [
                ComplexKANUnrolledBlock(
                    channels=self.complex_channels,
                    features=base_channels,
                    physics_emb_dim=physics_emb_dim,
                    kan_type=kan_type,
                    num_grids=kan_num_grids,
                )
                for _ in range(num_unrolls)
            ]
        )


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

        forward method for DiffVarNetKAN.

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

        curr_x = x
        ckpt = self._checkpointing_active()
        for block in self.blocks:
            curr_x = (
                checkpoint(block, curr_x, emb, measured_kspace, mask, use_reentrant=False)
                if ckpt
                else block(curr_x, emb, measured_kspace, mask)
            )

        return curr_x

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "DiffVarNetKAN"

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
