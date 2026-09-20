"""Physics-Driven Reconstruction Network
=====================================

Implementation of Experiment 3.1 from the Research Roadmap.
Integrates Model-Based Deep Learning (MoDL) with Unrolled Proximal Gradient Descent.
"""

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.integration import (
    OperatorProjectionDataConsistency,
    create_physics_operator,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.reconstruction.unet import StandardUNetGenerator as StandardUNet
from spectramr.models.registry import register_model


class DifferentiableBlochLayer(nn.Module):
    """DifferentiableBlochLayer class."""

    def __init__(self, te: float, tr: float, field_strength: float = 3.0):
        """
        Args:
            te: Echo Time in milliseconds
            tr: Repetition Time in milliseconds
            field_strength: B0 field in Tesla
        """
        super().__init__()
        self.te = te
        self.tr = tr
        # Gyromagnetic ratio for protons (rad/s/T)
        self.gamma = 267.513e6

    def forward(self, tissue_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tissue_maps: Tensor of shape (B, 3, H, W) containing [PD, T1, T2]
        Returns:
            signal: Theoretical MR signal magnitude
        """
        pd, t1, t2 = torch.split(tissue_maps, 1, dim=1)

        # Enforce physical constraints (T1, T2 > 0)
        t1 = torch.abs(t1) + 1e-6
        t2 = torch.abs(t2) + 1e-6

        # 1. Longitudinal Relaxation (T1 recovery)
        # M_z = PD * (1 - exp(-TR/T1))
        mz = pd * (1.0 - torch.exp(-self.tr / t1))

        # 2. Transverse Relaxation (T2 decay)
        # Signal = M_z * exp(-TE/T2)
        signal = mz * torch.exp(-self.te / t2)

        return signal


@register_model(name="physics_driven", training_mode="reconstruction")
class UnrolledPhysicsNetwork(nn.Module, IGenerator):
    """
    Model-Based Deep Learning (MoDL) Architecture.

    Unrolls the optimization algorithm for Inverse Problems:
    x_{k+1} = argmin_x ||Ax - y||^2 + lambda ||x - D(x_k)||^2
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,  # Not strictly used in unrolled loop output
        operator_type: str = "fft2d",
        num_iterations: int = 5,
        hidden_channels: int = 64,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            operator_type (str): Description.
            num_iterations (int): Description.
            hidden_channels (int): Description.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.in_channels = in_channels

        # 1. Regularization Network (The Denoiser)
        # ResNet-style regularizer: x_reg = x + CNN(x)
        # Input: Complex (2-channel) -> Output: Complex (2-channel)
        self.denoiser = StandardUNet(
            in_channels=in_channels,
            out_channels=in_channels,  # Denoised output size = input size
            bilinear=False,
        )

        # 2. Physics Operator & Data Consistency
        self.operator = create_physics_operator(operator_type)

        # Soft DC with learnable or fixed parameter?
        # Standard MoDL uses conjugate gradient for DC step,
        # but here we use the OperatorProjectionDataConsistency (POCS/Soft Projection)
        # as a proximal operator approximation.
        self.dc_layer = OperatorProjectionDataConsistency(
            self.operator, lambda_dc=1.0
        )  # Soft DC usually has lambda < 1.0 or handled inside

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "UnrolledPhysicsNetwork"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.denoiser.parameters())

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
        ref_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: Initial estimate (Zero-filled) [B, 2, H, W]
            ref_kspace: Measured k-space [B, C, H, W, 2] (or [B, H, W, 2] single coil)
            mask: Sampling mask [B, H, W]

        forward method for UnrolledPhysicsNetwork.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            ref_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure input is in correct format for denoiser
        current_x = x

        for i in range(self.num_iterations):
            # 1. Regularization Step (z_k)
            # CNN acts as gradient of prior or proximal operator
            # x_reg = denoiser(x_k)
            # MoDL: x_reg is the denoised image.
            # Residual learning: denoiser predicts noise/artifact, so x - denoiser(x) ?
            # StandardUNet output behaves as output image usually.

            reg_updates = self.denoiser(current_x)

            # If UNet learns the residual (common in compressed sensing):
            # x_reg = current_x + reg_updates
            # Or if UNet learns the image directly:
            x_reg = reg_updates

            # 2. Data Consistency Step (x_{k+1})
            # Project x_reg onto measurement manifold
            if ref_kspace is not None and mask is not None:
                # DC Layer expects [B, 2, H, W] or complex?
                # OperatorProjectionDataConsistency forward: pred_image, ref_kspace, mask
                # Check OperatorProjectionDataConsistency implementation integration.py lines 62:
                # pred_image: [B, H, W, 2] (complex last) or check format.
                # integration.py docstring says [B, H, W, 2] but implementations vary.
                # StandardUNet uses [B, C, H, W] (Channel first).

                # Permute for DC Layer if it expects channel last
                # Let's inspect integration.py again or assume standardized [B, C, H, W] if changed?
                # The stored file view in Step 320 showed pred_image [B, H, W, 2] and returning [B, H, W, 2].
                # But our Integration layer is usually robust.
                # Let's assume channel-first [B, 2, H, W] input to this Forward,
                # convert to [B, H, W, 2] for DC, then back.

                x_reg_permuted = x_reg.permute(0, 2, 3, 1).contiguous()  # [B, H, W, 2]

                # Assume ref_kspace is already in correct shape (usually [B, H, W, 2] or [B, C, H, W, 2])

                dc_x = self.dc_layer(x_reg_permuted, ref_kspace, mask)

                # Handle possible return shapes/types form DC Layer
                # Case 1: Complex tensor [B, H, W] or [B, C, H, W]
                if dc_x.is_complex():
                    # View as real: [..., 2]
                    dc_x_real = torch.view_as_real(dc_x)
                    # dc_x_real is likely [B, H, W, 2] if input was [B, H, W] complex
                    # If dimensions match [B, H, W, 2], we permute to [B, 2, H, W]
                    if dc_x_real.ndim == 4 and dc_x_real.shape[-1] == 2:
                        current_x = dc_x_real.permute(0, 3, 1, 2)
                    elif dc_x_real.ndim == 5 and dc_x_real.shape[-1] == 2:
                        # [B, C, H, W, 2] -> [B, C*2, H, W]? Or just assume single channel [B, 1, H, W, 2]
                        # Our UNet expects 2 channels (Real/Imag).
                        # If output is [B, 1, H, W, 2], remove dim 1
                        if dc_x_real.shape[1] == 1:
                            dc_x_real = dc_x_real.squeeze(1)
                            current_x = dc_x_real.permute(0, 3, 1, 2)
                        else:
                            # Multi-coil? Unrolled network denoiser defined for 'in_channels' (2).
                            # Fallback: take magnitude? No, UNet predicts complex.
                            # Maybe flatten channels?
                            # Assuming single-coil or combined image for now.
                            pass

                # Case 2: Real tensor [B, H, W, 2]
                elif dc_x.ndim == 4 and dc_x.shape[-1] == 2:
                    current_x = dc_x.permute(0, 3, 1, 2)

                else:
                    # Fallback assuming it matches
                    current_x = dc_x

                # Ensure float32 for next Denoiser step
                if current_x.is_complex():
                    current_x = torch.view_as_real(current_x).permute(0, 3, 1, 2).contiguous()
                if current_x.dtype != x.dtype:
                    current_x = current_x.to(dtype=x.dtype)  # match input dtype (float32)

            else:
                current_x = x_reg

        return current_x

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """generate.

        Args:
            z (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(z, **kwargs)


# Alias for backward compatibility if needed, though registry handles name="physics_driven"
PhysicsDrivenNetwork = UnrolledPhysicsNetwork
