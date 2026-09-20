"""Unrolled Reconstruction Models for MRI
======================================

Implementation of unrolled reconstruction networks like MoDL (Model-based
Deep Learning) and variational networks for physics-informed MRI
reconstruction.

These models alternate between data consistency (DC) steps and learned
regularization (CNN updates) in an unrolled optimization framework.
"""

import torch
from torch import nn
from torch.nn import functional as F

from spectramr.infrastructure.physics.forward_operator import (
    MultiCoilDataConsistency,
    MultiCoilForwardOperator,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class RegularizationBlock(nn.Module):
    """Learned regularization block for unrolled reconstruction.

    A CNN that learns to perform regularization/update steps.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        activation: str = "relu",
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (int): Description.
            num_layers (int): Description.
            kernel_size (int): Description.
            activation (str): Description.
        """
        super().__init__()

        layers = []
        # Input convolution
        layers.append(
            nn.Conv2d(in_channels, features, kernel_size, padding=kernel_size // 2),
        )
        layers.append(nn.BatchNorm2d(features))

        if activation == "relu":

            def _make_act() -> nn.Module:
                return nn.ReLU(inplace=True)

        elif activation == "leaky_relu":

            def _make_act() -> nn.Module:
                return nn.LeakyReLU(0.2, inplace=True)

        else:
            raise ValueError(
                f"Unknown activation {activation!r} for RegularizationBlock; "
                "expected one of {'relu', 'leaky_relu'}"
            )

        layers.append(_make_act())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(features, features, kernel_size, padding=kernel_size // 2),
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(_make_act())

        # Output convolution
        if num_layers > 1:
            layers.append(
                nn.Conv2d(features, out_channels, kernel_size, padding=kernel_size // 2),
            )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for RegularizationBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.model(x)


@register_model(
    name="unrolled_reconstruction",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=True,
)
class UnrolledReconstructionGenerator(nn.Module, IGenerator):
    """Unrolled reconstruction network (MoDL-style).

    Alternates between data consistency layers and learned regularization
    blocks in an unrolled optimization framework.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: tuple[int, int] = (256, 256),
        num_unrolls: int = 5,
        features: int = 64,
        kernel_size: int = 3,
        lambda_dc: float = 1.0,
        num_coils: int = 1,
        activation: str = "relu",
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            img_size (tuple[int, int]): Description.
            num_unrolls (int): Description.
            features (int): Description.
            kernel_size (int): Description.
            lambda_dc (float): Description.
            num_coils (int): Description.
            activation (str): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.num_unrolls = num_unrolls
        self.num_coils = num_coils

        # Initial reconstruction (e.g., zero-filled reconstruction)
        self.initial_recon = nn.Conv2d(in_channels, out_channels, 1)

        # Unified Physics Operator
        self.forward_operator = MultiCoilForwardOperator(
            coil_sensitivities=None,  # Dynamic
            mask=None,  # Dynamic
        )

        # Unrolled blocks
        # Uses shared forward operator for all steps
        self.dc_layers = nn.ModuleList(
            [
                MultiCoilDataConsistency(self.forward_operator, mode="soft", lambda_dc=lambda_dc)
                for _ in range(num_unrolls)
            ],
        )

        self.regularization_blocks = nn.ModuleList(
            [
                RegularizationBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    features=features,
                    kernel_size=kernel_size,
                    activation=activation,
                )
                for _ in range(num_unrolls)
            ],
        )

        # Final output activation
        self.output_activation = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
        kspace_measured: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through unrolled reconstruction.

        Args:
            x: Low-resolution input [B, C, H, W]
            kspace_measured: Measured k-space data [B, C, H, W]
            mask: Sampling mask [B, 1, H, W] or [H, W]
            sensitivity_maps: Coil sensitivity maps [B, num_coils, H, W]

        Returns:
            Reconstructed high-resolution image [B, C, H, W]

        forward method for UnrolledReconstructionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            kspace_measured (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            sensitivity_maps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Initial reconstruction
        x = self.initial_recon(x)

        # Unrolled optimization
        for dc_layer, reg_block in zip(self.dc_layers, self.regularization_blocks, strict=False):
            # Learned regularization step
            x = reg_block(x)

            # Data consistency step (if k-space data available)
            # Data consistency step (if k-space data available)
            if kspace_measured is not None and mask is not None:
                # Update operator state dynamically
                self.forward_operator.set_coil_sensitivities(sensitivity_maps)
                self.forward_operator.set_mask(mask)

                # Apply DC
                x = dc_layer(x, kspace_measured)

        # Final activation
        return self.output_activation(x)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate method for IGenerator interface."""
        kspace_measured = kwargs.get("kspace_measured")
        mask = kwargs.get("mask")
        sensitivity_maps = kwargs.get("sensitivity_maps")

        return self.forward(
            x,
            kspace_measured=kspace_measured,
            mask=mask,
            sensitivity_maps=sensitivity_maps,
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "unrolled_reconstruction"

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        return (input_shape[0], self.out_channels, self.img_size[0], self.img_size[1])

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@register_model(name="varnet", training_mode="reconstruction", spatial_dims=(2,))
class VariationalNetworkGenerator(nn.Module, IGenerator):
    """Variational Network for MRI reconstruction.

    Similar to unrolled reconstruction but with learned gradient steps
    and physics-informed regularization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        img_size: tuple[int, int] = (256, 256),
        num_stages: int = 5,
        features: int = 64,
        step_size: float = 0.1,
        num_coils: int = 1,
        output_activation: str | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            img_size (tuple[int, int]): Description.
            num_stages (int): Description.
            features (int): Description.
            step_size (float): Description.
            num_coils (int): Description.
            output_activation (str | None): Description.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.num_stages = num_stages
        self.step_size = step_size
        self.num_coils = num_coils

        from spectramr.models.layers.complex_conv import ComplexConv2d

        # Initial reconstruction (literal pytorch channel-count convention).
        self.initial_recon = nn.Conv2d(in_channels, out_channels, 1)

        # ComplexAwareBlock wraps ``ComplexConv2d``, which counts COMPLEX
        # coils in its ``in_channels`` arg. The caller below passes
        # ``out_channels`` (the real-channel count from the YAML) — so we
        # convert with ``// 2`` here. Fixes smoke crash at iter 1 with
        # ``weight of size [32, 4, 3, 3], expected input[4, 2, 256, 256]
        # to have 4 channels`` (experiment_38_parallel_imaging_deep_sense,
        # smoke 2026-05-10).
        class ComplexAwareBlock(nn.Module):
            def __init__(self, in_c, out_c, features=32):
                super().__init__()
                if in_c % 2 != 0 or out_c % 2 != 0:
                    raise ValueError(
                        "ComplexAwareBlock requires even in_c / out_c "
                        f"(real/imag pairs); got in_c={in_c}, out_c={out_c}"
                    )
                self.conv1 = ComplexConv2d(in_c // 2, features, 3, padding=1)
                self.conv2 = ComplexConv2d(features, features, 3, padding=1)
                self.conv3 = ComplexConv2d(features, out_c // 2, 3, padding=1)

            def forward(self, x):
                """forward method for ComplexAwareBlock.

                Executes PyTorch tensor operations.

                Args:
                    x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

                Returns:
                    torch.Tensor: Output tensor with shape matching the operation.

                Hardware/Device Context:
                    Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
                is_complex = x.is_complex()
                # ComplexConv2d handles complex automatically but returns interleaved real tensor
                # So we must map it back to complex if necessary
                h = self.conv1(x)
                h = F.relu(h)
                h = self.conv2(h)
                h = F.relu(h)
                h = self.conv3(h)
                if is_complex:
                    h = torch.complex(h[:, 0::2], h[:, 1::2])
                return h

        # Variational stages (learned gradient steps)
        # Upgraded to ComplexAware block to handle k-space/image phase robustly.
        self.variational_stages = nn.ModuleList(
            [ComplexAwareBlock(out_channels, out_channels, features=16) for _ in range(num_stages)],
        )

        # Unified Physics Operator
        self.forward_operator = MultiCoilForwardOperator(
            coil_sensitivities=None,
            mask=None,
        )

        # Data consistency operator
        self.dc_operator = MultiCoilDataConsistency(self.forward_operator, mode="soft", lambda_dc=1.0)

        # Output activation (configurable, defaults to None for k-space data)
        if output_activation is None or output_activation.lower() == "none":
            self.output_activation = nn.Identity()
        elif output_activation.lower() == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation.lower() == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation.lower() == "relu":
            self.output_activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unknown output_activation '{output_activation}'. "
                "Valid options: None, 'none', 'tanh', 'sigmoid', 'relu'."
            )

    def forward(
        self,
        x: torch.Tensor,
        kspace_measured: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through variational network.

        forward method for VariationalNetworkGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            kspace_measured (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            sensitivity_maps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Input validation and comprehensive normalization
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros(
                x.shape[0],
                self.out_channels,
                x.shape[2],
                x.shape[3],
                dtype=x.dtype,
                device=x.device,
            )

        # Normalize input: k-space can have very large values
        # Use robust normalization: RMS (root mean square) based approach
        # This avoids quantile() size limitations on large tensors
        rms = torch.sqrt(torch.mean(x**2)) + 1e-8
        # Scale by RMS then apply conservative factor
        lr_input_normalized = x / (rms * 100.0)  # Very conservative scaling

        # Ensure normalized input is in safe range
        lr_input_normalized = torch.clamp(lr_input_normalized, min=-1.0, max=1.0)

        # Initial reconstruction with properly normalized input
        x = self.initial_recon(lr_input_normalized)

        # Ensure initial output is safe
        x = torch.clamp(x, min=-10.0, max=10.0)

        # Variational stages with adaptive step size
        for stage_idx, stage in enumerate(self.variational_stages):
            # Learned gradient step
            gradient = stage(x)

            # Clamp gradient to reasonable range
            gradient = torch.clamp(gradient, min=-5.0, max=5.0)

            # Very conservative step size for stability
            adaptive_step = self.step_size * 0.01 * (1.0 / (stage_idx + 1))
            x = x - adaptive_step * gradient

            # Clamp to prevent numerical explosion
            x = torch.clamp(x, min=-10.0, max=10.0)

            # Data consistency (if available)
            # Data consistency (if available)
            if kspace_measured is not None and mask is not None:
                self.forward_operator.set_coil_sensitivities(sensitivity_maps)
                self.forward_operator.set_mask(mask)
                x = self.dc_operator(x, kspace_measured)

        return self.output_activation(x)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate method for IGenerator interface."""
        kspace_measured = kwargs.get("kspace_measured")
        mask = kwargs.get("mask")
        sensitivity_maps = kwargs.get("sensitivity_maps")

        return self.forward(
            x,
            kspace_measured=kspace_measured,
            mask=mask,
            sensitivity_maps=sensitivity_maps,
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "variational_network"

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        return (input_shape[0], self.out_channels, self.img_size[0], self.img_size[1])

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unrolled_reconstruction_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    img_size: tuple[int, int] = (256, 256),
    num_unrolls: int = 5,
    features: int = 64,
    kernel_size: int = 3,
    lambda_dc: float = 1.0,
    num_coils: int = 1,
    activation: str = "relu",
    **kwargs,
) -> UnrolledReconstructionGenerator:
    """Factory function for unrolled reconstruction generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        img_size: Image size
        num_unrolls: Number of unrolled iterations
        features: Number of features in regularization block
        kernel_size: Kernel size for convolution
        lambda_dc: Data consistency weight
        num_coils: Number of coils
        activation: Activation function
        **kwargs: Additional arguments (ignored)

    Returns:
        UnrolledReconstructionGenerator instance

    """
    return UnrolledReconstructionGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        num_unrolls=num_unrolls,
        features=features,
        kernel_size=kernel_size,
        lambda_dc=lambda_dc,
        num_coils=num_coils,
        activation=activation,
    )


def create_variational_network_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    img_size: tuple[int, int] = (256, 256),
    num_stages: int = 5,
    features: int = 64,
    step_size: float = 0.1,
    num_coils: int = 1,
    **kwargs,
) -> VariationalNetworkGenerator:
    """Factory function for variational network generator.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        img_size: Image size
        num_stages: Number of variational stages
        features: Number of features
        step_size: Step size for gradient update
        num_coils: Number of coils
        **kwargs: Additional arguments (ignored)

    Returns:
        VariationalNetworkGenerator instance

    """
    return VariationalNetworkGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        num_stages=num_stages,
        features=features,
        step_size=step_size,
        num_coils=num_coils,
    )
