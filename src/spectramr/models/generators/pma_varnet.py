"""Physics-Driven Marker-Anchored Variational Network (PMA-VarNet).

Implements the Task 5 unrolled cascade topology using Operator A_m4.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

try:
    from spectramr.infrastructure.physics.m4_operator import Operator_A_m4

    M4_OPERATOR_AVAILABLE = True
except ImportError:
    M4_OPERATOR_AVAILABLE = False

from spectramr.models.blocks.swin import SwinBlock
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)


class Complex3DRegularizer(nn.Module):
    """Abstracts the Task 1 MoDL-SR 3D Swin-Transformer block.

    Replaces the previous Conv3d placeholder with Swin Transformer blocks
    to perform attention across the spatial dimensions while treating
    contrasts (and real/imag parts) as embedding channels.
    """

    def __init__(self, num_contrasts: int = 1, hidden_dim: int = 64):
        super().__init__()
        # We treat the concatenated real and imaginary parts as channels.
        in_ch = num_contrasts * 2

        self.proj_in = nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1)

        # Swin blocks for spatial attention
        self.blocks = nn.ModuleList(
            [
                SwinBlock(dim=hidden_dim, num_heads=4, window_size=8, shift_size=0),
                SwinBlock(dim=hidden_dim, num_heads=4, window_size=8, shift_size=4),
            ]
        )

        self.proj_out = nn.Conv2d(hidden_dim, in_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply regularization via Swin Attention.

        Args:
            x: Complex or real image [B, N_con, H, W]

        Returns:
            Regularized image [B, N_con, H, W] (same dtype as input)

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams.
        """
        B, N_con, H, W = x.shape
        input_was_complex = x.is_complex()

        if input_was_complex:
            # Split complex → real/imag channels for 2D Convs
            # [B, N_con, H, W] -> [B, N_con*2, H, W]
            x_ri = torch.cat([x.real, x.imag], dim=1)
        else:
            # Real-valued input: pad with zeros to match in_ch
            x_ri = torch.cat([x, torch.zeros_like(x)], dim=1)

        # 1. Project to hidden dimension
        feat = self.proj_in(x_ri)

        # 2. Reshape for SwinBlock: [B, C, H, W] -> [B, H*W, C]
        feat = feat.flatten(2).transpose(1, 2)

        # 3. Apply Attention
        for blk in self.blocks:
            feat = blk(feat, input_resolution=(H, W))

        # 4. Reshape back: [B, H*W, C] -> [B, C, H, W]
        feat = feat.transpose(1, 2).view(B, -1, H, W)

        # 5. Project out to real/imag channels
        out_ri = self.proj_out(feat)

        if input_was_complex:
            out_complex = torch.complex(out_ri[:, :N_con], out_ri[:, N_con:])
            return out_complex
        else:
            return out_ri[:, :N_con]


@register_model(
    "pma_varnet",
    "reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=True,
)
@register_model(
    "jfe_varnet",
    "reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=True,
)
class PMAVarNet(nn.Module):
    """Physics-Driven Marker-Anchored Variational Network.

    Implements a K-cascade unrolled optimization ``x <- R_k(x - eta_k A^H(Ax - y))`` with
    the forward operator A_m4 built from ``maps_dict``. The operator models only the maps it
    is given; any corruption it is not given (motion, B0) is left to the regularizers.

    Args:
        in_channels: Number of input contrasts (N_con).
        out_channels: Number of output contrasts (N_con).
        num_cascades: Number of unrolled iterations (K).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_cascades: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_cascades = num_cascades
        self.in_channels = in_channels
        self.out_channels = out_channels

        # The exact mathematical physics operator
        if M4_OPERATOR_AVAILABLE:
            self.operator = Operator_A_m4()
        else:
            self.operator = None

        # Learnable step size parameter per cascade
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(num_cascades)])

        # Deep CNN Regularizers per cascade (or shared, here shared per cascade for capacity)
        self.regularizers = nn.ModuleList(
            [Complex3DRegularizer(num_contrasts=in_channels) for _ in range(num_cascades)]
        )

    def forward(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute the PMA-VarNet cascade.

        Args:
            x: Raw corrupted k-space Measurements y [B, Nc, N_k, N_con] or similar.
               For this implementation, we expect y in kwargs['y_meas'] if x is zero-filled.
               Or x is `y_meas` [B, N_con, Nc, N_k].
            kwargs: Must contain 'maps_dict' holding physics parameters.
                    Optionally 'y_meas'.

        Returns:
            Reconstructed image [B, N_con, H, W].

        forward method for PMAVarNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Parse inputs
        y_meas = kwargs.get("y_meas", x)  # [B, N_con, Nc, N_k]
        maps_dict = kwargs.get("maps_dict", {})

        if self.operator is None:
            raise RuntimeError("Operator_A_m4 is required for PMAVarNet.")

        B, N_con, Nc, N_k = y_meas.shape

        # Initial guess x^(0) = A^H y
        x_est = []
        for c in range(N_con):
            # Adjoint handles [B, Nc, N_k] -> [B, 1, H, W]
            x_c = self.operator.adjoint(y_meas[:, c], maps_dict)
            x_est.append(x_c.squeeze(1))  # [B, H, W]

        x_n = torch.stack(x_est, dim=1)  # [B, N_con, H, W]

        # Unrolled cascade
        for k in range(self.num_cascades):
            # 1. Data Consistency Gradient Calculation
            g_n_list = []
            for c in range(N_con):
                # A x_n
                Ax = self.operator.forward(x_n[:, c : c + 1], maps_dict)  # [B, Nc, N_k]

                # A^H (A x_n - y)
                grad = self.operator.adjoint(Ax - y_meas[:, c], maps_dict)  # [B, 1, H, W]
                g_n_list.append(grad.squeeze(1))

            g_n = torch.stack(g_n_list, dim=1)  # [B, N_con, H, W]

            # 2. CNN Regularization step
            # x_cnn = Regularizer( x_n - eta * g_n )
            x_update = x_n - self.eta[k] * g_n
            x_cnn = self.regularizers[k](x_update)

            x_n = x_cnn

        return x_n

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        # Out shape is [B, in_channels, H, W]
        # H, W must be derived, return generic
        return (input_shape[0], self.in_channels, 256, 256)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(x, **kwargs)
