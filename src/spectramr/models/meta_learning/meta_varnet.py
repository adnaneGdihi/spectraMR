"""Meta-VarNet: Variational Network for Meta-Learning.

Implements an unrolled Variational Network (VarNet) structure that supports
stateless functional calls for efficient meta-learning steps (MAML).
"""

import torch
import torch.nn as nn
from jaxtyping import Complex, Float
from torch import Tensor

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.registry import register_model


class NormUnet(nn.Module):
    """Normalized U-Net for VarNet Regularization."""

    def __init__(self, in_ch: int, out_ch: int, ch: int = 18):
        """__init__.

        Args:
            in_ch (int): Description.
            out_ch (int): Description.
            ch (int): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.InstanceNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        Args:
            x (Tensor): Description.
        Returns:
            Tensor: Description.

        forward method for NormUnet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


class DataConsistency(nn.Module):
    """Data Consistency Layer in k-space."""

    def __init__(self, noise_lvl: float | None = None):
        """__init__.

        Args:
            noise_lvl (Optional[float]): Description.
        """
        super().__init__()
        self.noise_lvl = nn.Parameter(torch.tensor(0.1)) if noise_lvl is None else noise_lvl

    def forward(
        self,
        k_curr: Complex[Tensor, "B C H W"],
        k_obs: Complex[Tensor, "B C H W"],
        mask: Float[Tensor, "B 1 H W"],
    ) -> Complex[Tensor, "B C H W"]:
        """
        Argmin_z ||z - x||^2 + lambda ||Pz - y||^2
        Solution: z = (x + lambda P^T y) / (1 + lambda P^T P)

        forward method for DataConsistency.

        Executes PyTorch tensor operations.

        Args:
            k_curr (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            k_obs (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Complex[Tensor, 'B C H W']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        lam = (
            torch.exp(self.noise_lvl)
            if isinstance(self.noise_lvl, nn.Parameter)
            else self.noise_lvl
        )

        # Consistent k-space
        # k_out = (k_curr + lam * k_obs * mask) / (1 + lam * mask)

        numerator = k_curr + lam * k_obs * mask
        denominator = 1.0 + lam * mask

        return numerator / denominator


@register_model(name="meta_varnet", training_mode="meta_learning")
class MetaVarNet(nn.Module):
    """Variational Network compatible with Functional MAML usage.

    Structure:
    Series of Cascades. Each Cascade:
    1. Image Space: x_k = x_{k-1} - eta * CNN(x_{k-1})
    2. K-Space: k_k = DC(FFT(x_k), k_obs)
    3. x_k = IFFT(k_k)
    """

    def __init__(self, num_cascades: int = 6, channels: int = 18):
        """__init__.

        Args:
            num_cascades (int): Description.
            channels (int): Description.
        """
        super().__init__()
        self.num_cascades = num_cascades

        self.cascades = nn.ModuleList([NormUnet(2, 2, channels) for _ in range(num_cascades)])
        # Shared or separate DC per cascade? Usually separate learned weights.
        self.dcs = nn.ModuleList([DataConsistency() for _ in range(num_cascades)])

    def forward(
        self,
        masked_kspace: Complex[Tensor, "B C H W"],
        mask: Float[Tensor, "B 1 H W"],
        params: dict[str, Tensor] | None = None,
    ) -> Complex[Tensor, "B C H W"]:
        """Forward pass with optional functional parameters for MAML.

        Args:
            masked_kspace: Subsampled k-space.
            mask: Sampling mask.
            params: Dictionary of parameters for functional call (optional).

        forward method for MetaVarNet.

        Executes PyTorch tensor operations.

        Args:
            masked_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            params (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Complex[Tensor, 'B C H W']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # If params provided, we use torch.func.functional_call (or manual)
        # Assuming we use the context of `torch.func` outside or explicit replacement.
        # But `torch.func` functional_call is the modern way.
        # Since we are implementing the module logic, we just handle standard forward
        # unless params are injected via functional_call wrapper in the MAML loop.
        # So standard forward is sufficient here.

        # Initial estimate zero-filled
        k_curr = masked_kspace.clone()
        x_curr = ifft2c(k_curr)  # Use physics module for proper centering

        for i, (net, dc) in enumerate(zip(self.cascades, self.dcs, strict=False)):
            # 1. Regularization (CNN)
            # x_curr: [B, C, H, W] complex
            # view_as_real: [B, C, H, W, 2]
            # Permute to [B, C, 2, H, W] then flatten to [B, 2*C, H, W]
            x_in = torch.view_as_real(x_curr).permute(0, 1, 4, 2, 3).contiguous()
            B, C, H, W = x_curr.shape
            x_in = x_in.view(B, C * 2, H, W)

            grads = net(x_in)  # [B, 2*C, H, W]

            # Reshape back to [B, C, H, W, 2]
            grads = grads.view(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
            grads_complex = torch.view_as_complex(grads)

            x_reg = x_curr - grads_complex

            # 2. Data Consistency
            k_reg = fft2c(x_reg)  # Use physics module for proper centering
            k_curr = dc(k_reg, masked_kspace, mask)

            x_curr = ifft2c(k_curr)  # Use physics module for proper centering

        return x_curr
