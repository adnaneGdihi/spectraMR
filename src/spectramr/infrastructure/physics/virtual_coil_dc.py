from typing import ClassVar

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


class VirtualCoilDataConsistency(nn.Module):
    """
    Virtual-Coil Data Consistency (DC) Block for the SIREN-VarNet architecture.

    Dynamically injects the full-grid SIREN-predicted complex field map \\hat{S}_{\theta}(\\mathbf{r})
    as a virtual sensitivity map into the unrolled DC block.

    Equation:
    x_DC^{(k)} = x^{(k-1)} - \alpha \\hat{S}_{\theta}^* \\odot \\mathcal{F}^H \\mathcal{P}^T
                 \\left( \\mathcal{P} \\mathcal{F} ( \\hat{S}_{\theta} \\odot x^{(k-1)} ) - y \right)
    """

    #: Recorded as a ``post`` stage of the debug snapshot (``snapshot.hooks``).
    snapshot_stage: ClassVar[str] = "post"

    def __init__(self, init_alpha: float = 0.5):
        """__init__.

        Args:
            init_alpha (float): Description.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(
        self,
        x_current: torch.Tensor,
        y_measured: torch.Tensor,
        mask: torch.Tensor,
        S_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_current: Current image estimate [B, 2, H, W] (real/imag stacked)
            y_measured: Measured k-space [B, 2, K] or [B, 2, H, W]
            mask: Undersampling mask [B, 1, H, W]
            S_map: SIREN-predicted complex field map [B, 2, H, W] (real/imag stacked)
        Returns:
            x_dc: Data-consistent image estimate [B, 2, H, W]

        forward method for VirtualCoilDataConsistency.

        Executes PyTorch tensor operations.

        Args:
            x_current (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y_measured (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            S_map (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Fail-fast contract check: this block parses channel-0/1 as real/imag,
        # so every tensor must be the documented real/imag-stacked [B, 2, H, W]
        # layout. A complex tensor or a non-2-channel tensor would either crash
        # (index 1 out of range) or be silently mis-parsed; raise instead.
        for name, t in (
            ("x_current", x_current),
            ("y_measured", y_measured),
            ("S_map", S_map),
        ):
            if torch.is_complex(t):
                raise ValueError(
                    f"VirtualCoilDataConsistency expects real/imag-stacked "
                    f"[B, 2, ...] tensors, but {name} is already complex "
                    f"(shape {tuple(t.shape)}). Pass torch.stack([t.real, "
                    f"t.imag], dim=1) instead."
                )
            if t.dim() < 2 or t.shape[1] != 2:
                raise ValueError(
                    f"VirtualCoilDataConsistency expects {name} with exactly 2 "
                    f"channels (real, imag) at dim=1, got shape {tuple(t.shape)}."
                )

        # Convert to complex tensors
        x_c = torch.complex(x_current[:, 0], x_current[:, 1])
        y_c = torch.complex(y_measured[:, 0], y_measured[:, 1])
        S_c = torch.complex(S_map[:, 0], S_map[:, 1])

        # Ensure mask is broadcastable
        if mask.dim() == 4 and mask.shape[1] == 1:
            mask_c = mask[:, 0]
        else:
            mask_c = mask

        # Forward operator: P * F * (S * x)
        S_x = S_c * x_c
        k_pred = fft2c(S_x)
        # Apply mask
        k_pred_sampled = k_pred * mask_c

        # Residual in k-space
        k_resid = k_pred_sampled - y_c

        # Adjoint operator: S^* * F^H * P^T * (residual)
        # Since mask is symmetric/diagonal, P^T = P
        resid_img = ifft2c(k_resid * mask_c)
        S_conj = torch.conj(S_c)
        grad_x = S_conj * resid_img

        # Update
        x_next = x_c - self.alpha * grad_x

        # Return to real/imag stacked format
        return torch.stack([x_next.real, x_next.imag], dim=1)
