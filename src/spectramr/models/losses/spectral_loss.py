import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.registry import register_loss


@register_loss(name="log_spectral", aliases=["LogSpectralLoss"], domain="agnostic")
class LogSpectralLoss(nn.Module):
    r"""Log-Spectral Distance Loss.

    **DOMAIN**: AGNOSTIC (works on both k-space and image)
    **Input**: [B, 2, H, W] complex or [B, 1, H, W] image
    **3D Support**: No (2D FFT only)

    Computes distance in log-frequency domain. Works on both domains via
    magnitude spectrum representation.
    Loss = || log(eps + |FFT(pred)|) - log(eps + |FFT(target)|) ||_1, eps = 1 by default.
    With ``skip_fft=True`` the input is taken as the spectrum itself and no FFT is applied.

    The input is never decompressed. When the data path has already
    ``log1p``-compressed the k-space (``enable_log_scaling``, magnitude
    ``m -> log1p(m)``), the term therefore scores ``log(1 + log1p(m))``, a second compression of the
    spectrum rather than the log-spectral distance of the raw data. Its gradient
    with respect to the compressed magnitude is ``1 / (1 + log1p(m))``, so the
    high-energy centre is weighted about 4-7x less than the periphery. The term
    is phase-blind either way.

    Mathematical Formulation:
    .. math::

        \mathcal{L}_{LogSpectral} = \| \log(|\hat{k}| + \epsilon) - \log(|k| + \epsilon) \|_1"""

    def __init__(self, eps: float = 1.0, skip_fft: bool = False):
        """__init__.

        Args:
            eps (float): Description.
            skip_fft (bool): Description.
        """
        super().__init__()
        self.eps = eps
        self.skip_fft = skip_fft

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Log-Spectral loss.

        forward method for LogSpectralLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Handle complex inputs (convert to complex tensor if needed)
        pred_c = self._to_complex(pred)
        target_c = self._to_complex(target)

        # Compute FFT if requested, otherwise assume already in frequency domain
        if not self.skip_fft:
            pred_fft = fft2c(pred_c)
            target_fft = fft2c(target_c)
        else:
            pred_fft = pred_c
            target_fft = target_c

        # Log-Magnitude
        log_pred = torch.log(torch.abs(pred_fft) + self.eps)
        log_target = torch.log(torch.abs(target_fft) + self.eps)

        return F.l1_loss(log_pred, log_target)

    def _to_complex(self, t: torch.Tensor) -> torch.Tensor:
        """Ensures tensor is complex type for separate real/imag or magnitude handling."""
        if torch.is_complex(t):
            return t

        # If multi-channel, assume real/imag
        if t.ndim >= 3 and t.shape[-3] % 2 == 0:
            shape = list(t.shape)
            C = shape[-3]
            spatial_dims = shape[-2:]
            batch_dims = shape[:-3]

            t_reshaped = t.view(*batch_dims, C // 2, 2, *spatial_dims)
            real_part = t_reshaped[..., 0, :, :]
            imag_part = t_reshaped[..., 1, :, :]
            return torch.complex(real_part, imag_part)

        # If 1-channel or other, treat as real part, zero imag
        if t.ndim >= 3 and t.shape[-3] == 1:
            return torch.complex(t.squeeze(-3), torch.zeros_like(t.squeeze(-3)))

        # Flattened/other shapes: treat as real
        return torch.complex(t, torch.zeros_like(t))
