"""SENSE (Sensitivity Encoding) physics operations for MRI reconstruction.

Provides tools for explicit SENSE-manifold projections and loss computations,
bypassing SVD dimensionality reduction in favor of raw physical coil data
combined with optimal spatial matched filtering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SENSESubspaceProjector(nn.Module):
    """Orthogonally projects hallucinated multi-coil k-space onto the physical SENSE manifold.

    Forces neural network outputs to respect the hardware geometries of the
    receive coils by applying Adjoint -> Forward SENSE combinations.
    """

    def __init__(self):
        """__init__."""
        super().__init__()

    def forward(
        self, kspace_pred: torch.Tensor, smaps: torch.Tensor, *, log_scaled: bool = False
    ) -> torch.Tensor:
        """Project multi-coil k-space onto true SENSE subspace.

        The projection is linear-domain physics: a true multi-coil image is
        ``s_c * m`` at every pixel, and only on linear k-space does the inverse
        FFT return those coil images. On ``log1p``-compressed k-space it changed an
        exact prediction by NRMSE 0.46 (#2488), so ``log_scaled`` input is
        decompressed, projected and compressed again.

        Args:
            kspace_pred: K-space output from network [B, C, H, W] (real-stacked or complex)
            smaps: Spatial coil sensitivity maps [B, C, H, W] (real-stacked or complex)
            log_scaled: Whether ``kspace_pred`` is ``log1p``-compressed in magnitude.

        Returns:
            Projected k-space [B, C, H, W] mathematically consistent with coil geometries.

        forward method for SENSESubspaceProjector.

        Executes PyTorch tensor operations.

        Args:
            kspace_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            smaps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if log_scaled:
            from spectramr.data.transforms.normalization import (
                compress_kspace_log,
                decompress_kspace_log,
            )

            linear = decompress_kspace_log(kspace_pred, channel_dim=1)
            return compress_kspace_log(self.forward(linear, smaps), channel_dim=1)

        # Ensure smaps and kspace are complex for FFT ops
        is_real_stacked = False
        k_pred_comp = kspace_pred
        s_comp = smaps

        if not torch.is_complex(kspace_pred):
            is_real_stacked = True
            b, c, h, w = kspace_pred.shape
            c_comp = c // 2
            k_pred_comp = torch.view_as_complex(
                kspace_pred.view(b, c_comp, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
            )

        if not torch.is_complex(smaps):
            b, c, h, w = smaps.shape
            c_comp = c // 2
            s_comp = torch.view_as_complex(
                smaps.view(b, c_comp, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
            )

        # 1. Normalize S-maps so Sum(|S_c|^2) = 1 (Energy preservation)
        rss = torch.sqrt(torch.sum(s_comp.abs() ** 2, dim=1, keepdim=True) + 1e-8)
        smaps_norm = s_comp / rss

        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        # 2. Transform to spatial domain
        img_coils = ifft2c(k_pred_comp)

        # 3. SENSE-Adjoint: Collapse to optimal physical magnetization
        img_combined = torch.sum(img_coils * smaps_norm.conj(), dim=1, keepdim=True)

        # 4. SENSE-Forward: Re-expand to ideal multi-coil images
        img_coils_projected = img_combined * smaps_norm

        # 5. Return to K-Space
        kspace_projected = fft2c(img_coils_projected)

        if is_real_stacked:
            b, c_comp, h, w = kspace_projected.shape
            # Return identical format
            kspace_projected = (
                torch.view_as_real(kspace_projected)
                .permute(0, 1, 4, 2, 3)
                .reshape(b, c_comp * 2, h, w)
            )

        return kspace_projected


def compute_sense_loss(
    kspace_pred: torch.Tensor, kspace_target: torch.Tensor, smaps: torch.Tensor
) -> torch.Tensor:
    """Compute structural L1 loss strictly on the SENSE-Adjoint Combined Image.

    By phase-aligning and collapsing the multi-coil arrays before computing loss,
    the network avoids penalizing independent thermal AWGN state-spaces per coil.

    Args:
        kspace_pred: Predicted k-space [B, C, H, W]
        kspace_target: Ground truth k-space [B, C, H, W]
        smaps: Spatial coil sensitivity maps [B, C, H, W]

    Returns:
        Scalar L1 loss over the stacked real/imag channels of the
        SENSE-adjoint-combined image (penalizes the real and imaginary
        parts independently via ``torch.view_as_real``; this is the
        component-wise L1 ``|dRe|+|dImag|``, not the complex-modulus L1
        ``mean(|z_pred - z_target|)``).
    """
    if not torch.is_complex(smaps):
        b, c, h, w = smaps.shape
        c_comp = c // 2
        smaps_comp = torch.view_as_complex(
            smaps.view(b, c_comp, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
        )
    else:
        smaps_comp = smaps

    rss = torch.sqrt(torch.sum(smaps_comp.abs() ** 2, dim=1, keepdim=True) + 1e-8)
    smaps_norm = smaps_comp / rss

    # Handle real-stacked complex
    if not torch.is_complex(kspace_pred):
        b, c, h, w = kspace_pred.shape
        c_complex = c // 2
        k_pred_comp = torch.view_as_complex(
            kspace_pred.view(b, c_complex, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
        )
        k_target_comp = torch.view_as_complex(
            kspace_target.view(b, c_complex, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
        )
    else:
        k_pred_comp = kspace_pred
        k_target_comp = kspace_target

    from spectramr.infrastructure.physics.fft_ops import ifft2c

    pred_img = ifft2c(k_pred_comp)
    target_img = ifft2c(k_target_comp)

    # Adjoint combination
    pred_sense = torch.sum(pred_img * smaps_norm.conj(), dim=1, keepdim=True)
    target_sense = torch.sum(target_img * smaps_norm.conj(), dim=1, keepdim=True)

    # Compute loss on anatomical contrast
    return F.l1_loss(torch.view_as_real(pred_sense), torch.view_as_real(target_sense))
