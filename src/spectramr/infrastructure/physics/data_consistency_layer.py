"""Data Consistency Layer for MRI Reconstruction.
==============================================

This module implements explicit Data Consistency (DC) layers that enforce
fidelity to measured k-space data within the network.
"""

import logging

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.dc_mask import align_dc_mask
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

logger = logging.getLogger(__name__)


class MaskedReplacementDataConsistency(nn.Module):
    """
    Physics-Informed Data Consistency Layer.

    Enforces that the reconstructed image's k-space matches the measured
    k-space data at sampled frequencies.

    Operation:
        x_out = iFFT( (1 - M) * FFT(x_in) + M * y )

    where:
        x_in: Input image guess
        M: Sampling mask (1 at sampled locs, 0 elsewhere)
        y: Measured k-space (0 at unsampled locs)
    """

    def __init__(self, noise_robust: bool = False):
        """
        Args:
            noise_robust: If True, uses a softer consistency (e.g. weighted average)
                          Currently strictly enforcing hard consistency as per standard DC.
        """
        super().__init__()
        self.noise_robust = noise_robust

    def forward(
        self, image: torch.Tensor, measured_kspace: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Data Consistency.

        Args:
            image: Input image guess [B, C, H, W] (Complex-valued or 2-channel real)
            measured_kspace: Measured k-space [B, C, H, W] (Same format as image)
            mask: Sampling mask [B, 1, H, W] or broadcastable

        Returns:
            Corrected image [B, C, H, W]

        forward method for MaskedReplacementDataConsistency.

        Executes PyTorch tensor operations.

        Args:
            image (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            measured_kspace (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Transform Image -> K-Space
        # Handle complex vs stacked-real input
        is_complex = torch.is_complex(image)

        if is_complex:
            k_guessed = fft2c(image)
        else:
            # Assume 2-channel real represents complex
            # We need to treat it as complex for FFT operations
            # Check for interleaved real/imag channels (e.g., [B, 2*Coils, H, W])
            if image.shape[1] % 2 == 0:
                img_complex = torch.complex(image[:, 0::2, ...], image[:, 1::2, ...])
                k_guessed_complex = fft2c(img_complex)
                # Keep distinct for now
            else:
                # If C is odd, we can't easily do DC
                # unless we have measured k-space for those features (unlikely)
                # or if we treat channels as independent modification (Batchelor et al).
                # For standard MRI DC, we usually operate on the "image" channels.
                # If this layer is used inside a network with C > 2 hidden channels,
                # we technically shouldn't apply DC unless those channels map to image space.
                # Assuming this is used at "image-like" stages or C=2.
                # If C > 2, we might verify if it's pairs.
                raise ValueError(
                    f"MaskedReplacementDataConsistency expects 2-channel (Re/Im) or Complex input, got {image.shape}"
                )

        # Ensure mask is float for arithmetic (bool tensors don't support subtraction)
        if mask.dtype == torch.bool:
            mask = mask.float()

        # A mask shaped like the interleaved input rather than [B, 1, H, W] does not
        # broadcast against the complex blend below; ``align_dc_mask`` owns that
        # reduction for every DC layer (the 2026-05-10 diff_varnet crash is its
        # first entry).
        target_channels = k_guessed_complex.shape[1] if not is_complex else k_guessed.shape[1]
        mask = align_dc_mask(mask, target_channels)

        # 2. Apply Consistency
        # For complex tensor:
        if is_complex:
            # Hard DC: Replace sampled freq with measured
            # k_out = k_guessed * (1 - mask) + measured_kspace * mask
            # Note: measured_kspace should already be masked (0 at unsampled), but * mask is safer.
            k_corrected = k_guessed * (1.0 - mask) + measured_kspace * mask

            # 3. Transform K-Space -> Image
            image_corrected = ifft2c(k_corrected)
            return image_corrected

        else:
            # For 2-channel real representation
            # We need measured_kspace to also be 2-channel real or convertable
            if torch.is_complex(measured_kspace):
                # Unlikely if image is real-stacked, but handle it
                measured_kspace_complex = measured_kspace
            else:
                measured_kspace_complex = torch.complex(
                    measured_kspace[:, 0::2, ...], measured_kspace[:, 1::2, ...]
                )

            k_guessed_shape = k_guessed_complex.shape
            measured_shape = measured_kspace_complex.shape
            mask_shape = mask.shape
            # RuntimeError, not Exception: a non-broadcastable blend raises
            # RuntimeError, while ``torch.utils.checkpoint`` signals the end of a
            # recompute by raising ``_StopRecomputationError(Exception)`` THROUGH
            # whatever user code is running. Catching that logged a [DC LAYER CRASH]
            # per unrolled block per step on every checkpointed arm -- ten fabricated
            # errors in the 2026-09-17 diff_varnet run, which completed normally.
            try:
                k_corrected_complex = (
                    k_guessed_complex * (1.0 - mask) + measured_kspace_complex * mask
                )
            except RuntimeError:
                logger.error(
                    "[DC LAYER CRASH] k_guessed_complex: %s, measured_kspace_complex: %s, mask: %s",
                    k_guessed_shape,
                    measured_shape,
                    mask_shape,
                    exc_info=True,
                )
                raise

            # iFFT
            image_corrected_complex = ifft2c(k_corrected_complex)

            # Stack back to Real/Imag keeping the interleaved channel layout
            return torch.stack(
                [image_corrected_complex.real, image_corrected_complex.imag], dim=2
            ).flatten(1, 2)
