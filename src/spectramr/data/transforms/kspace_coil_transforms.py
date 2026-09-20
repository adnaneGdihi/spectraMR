"""K-Space Coil Transforms for Multi-Coil MRI Data.

Provides transforms to convert multi-coil complex k-space data into formats
compatible with models expecting 2-channel (Real/Imag) input.

Two strategies:
1. ComplexToRealTransform: Flatten coils to channels (2*Coils channels)
2. CoilCombineTransform: RSS/SENSE coil combination to single-coil (2 channels)

Usage:
    # Option A: Coil Combination (recommended for reconstruction)
    transforms = tio.Compose([
        CoilCombineTransform(method="rss"),  # (Coils, H, W, D) complex -> (2, H, W, D) real
        # ... other transforms
    ])

.. math::

    x_{rss} = \\sqrt{\\sum_{c=1}^{N_c} |x_c|^2}

    x_{sense} = \frac{\\sum_{c=1}^{N_c} S_c^* \\cdot x_c}{\\sum_{c=1}^{N_c} |S_c|^2 + \\epsilon}

    # Option B: Flatten coils (preserves coil information)
    transforms = tio.Compose([
        ComplexToRealTransform(),  # (Coils, H, W, D) complex -> (2*Coils, H, W, D) real
        # ... other transforms
    ])
"""

import logging
from typing import Literal

import torch
import torchio as tio

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

logger = logging.getLogger(__name__)


class ComplexToRealTransform(tio.Transform):
    """Convert complex k-space to real by splitting Real/Imag channels.

    Input:  (C, H, W, D) complex tensor
    Output: (2*C, H, W, D) real tensor where channels are [Re(c0), Im(c0), Re(c1), Im(c1), ...]

    This preserves all coil information but doubles the channel count.
    Model must be configured with in_channels = 2 * num_coils.
    """

    # Keys that should be transformed (k-space data)
    # Keys that should be transformed (k-space data)
    # [UPDATED] Added canonical 'input' and 'target'
    KSPACE_KEYS = {
        "kspace",
        "kspace_sub",
        "kspace_full",
        "measured_kspace",
        "input",
        "target",
    }

    def __init__(self, keys: list[str] | None = None, interleave: bool = True):
        """Initialize transform.

        Args:
            keys: Image keys to transform. If None, uses KSPACE_KEYS.
            interleave: If True, channels are [Re(c0), Im(c0), Re(c1), Im(c1), ...].
                        If False, channels are [Re(c0), Re(c1), ..., Im(c0), Im(c1), ...].
        """
        super().__init__()
        self.keys = keys or list(self.KSPACE_KEYS)
        self.interleave = interleave

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Apply complex-to-real conversion."""
        for image_name in subject.get_images_names():
            if image_name not in self.keys:
                continue

            image = subject[image_name]
            data = image.data

            # Skip if already real
            if not torch.is_complex(data):
                logger.debug(f"[ComplexToReal] {image_name} already real, skipping")
                continue

            # Split complex to real/imag
            # Input: (C, H, W, D) complex
            # Output: (2C, H, W, D) real
            C, H, W, D = data.shape

            if self.interleave:
                # Interleave: [Re(c0), Im(c0), Re(c1), Im(c1), ...]
                real_data = torch.zeros(2 * C, H, W, D, dtype=torch.float32, device=data.device)
                real_data[0::2] = data.real
                real_data[1::2] = data.imag
            else:
                # Stacked: [Re(c0), Re(c1), ..., Im(c0), Im(c1), ...]
                real_data = torch.cat([data.real, data.imag], dim=0)

            # BUG FIX: TorchIO dict-style assignment subject[key] = Image() doesn't persist.
            # Use ScalarImage.set_data() instead to properly update tensor in-place.
            image.set_data(real_data)

            logger.debug(
                f"[ComplexToReal] {image_name}: ({C}, {H}, {W}, {D}) complex -> "
                f"({2 * C}, {H}, {W}, {D}) real"
            )

        return subject


class CoilCombineTransform(tio.Transform):
    """Combine multi-coil k-space to single-coil using RSS or SENSE.

    Input:  (Coils, H, W, D) complex tensor
    Output: (2, H, W, D) real tensor [Real, Imag]

    Methods:
    - RSS: Root Sum of Squares (no sensitivity maps needed)
    - RSS per channel: RSS reduced once per real/imaginary channel (Shen 2024)
    - SENSE: Uses sensitivity maps for optimal combination (requires 'sensitivity' key)

    For RSS in k-space domain:
    1. Compute per-coil images: img_c = IFFT(k_c)
    2. RSS combine: img_combined = sqrt(sum(|img_c|^2))
    3. Back to k-space: k_combined = FFT(img_combined)
    4. Split to Real/Imag channels
    """

    # Keys that should be transformed (k-space data)
    # Keys that should be transformed (k-space data)
    # [UPDATED] Added canonical 'input' and 'target'
    KSPACE_KEYS = {
        "kspace",
        "kspace_sub",
        "kspace_full",
        "measured_kspace",
        "input",
        "target",
    }

    def __init__(
        self,
        method: Literal["rss", "rss_image", "rss_per_channel", "sense"] = "rss",
        keys: list[str] | None = None,
        sensitivity_key: str = "sensitivity",
    ):
        """Initialize coil combine transform.

        Args:
            method: Combination method.
                - ``"rss"`` — IFFT → RSS magnitude → FFT-back to k-space
                  (output domain: k-space, ``(2, H, W, D)`` real [R, I]).
                  Note this round-trip strips phase: ``FFT(|x|)`` is
                  Hermitian-symmetric, so a downstream IFFT will
                  produce a centro-symmetric image (the "doubled brain"
                  signature reported in audit-2026-05-14 F5).
                  Use this only when a downstream k-space pipeline is
                  expected to overwrite the phase before visualisation.
                - ``"rss_image"`` — IFFT → RSS magnitude, return as
                  image-domain ``(1, H, W, D)`` real. The canonical
                  mode for image-domain reconstruction networks that
                  want pre-combined input (no phase strip downstream).
                - ``"rss_per_channel"`` — IFFT → RSS over coils applied
                  separately to the real and the imaginary channel → FFT
                  back to k-space (output domain: k-space,
                  ``(2, H, W, D)`` real). Shen 2024's own combination;
                  see :meth:`_rss_per_channel_combine`.
                - ``"sense"`` — IFFT → sensitivity-weighted complex sum
                  → FFT back to k-space (preserves phase).
            keys: Image keys to transform. If None, uses KSPACE_KEYS.
            sensitivity_key: Key for sensitivity maps (used with SENSE method)
        """
        super().__init__()
        self.method = method
        self.keys = keys or list(self.KSPACE_KEYS)
        self.sensitivity_key = sensitivity_key

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Apply coil combination."""
        # Get sensitivity maps if using SENSE
        sensitivity = None
        if self.method == "sense" and self.sensitivity_key in subject:
            sensitivity = subject[self.sensitivity_key].data

        for image_name in subject.get_images_names():
            if image_name not in self.keys:
                continue

            image = subject[image_name]
            data = image.data

            # Ensure complex
            if not torch.is_complex(data):
                # Real input is "already combined" when the channel count
                # matches a domain-canonical layout. Skipping these is the
                # expected outcome — the transform stays mounted so it
                # handles complex multi-coil tensors when present, but
                # passes through already-combined tensors unchanged.
                #   - 1 channel  → real magnitude image (single-coil)
                #   - 2 channels → interleaved real/imag (already combined)
                if data.shape[0] in (1, 2):
                    logger.debug(
                        f"[CoilCombine] {image_name} already combined "
                        f"({data.shape[0]}-channel real), pass-through."
                    )
                    continue
                # F2/E10 (2026-05-21 smoke audit): even-channel real
                # tensors are the canonical real-stacked complex multi-
                # coil layout in this codebase (output of
                # ``ComplexToRealTransform``). The previous loud-fail
                # rejected 20 promoted experiments at runtime even
                # though the layout is unambiguous when the operator
                # explicitly asked for a coil-combination method
                # (sense / rss / rss_image / etc.) — they wouldn't
                # mount this transform unless they meant "treat the
                # input as multi-coil complex". Re-interpret
                # ``[2C, H, W, D]`` as ``[C, H, W, D]`` complex (pair
                # consecutive channels as R, I) and proceed.
                #
                # Multi-contrast / interleaved-coil ambiguity remains
                # for ODD-channel real (the historical concern); that
                # still raises. See
                # TODO/audit/smoke_audit_20260521.md §F2.
                if data.shape[0] % 2 == 0:
                    n_real_channels = data.shape[0]
                    c = n_real_channels // 2
                    data = torch.complex(data[0::2].float(), data[1::2].float())
                    logger.debug(
                        f"[CoilCombine] {image_name}: real-stacked "
                        f"({n_real_channels}-real ch) reinterpreted as "
                        f"{c}-complex-coil — proceeding with "
                        f"method={self.method!r} (F2/E10)."
                    )
                else:
                    raise ValueError(
                        f"[CoilCombine] {image_name} is real with "
                        f"{data.shape[0]} channels — odd channel count "
                        "is genuinely ambiguous (multi-contrast vs "
                        "interleaved coils). Either "
                        "(a) pass complex tensors, "
                        "(b) cast to 1-channel magnitude before this transform, "
                        "or (c) unmount CoilCombineTransform when "
                        "coil_processing_mode != 'multi_coil'."
                    )

            # Apply combination
            if self.method == "rss":
                combined = self._rss_combine(data)
            elif self.method == "rss_image":
                # Image-domain RSS: produces a single-channel real
                # magnitude tensor, NO ``FFT(magnitude)`` round-trip —
                # so downstream IFFT-then-magnitude visualisation does
                # not see Hermitian k-space and the "doubled brain"
                # centro-symmetric signature is avoided.
                rss_mag = self._rss_image_combine(data)
                subject[image_name] = tio.ScalarImage(
                    tensor=rss_mag,
                    affine=image.affine,
                )
                logger.debug(
                    f"[CoilCombine] {image_name}: {data.shape} complex -> "
                    f"{rss_mag.shape} real (method=rss_image, image-domain)"
                )
                continue
            elif self.method == "rss_per_channel":
                combined = self._rss_per_channel_combine(data)
            elif self.method == "sense":
                combined = self._sense_combine(data, sensitivity)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Split complex to Real/Imag: (1, H, W, D) complex -> (2, H, W, D) real
            real_imag = torch.stack([combined.real, combined.imag], dim=0).squeeze(1)

            subject[image_name] = tio.ScalarImage(
                tensor=real_imag,
                affine=image.affine,
            )

            logger.debug(
                f"[CoilCombine] {image_name}: {data.shape} complex -> "
                f"{real_imag.shape} real (method={self.method})"
            )

        return subject

    def _rss_combine(self, kspace: torch.Tensor) -> torch.Tensor:
        """Root Sum of Squares coil combination in k-space.

        Args:
            kspace: (Coils, H, W, D) complex k-space

        Returns:
            (1, H, W, D) complex combined k-space

        .. warning::

            Step 3 (FFT(RSS-magnitude)) is **phase-stripping**: the
            output k-space is Hermitian-symmetric, and any downstream
            IFFT produces a centro-symmetric (mirrored) image — the
            "doubled brain" signature reported in audit-2026-05-14 F5.
            Use :meth:`_rss_image_combine` instead when the downstream
            pipeline visualises via IFFT.
        """
        # 1. IFFT to image domain (per coil) using physics module
        # Swap to put spatial dims last for fft: (C, H, W, D) -> (C, D, H, W)
        kspace_transposed = kspace.permute(0, 3, 1, 2)

        # Apply centered IFFT using physics module for proper centering
        images = ifft2c(kspace_transposed)

        # 2. RSS combine: sqrt(sum(|img_c|^2))
        rss_combined = torch.sqrt(torch.sum(torch.abs(images) ** 2, dim=0, keepdim=True))

        # 3. FFT back to k-space using physics module
        combined_kspace = fft2c(rss_combined)

        # Transpose back: (1, D, H, W) -> (1, H, W, D)
        combined_kspace = combined_kspace.permute(0, 2, 3, 1)

        return combined_kspace

    def _rss_image_combine(self, kspace: torch.Tensor) -> torch.Tensor:
        """Root Sum of Squares coil combination, returning image-domain magnitude.

        Steps:

        1. IFFT each coil to image domain.
        2. RSS combine: ``sqrt(sum(|img_c|^2))``.
        3. **No** FFT round-trip — return the real magnitude image.

        This is the "right" RSS for downstream image-domain networks
        and visualisation. The legacy :meth:`_rss_combine` strips phase
        by FFTing the magnitude back to k-space, which causes the
        Hermitian-symmetric / centro-symmetric IFFT signature flagged
        in audit-2026-05-14 F5.

        Args:
            kspace: ``(Coils, H, W, D)`` complex k-space.

        Returns:
            ``(1, H, W, D)`` real magnitude image.
        """
        kspace_transposed = kspace.permute(0, 3, 1, 2)
        images = ifft2c(kspace_transposed)

        rss_combined = torch.sqrt(torch.sum(torch.abs(images) ** 2, dim=0, keepdim=True))
        # ``rss_combined`` is real-valued already; return as image.
        # Transpose back: (1, D, H, W) -> (1, H, W, D)
        return rss_combined.permute(0, 2, 3, 1)

    def _rss_per_channel_combine(self, kspace: torch.Tensor) -> torch.Tensor:
        """RSS over coils applied separately to the real and imaginary channels.

        Reproduces Shen 2024's ``DataTransform_Diffusion``
        (``utils/data_transform.py:62``), which calls ``fastmri.rss`` on a
        ``[Nc, H, W, 2]`` tensor. That function is ``sqrt(sum(x**2, dim=0))`` and
        takes no view of the trailing real/imaginary axis, so it reduces coils
        once per channel and returns two non-negative ones -- ``rss_complex`` is
        the sibling that reduces ``|x_c|``. The published numbers were produced
        from this input, so the difference is reproduced rather than corrected;
        :meth:`_rss_combine` is the conventional magnitude RSS.

        Args:
            kspace: ``(Coils, H, W, D)`` complex k-space.

        Returns:
            ``(1, H, W, D)`` complex k-space of the combined image.
        """
        kspace_transposed = kspace.permute(0, 3, 1, 2)
        images = ifft2c(kspace_transposed)

        real = torch.sqrt(torch.sum(images.real**2, dim=0, keepdim=True))
        imag = torch.sqrt(torch.sum(images.imag**2, dim=0, keepdim=True))
        combined_kspace = fft2c(torch.complex(real, imag))

        # Transpose back: (1, D, H, W) -> (1, H, W, D)
        return combined_kspace.permute(0, 2, 3, 1)

    def _sense_combine(
        self, kspace: torch.Tensor, sensitivity: torch.Tensor | None
    ) -> torch.Tensor:
        """SENSE coil combination using sensitivity maps.

        Args:
            kspace: (Coils, H, W, D) complex k-space
            sensitivity: (Coils, H, W, D) complex sensitivity maps

        Returns:
            (1, H, W, D) complex combined k-space
        """
        if sensitivity is None:
            raise ValueError(
                "[CoilCombine] method='sense' requires sensitivity maps but none were "
                f"found under key {self.sensitivity_key!r} in the subject. "
                "Either supply sensitivity maps or choose method='rss'/'rss_image'. "
                "(CLAUDE.md pitfall #9)"
            )

        # Ensure sensitivity is complex
        if not torch.is_complex(sensitivity):
            if sensitivity.shape[0] == kspace.shape[0] * 2:
                # Assume interleaved Real/Imag
                C = kspace.shape[0]
                sensitivity = torch.complex(sensitivity[:C], sensitivity[C:])
            else:
                # Treat as real-valued magnitude maps
                sensitivity = sensitivity.to(dtype=torch.complex64)

        # 1. IFFT to image domain using physics module
        kspace_transposed = kspace.permute(0, 3, 1, 2)
        images = ifft2c(kspace_transposed)

        # Transpose sensitivity: (C, H, W, D) -> (C, D, H, W)
        sens_transposed = sensitivity.permute(0, 3, 1, 2)

        # 2. SENSE combination: sum(conj(S) * img) / sum(|S|^2)
        numerator = torch.sum(sens_transposed.conj() * images, dim=0, keepdim=True)
        denominator = torch.sum(torch.abs(sens_transposed) ** 2, dim=0, keepdim=True) + 1e-8
        combined_image = numerator / denominator

        # 3. FFT back to k-space using physics module
        combined_kspace = fft2c(combined_image)

        # Transpose back: (1, D, H, W) -> (1, H, W, D)
        combined_kspace = combined_kspace.permute(0, 2, 3, 1)

        return combined_kspace


__all__ = [
    "CoilCombineTransform",
    "ComplexToRealTransform",
]
