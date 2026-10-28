"""Homomorphic Bias Field Decomposition Transform.

Decomposes a ULF magnitude image into log-domain anatomy and low-frequency
bias field (B₁⁻ receive coil profile) components via Retinex-style separation.

Physics:
    At ULF, the received signal is multiplicatively corrupted by the B₁⁻ coil
    receive sensitivity profile:
        y(r) = ρ(r) · B₁⁻(r)

    where ρ is the true spin density and B₁⁻ is a smooth, low-frequency field.

    In the log domain this becomes additive:
        log(y) = log(ρ) + log(B₁⁻)

    The bias field is separated by low-pass filtering (smooth by physics), and
    the anatomy is the residual. The smoothness_loss (Helmholtz PDE) or the
    MaxwellRegularizer enforces ∇²(log B₁⁻) ≈ 0 during training.

References:
    Hou, Z. et al. (2006). "A homomorphic filtering framework for MRI bias
    field correction." Medical Image Analysis 10(3):443-453.

    Sled, J. G. et al. (1998). "A nonparametric method for automatic correction
    of intensity nonuniformity in MRI data." IEEE TMI 17(1):87-97.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
import torchio as tio

from spectramr.data.transforms.registry import register_transform

logger = logging.getLogger(__name__)


@register_transform(
    "homomorphic_bias_field",
    produces=("log_anatomy", "log_bias"),
)
class HomomorphicBiasFieldDecomposition(tio.transforms.Transform):
    """TorchIO transform that decomposes ULF magnitude into anatomy + bias field.

    Produces two additional subject keys:
        - ``log_anatomy``: High-frequency anatomical content in log domain
        - ``log_bias``: Low-frequency bias field in log domain

    The original image is preserved. The decomposition enables
    ``DisentangledEncoder`` to process anatomy independently of hardware shading.

    Args:
        epsilon: Floor value to avoid log(0). Default: 1e-6.
        smoothness: Gaussian kernel sigma for low-pass extraction of bias field.
            Higher = smoother bias estimate. Default: 8.0.
        kernel_size: Size of the Gaussian kernel (must be odd). Default: 31.
        image_key: Subject key to decompose. Default: "image".
        store_bias_key: Key for the bias field output. Default: "log_bias".
        store_anatomy_key: Key for the anatomy output. Default: "log_anatomy".
        enabled: If False, transform is a no-op. Default: True.
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        smoothness: float = 8.0,
        kernel_size: int = 31,
        image_key: str = "image",
        store_bias_key: str = "log_bias",
        store_anatomy_key: str = "log_anatomy",
        enabled: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if smoothness <= 0:
            raise ValueError(f"smoothness must be positive, got {smoothness}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.epsilon = epsilon
        self.smoothness = smoothness
        self.kernel_size = kernel_size
        self.image_key = image_key
        self.store_bias_key = store_bias_key
        self.store_anatomy_key = store_anatomy_key
        self.enabled = enabled

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Apply homomorphic decomposition to magnitude image in subject.

        Steps:
            1. log(|x| + ε) → log_image
            2. Low-pass(log_image) → log_bias (smooth B₁⁻ estimate)
            3. log_image - log_bias → log_anatomy (structure)
        """
        if not self.enabled:
            return subject

        if self.image_key not in subject:
            logger.debug(
                "HomomorphicBiasField: key '%s' not in subject, skipping",
                self.image_key,
            )
            return subject

        data = subject[self.image_key].data  # (C, *spatial)
        affine = subject[self.image_key].affine

        # Convert to magnitude if complex
        if torch.is_complex(data):
            magnitude = data.abs()
        else:
            magnitude = data.abs()  # Handle negative values

        # Step 1: Log domain transform
        log_image = torch.log(magnitude + self.epsilon)

        # Step 2: Extract low-frequency bias field via Gaussian smoothing
        log_bias = self._gaussian_smooth(log_image)

        # Step 3: Anatomy = residual (high-frequency content)
        log_anatomy = log_image - log_bias

        # Store in subject as new images
        subject.add_image(
            tio.ScalarImage(tensor=log_bias, affine=affine),
            self.store_bias_key,
        )
        subject.add_image(
            tio.ScalarImage(tensor=log_anatomy, affine=affine),
            self.store_anatomy_key,
        )

        logger.debug(
            "HomomorphicBiasField: decomposed '%s' → '%s' + '%s' (σ=%.1f, shape=%s)",
            self.image_key,
            self.store_anatomy_key,
            self.store_bias_key,
            self.smoothness,
            tuple(data.shape),
        )

        return subject

    def _gaussian_smooth(self, log_image: torch.Tensor) -> torch.Tensor:
        """Apply 2D Gaussian smoothing slice-by-slice for 3D volumes.

        The bias field is physically constrained to be spatially smooth
        (electromagnetic fields cannot contain high frequencies in source-free
        regions). A Gaussian low-pass accurately captures this.

        Args:
            log_image: Log-domain image (C, *spatial). Supports 3D and 4D.

        Returns:
            Smoothed log-domain bias estimate, same shape.
        """
        sigma = self.smoothness
        k = self.kernel_size

        # Build 2D Gaussian kernel
        coords = torch.arange(k, dtype=log_image.dtype, device=log_image.device) - k // 2
        gauss_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
        gauss_2d = gauss_1d.outer(gauss_1d)
        gauss_2d = gauss_2d / gauss_2d.sum()

        C = log_image.shape[0]
        # Depthwise convolution kernel: (C, 1, k, k)
        kernel = gauss_2d.unsqueeze(0).unsqueeze(0).expand(C, 1, k, k)

        pad = k // 2

        if log_image.ndim == 4:
            # TorchIO 3D volume: (C, W, H, D) — smooth each depth slice
            C, W, H, D = log_image.shape
            # Process as batch of depth slices: (D, C, W, H)
            slices = log_image.permute(3, 0, 1, 2)  # (D, C, W, H)
            smoothed = F.conv2d(slices, kernel, padding=pad, groups=C)
            result = smoothed.permute(1, 2, 3, 0)  # (C, W, H, D)
            return result

        if log_image.ndim == 3:
            # 2D image: (C, H, W) — straightforward
            data_4d = log_image.unsqueeze(0)  # (1, C, H, W)
            smoothed = F.conv2d(data_4d, kernel, padding=pad, groups=C)
            return smoothed.squeeze(0)

        # Fallback: return as-is for unexpected shapes
        logger.warning(
            "HomomorphicBiasField: unexpected ndim=%d, skipping smoothing",
            log_image.ndim,
        )
        return log_image


def reconstruct_from_log_components(
    log_anatomy: torch.Tensor,
    log_bias: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct magnitude image from log-domain components.

    Inverse of homomorphic decomposition:
        |x_reconstructed| = exp(log_anatomy + log_bias)

    Args:
        log_anatomy: Log-domain anatomy (high-frequency).
        log_bias: Log-domain bias field (low-frequency).

    Returns:
        Reconstructed magnitude image.
    """
    return torch.exp(log_anatomy + log_bias)


@register_transform(
    "homomorphic_bias_correction", produces=("input_log_bias",), requires=("input",)
)
class HomomorphicBiasFieldCorrection(tio.transforms.Transform):
    r"""Divide the receive-bias estimate out of one magnitude image, in place.

    The bias is the normalised Gaussian convolution of ``log|y|`` over the
    foreground, ``b = G*(m log|y|) / G*m`` with ``m = |y| > f * max|y|``, centred
    to zero mean on the foreground so the global intensity is kept. The image
    becomes ``y * exp(-b)`` on the foreground and is left unchanged elsewhere.
    Weighting by ``m`` keeps the near-zero background from dragging the
    estimate down at the brain edge, which plain low-pass filtering of
    ``log(|y| + eps)`` does. The applied ``b`` is stored under ``store_bias_key``
    so a snapshot can show that the correction ran and what it removed.

    Args:
        smoothness: Gaussian sigma of the bias estimate, in pixels.
        kernel_size: Odd Gaussian kernel size.
        foreground_fraction: Foreground threshold as a fraction of ``max|y|``.
        image_key: Subject key corrected; ``input`` corrects the ULF side only.
        store_bias_key: Subject key the applied log-bias map is stored under.
    """

    def __init__(
        self,
        smoothness: float = 8.0,
        kernel_size: int = 31,
        foreground_fraction: float = 0.05,
        image_key: str = "input",
        store_bias_key: str = "input_log_bias",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 < foreground_fraction < 1.0:
            raise ValueError(f"foreground_fraction must lie in (0, 1), got {foreground_fraction}")
        self._smoother = HomomorphicBiasFieldDecomposition(
            smoothness=smoothness, kernel_size=kernel_size
        )
        self.foreground_fraction = float(foreground_fraction)
        self.image_key = image_key
        self.store_bias_key = store_bias_key

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.image_key not in subject:
            raise KeyError(
                f"homomorphic_bias_correction: subject has no {self.image_key!r} image "
                f"(keys: {sorted(subject.keys())})"
            )
        image = subject[self.image_key]
        data = image.data
        if torch.is_complex(data) or data.ndim not in (3, 4):
            raise ValueError(
                "homomorphic_bias_correction needs a real magnitude image (C, H, W) or "
                f"(C, W, H, D); got dtype={data.dtype}, shape={tuple(data.shape)}"
            )
        magnitude = data.abs().float()
        peak = magnitude.flatten(1).amax(dim=1).view(-1, *([1] * (data.ndim - 1)))
        fg = (magnitude > self.foreground_fraction * peak).float()
        smooth = self._smoother._gaussian_smooth
        log_bias = smooth(fg * torch.log(magnitude.clamp_min(1e-12))) / smooth(fg).clamp_min(1e-6)
        centre = (log_bias * fg).flatten(1).sum(1) / fg.flatten(1).sum(1).clamp_min(1.0)
        log_bias = (log_bias - centre.view_as(peak)) * fg
        image.set_data((data * torch.exp(-log_bias)).to(data.dtype))
        subject.add_image(tio.ScalarImage(tensor=log_bias, affine=image.affine), self.store_bias_key)
        return subject


__all__ = [
    "HomomorphicBiasFieldCorrection",
    "HomomorphicBiasFieldDecomposition",
    "reconstruct_from_log_components",
]
