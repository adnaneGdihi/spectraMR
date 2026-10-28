"""Rician Consistency Loss for MRI Reconstruction.

Models the Rician distribution noise bias:
    y = |x + n|   where n ~ CN(0, σ²) (complex Gaussian).

The Rician distribution has a positive bias E[y] > |x| that grows at low SNR.
This loss computes the noise-unbiased target and forces the prediction toward it,
which:
  1. Prevents the model from hallucinating a grey bias floor in flat backgrounds.
  2. Rewards outputs that correctly predict zero (not σ√(π/2)) in air regions.

Unbiased Rician estimator (Gudbjartsson & Patz, 1995):
    x̂ = sqrt(max(y² - 2σ², 0))

References:
    Gudbjartsson, H., & Patz, S. (1995). The Rician distribution of noisy MRI data. MRM.
    Aja-Fernandez, S., et al. (2008). Noise estimation in single- and
    multiple-coil MRI. MRM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.physics_losses import DifferentiableFourierBridge
from spectramr.models.losses.registry import register_loss


def _estimate_sigma_mad(image: torch.Tensor, background_quantile: float = 0.1) -> torch.Tensor:
    """Estimate noise standard deviation using Median Absolute Deviation (MAD).

    MAD estimator: σ̂ = median(|x - median(x)|) / 0.6745
    Applied only to background voxels (low-magnitude region) to avoid
    signal contamination.

    Args:
        image: Magnitude image [B, 1, H, W] (real, non-negative).
        background_quantile: Pixels below this percentile of each image
            are considered background. Default 0.10 (10th percentile).

    Returns:
        Per-sample noise sigma [B, 1, 1, 1].
    """
    B = image.shape[0]
    flat = image.reshape(B, -1)  # [B, N]

    # Threshold: pixels at or below bottom `background_quantile`
    threshold = torch.quantile(flat, background_quantile, dim=1, keepdim=True)  # [B, 1]
    bg_mask = flat <= threshold  # [B, N] boolean

    # Background-only statistics without boolean indexing: non-background entries become NaN
    # and nanmedian skips them. A row with < 4 background voxels falls back to the whole image.
    sparse = bg_mask.sum(dim=1, keepdim=True) < 4
    src = torch.where(bg_mask | sparse, flat, torch.full_like(flat, float("nan")))
    med = src.nanmedian(dim=1, keepdim=True).values
    mad = (src - med).abs().nanmedian(dim=1).values
    sigma = (mad / 0.6745).view(B, 1, 1, 1)
    return sigma.clamp(min=1e-8)


def _rician_unbiased(magnitude: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Compute the Rician-unbiased estimate.

    x̂ = sqrt(max(y² - 2σ², 0))

    This is the method-of-moments estimator for the true signal amplitude
    from a Rician-distributed observation.

    Args:
        magnitude: Observed magnitude image [B, C, H, W] (real ≥ 0).
        sigma: Per-sample noise std [B, 1, 1, 1], or a fixed Python float.

    Returns:
        Unbiased estimate, same shape as magnitude.
    """
    variance_correction = 2.0 * sigma**2  # 2 sigma^2: two degrees of freedom
    unbiased_sq = (magnitude.pow(2) - variance_correction).clamp(min=0.0)
    return torch.sqrt(unbiased_sq + 1e-12)  # sqrt-safe eps


@register_loss(
    name="rician_consistency",
    aliases=["RicianConsistencyLoss", "rician_denoising"],
)
class RicianConsistencyLoss(nn.Module):
    """Rician-noise-aware consistency loss for MRI magnitude reconstruction.

    Encourages model predictions to match the Rician-unbiased target rather
    than the raw noisy magnitude. This removes the noise floor bias present
    in background regions (grey fog artifact) without requiring denoising as
    a pre-processing step.

    Algorithm:
        1. Convert inputs to image-domain magnitude via DifferentiableFourierBridge
           (if use_fourier_bridge=True, i.e. inputs are k-space).
        2. Estimate per-sample noise σ from background voxels using MAD.
        3. Compute unbiased target: T = sqrt(max(target² - 2σ², 0)).
        4. Loss = MAE(prediction, T) = mean(|pred - T|).

    Args:
        use_fourier_bridge: If True, wraps with DifferentiableFourierBridge to
            accept k-space inputs (iFFT to image, then compute loss on magnitudes).
            Set False if inputs are already image-domain magnitudes.
        sigma: Fixed noise σ. If None (default), estimated per-batch via MAD.
        background_quantile: Lower percentile of image used as background for
            MAD sigma estimation. Default: 0.10.
        bias_correction: If True (default), apply Rician unbiased estimator to
            target before computing MAE. If False, falls back to plain MAE.
    """

    def __init__(
        self,
        use_fourier_bridge: bool = True,
        sigma: float | None = None,
        background_quantile: float = 0.10,
        bias_correction: bool = True,
    ) -> None:
        """__init__.

        Args:
            use_fourier_bridge (bool): Description.
            sigma (float | None): Description.
            background_quantile (float): Description.
            bias_correction (bool): Description.
        """
        super().__init__()
        self.use_fourier_bridge = use_fourier_bridge
        self.fixed_sigma = sigma
        self.background_quantile = background_quantile
        self.bias_correction = bias_correction

        self._bridge: DifferentiableFourierBridge | None = None
        if use_fourier_bridge:
            self._bridge = DifferentiableFourierBridge(
                spatial_loss_fn=None,  # We call it manually below
                return_complex=False,
            )

    def _to_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Convert tensor to real magnitude.

        Handles:
        - Already-real 1-channel or multi-channel: abs()
        - 2-channel stacked real/imag: sqrt(re² + im²)
        - Complex dtype: abs()
        """
        if torch.is_complex(x):
            return x.abs()
        if x.shape[1] == 2:
            return torch.sqrt(x[:, 0:1].pow(2) + x[:, 1:2].pow(2) + 1e-12)
        return x.abs()

    def _kspace_to_image_magnitude(self, kspace: torch.Tensor) -> torch.Tensor:
        """Project k-space tensor to image-domain magnitude.

        Uses the bridge's internal iFFT for gradient continuity, then takes magnitude.
        """
        if self._bridge is None:
            raise RuntimeError("k-space input requires use_fourier_bridge=True.")
        img = self._bridge._kspace_to_image(kspace)
        return self._to_magnitude(img)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Rician consistency loss.

        Args:
            pred: Predicted tensor — k-space [B, 2, H, W] if use_fourier_bridge=True,
                  or image-domain magnitude [B, C, H, W] if use_fourier_bridge=False.
            target: Ground truth tensor (same domain as pred).

        Returns:
            Scalar loss value.

        forward method for RicianConsistencyLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.use_fourier_bridge:
            # Project both to image-domain magnitude (gradients flow through iFFT)
            pred_mag = self._kspace_to_image_magnitude(pred)
            with torch.no_grad():
                target_mag = self._kspace_to_image_magnitude(target)
        else:
            pred_mag = self._to_magnitude(pred)
            with torch.no_grad():
                target_mag = self._to_magnitude(target)

        if self.bias_correction:
            # A fixed sigma stays a Python float so a constant costs no per-step
            # host-to-device copy.
            sigma: torch.Tensor | float
            if self.fixed_sigma is not None:
                sigma = float(self.fixed_sigma)
            else:
                with torch.no_grad():
                    sigma = _estimate_sigma_mad(
                        target_mag, background_quantile=self.background_quantile
                    )

            # Compute unbiased Rician target: sqrt(max(y² - 2σ², 0))
            with torch.no_grad():
                unbiased_target = _rician_unbiased(target_mag, sigma)
        else:
            unbiased_target = target_mag

        # L1 between prediction and unbiased target
        return torch.mean(torch.abs(pred_mag - unbiased_target))
