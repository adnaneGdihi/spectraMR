"""Virtual Fiducial ML Losses — Marker-Anchored Reconstruction.

Registered losses that leverage known fiducial markers as absolute ground-truth
probes for transferring corrections to unknown anatomy.

Losses
------
- :class:`ZeroShotDenoisingLoss`           — inner-loop self-supervised denoising
- :class:`MarkerMaskedDiscriminatorLoss`   — spatial masking for GAN discriminator
- :class:`MarkerConditionedVAELoss`        — disentanglement with known prior
- :class:`MarkerPriorLoss`                 — hard boundary constraint (PINN-style)
- :class:`DiffusionVarianceCalibrationLoss` — noise floor from marker ROI
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss

logger = logging.getLogger(__name__)


# ======================================================================
# 1. Zero-Shot Self-Supervised Denoising Loss
# ======================================================================


@register_loss("zero_shot_denoising", aliases=["vf_denoising", "zs_denoising"])
class ZeroShotDenoisingLoss(nn.Module):
    r"""Zero-shot self-supervised denoising loss from marker prior.

    Trains a lightweight denoiser *during the scan* using only the marker
    region.  The loss is computed between the denoised marker and the
    known ideal marker prior:

    .. math::

        \mathcal{L}_{\text{ZS}} = \| f_\theta(M_{\text{noisy}}) -
        M_{\text{prior}} \|_1

    The key insight: once the denoiser learns this scan's noise statistics
    from the marker, those learned weights transfer globally to anatomy.

    Args:
        lambda_marker: Weight for marker reconstruction term.
    """

    def __init__(self, lambda_marker: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.lambda_marker = lambda_marker

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
        marker_prior: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute zero-shot denoising loss.

        Args:
            prediction: Model output ``[B, C, H, W]``.
            target: Clean target ``[B, C, H, W]``.
            marker_mask: Binary mask ``[1, 1, H, W]`` for marker region.
            marker_prior: Ideal marker image ``[1, 1, H, W]``.

        Returns:
            Scalar loss.

        forward method for ZeroShotDenoisingLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_prior (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Global reconstruction loss
        loss_recon = F.l1_loss(prediction, target)

        # Marker-specific anchor loss
        if marker_mask is not None and marker_prior is not None:
            pred_marker = prediction * marker_mask
            target_marker = (marker_prior * marker_mask).expand_as(pred_marker)
            loss_marker = F.l1_loss(pred_marker, target_marker)
            return loss_recon + self.lambda_marker * loss_marker

        return loss_recon


# ======================================================================
# 2. Marker-Masked Discriminator Loss
# ======================================================================


@register_loss(
    "marker_masked_discriminator",
    aliases=["vf_masked_disc", "marker_disc"],
)
class MarkerMaskedDiscriminatorLoss(nn.Module):
    r"""GAN discriminator loss restricted to the marker region.

    The discriminator evaluates *only* the marker region of the generated
    image.  To fool the discriminator the generator must learn the exact
    physics required to make the marker sharp, preventing hallucination
    of fake anatomy.

    .. math::

        \mathcal{L}_D = \mathbb{E}[\log D(M_{\text{real}})]
        + \mathbb{E}[\log(1 - D(M_{\text{fake}}))]

    Args:
        loss_type: 'bce' (vanilla) or 'hinge'.
    """

    def __init__(self, loss_type: str = "hinge", **kwargs: Any) -> None:
        super().__init__()
        if loss_type not in ("bce", "hinge"):
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        discriminator: nn.Module | None = None,
        marker_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute marker-masked discriminator loss.

        Args:
            prediction: Generator output ``[B, C, H, W]``.
            target: Real image ``[B, C, H, W]``.
            discriminator: PatchGAN discriminator module.
            marker_mask: Binary mask ``[1, 1, H, W]``.

        Returns:
            Scalar adversarial loss.

        forward method for MarkerMaskedDiscriminatorLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            discriminator (nn.Module | None): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if discriminator is None or marker_mask is None:
            return F.l1_loss(prediction, target)

        fake_masked = prediction * marker_mask
        real_masked = target * marker_mask

        d_fake: torch.Tensor = discriminator(fake_masked)
        d_real: torch.Tensor = discriminator(real_masked)

        if self.loss_type == "hinge":
            loss_g = -d_fake.mean()
            loss_d = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake.detach()).mean()
        else:
            ones = torch.ones_like(d_real)
            zeros = torch.zeros_like(d_fake)
            loss_d = F.binary_cross_entropy_with_logits(
                d_real, ones
            ) + F.binary_cross_entropy_with_logits(d_fake.detach(), zeros)
            loss_g = F.binary_cross_entropy_with_logits(d_fake, ones)

        return loss_g + loss_d


# ======================================================================
# 3. Marker-Conditioned VAE Loss
# ======================================================================


@register_loss(
    "marker_conditioned_vae",
    aliases=["vf_vae", "marker_vae"],
)
class MarkerConditionedVAELoss(nn.Module):
    r"""VAE loss with marker-forced disentanglement.

    Forces the VAE to isolate a "degradation" latent vector by requiring
    that the "anatomy + contrast" latents perfectly reconstruct the known
    marker:

    .. math::

        \mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot
        D_{KL}(q(z|x) \| p(z)) + \lambda \cdot
        \| \text{Dec}(z_{\text{anat}}, z_{\text{contrast}}) \odot M
        - x_{\text{prior}} \odot M \|_1

    Args:
        beta: KL divergence weight.
        lambda_marker: Marker reconstruction weight.
    """

    def __init__(
        self,
        beta: float = 1.0,
        lambda_marker: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.lambda_marker = lambda_marker

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor | None = None,
        logvar: torch.Tensor | None = None,
        marker_mask: torch.Tensor | None = None,
        marker_prior: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute marker-conditioned VAE loss.

        Args:
            prediction: Reconstructed image ``[B, C, H, W]``.
            target: Ground truth ``[B, C, H, W]``.
            mu: Latent mean ``[B, D]``.
            logvar: Log-variance ``[B, D]``.
            marker_mask: Binary mask ``[1, 1, H, W]``.
            marker_prior: Ideal marker ``[1, 1, H, W]``.

        Returns:
            Scalar composite loss.

        forward method for MarkerConditionedVAELoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            mu (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            logvar (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_prior (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Reconstruction loss
        loss_recon = F.l1_loss(prediction, target)

        # KL divergence
        loss_kl = torch.tensor(0.0, device=prediction.device)
        if mu is not None and logvar is not None:
            loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Marker anchor
        loss_marker = torch.tensor(0.0, device=prediction.device)
        if marker_mask is not None and marker_prior is not None:
            loss_marker = F.l1_loss(prediction * marker_mask, marker_prior * marker_mask)

        return loss_recon + self.beta * loss_kl + self.lambda_marker * loss_marker


# ======================================================================
# 4. Marker Prior Loss (PINN Hard Constraint)
# ======================================================================


@register_loss("marker_prior", aliases=["vf_prior", "pinn_marker"])
class MarkerPriorLoss(nn.Module):
    r"""PINN-style hard boundary constraint loss from marker prior.

    Anchors the reconstruction by penalising deviation at known marker
    voxels:

    .. math::

        \mathcal{L}_{\text{prior}} = \lambda \cdot
        \| M \odot x_{\text{pred}} - M \odot x_{\text{prior}} \|_2^2

    Args:
        lambda_prior: Weight for prior constraint.
    """

    def __init__(self, lambda_prior: float = 10.0, **kwargs: Any) -> None:
        super().__init__()
        self.lambda_prior = lambda_prior

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        marker_mask: torch.Tensor | None = None,
        marker_prior: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute marker prior loss.

        Args:
            prediction: Model output ``[B, C, H, W]``.
            target: Unused (prior is the anchor).
            marker_mask: Binary mask ``[1, 1, H, W]``.
            marker_prior: Known marker image ``[1, 1, H, W]``.

        Returns:
            Scalar loss.

        forward method for MarkerPriorLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_prior (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if marker_mask is None or marker_prior is None:
            return torch.tensor(0.0, device=prediction.device)

        diff = prediction * marker_mask - marker_prior * marker_mask
        # Use abs()**2 for complex-safe squared magnitude (|z|² not z²)
        return self.lambda_prior * torch.abs(diff).square().mean()


# ======================================================================
# 5. Diffusion Variance Calibration Loss
# ======================================================================


@register_loss(
    "diffusion_variance_calibration",
    aliases=["vf_diffusion_cal", "marker_noise_cal"],
)
class DiffusionVarianceCalibrationLoss(nn.Module):
    r"""Calibrate diffusion noise schedule using marker noise floor.

    Instead of guessing the starting noise variance, measures it exactly
    from the uniform regions of the marker.  Penalises the mismatch
    between the estimated noise and the measured marker noise:

    .. math::

        \mathcal{L}_{\text{cal}} = \| \hat{\sigma}^2_{\text{model}}
        - \sigma^2_{\text{marker}} \|_2^2

    This prevents the diffusion model from over-smoothing real tissue.

    Args:
        lambda_cal: Calibration loss weight.
    """

    def __init__(self, lambda_cal: float = 1.0, **kwargs: Any) -> None:
        super().__init__()
        self.lambda_cal = lambda_cal

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        estimated_noise_var: torch.Tensor | None = None,
        marker_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute variance calibration loss.

        Args:
            prediction: Model output ``[B, C, H, W]``.
            target: Ground truth ``[B, C, H, W]``.
            estimated_noise_var: Model's estimated noise variance ``[B]``.
            marker_mask: Binary mask for uniform marker region ``[1, 1, H, W]``.

        Returns:
            Scalar calibration loss.

        forward method for DiffusionVarianceCalibrationLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            estimated_noise_var (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            marker_mask (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Standard reconstruction loss (complex-safe via abs)
        loss_recon = F.l1_loss(torch.abs(prediction), torch.abs(target))

        if estimated_noise_var is None or marker_mask is None:
            return loss_recon

        # Measure actual noise variance from marker region. It is a measurement the
        # estimate is calibrated against, so no gradient reaches the prediction
        # through it: that would reward a larger marker error whenever the
        # estimated variance exceeds the measured one.
        with torch.no_grad():
            residual = (prediction - target) * marker_mask
            n_voxels = marker_mask.sum().clamp(min=1)
            # Complex-safe squared magnitude: |z|² not z²
            actual_var = torch.abs(residual).square().sum() / n_voxels

        # Calibration: model's estimate should match measured
        loss_cal = (estimated_noise_var.mean() - actual_var).square()

        return loss_recon + self.lambda_cal * loss_cal


__all__ = [
    "DiffusionVarianceCalibrationLoss",
    "MarkerConditionedVAELoss",
    "MarkerMaskedDiscriminatorLoss",
    "MarkerPriorLoss",
    "ZeroShotDenoisingLoss",
]
