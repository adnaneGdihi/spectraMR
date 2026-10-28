"""Spectral Band-Split Loss for Frequency-Selective Unpaired Translation.

Decomposes k-space into low-frequency (anatomy) and high-frequency (texture)
bands. Enforces hard data consistency on the low-frequency center while
applying Wasserstein adversarial matching only on the high-frequency periphery.

This prevents global structural hallucination by locking the macro-topology
from the ULF input, while focusing 100% of the adversarial capacity on
synthesizing realistic HF texture detail.

Mathematical Foundation:
    L = λ₁ ||M_low ⊙ (F(G(x_ulf)) - F(x_ulf))||₁
      + λ₂ W₁(M_high ⊙ F(G(x_ulf)), M_high ⊙ F(y_hf))

References:
    - Jiang, L. et al. (2021). "Focal Frequency Loss for Image
      Reconstruction and Synthesis." ICCV.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.registry import register_loss

logger = logging.getLogger(__name__)


def _build_frequency_mask(
    shape: tuple[int, int],
    center_fraction: float,
    device: torch.device,
    band: str = "low",
) -> torch.Tensor:
    """Build a circular frequency mask.

    Args:
        shape: (H, W) spatial dimensions.
        center_fraction: Fraction of k-space radius for the cutoff.
        device: Target device.
        band: 'low' for center, 'high' for periphery.

    Returns:
        Binary mask [1, 1, H, W].
    """
    H, W = shape
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    cutoff_r = center_fraction * max_r

    y = torch.arange(H, device=device).float() - cy
    x = torch.arange(W, device=device).float() - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)

    if band not in ("low", "high"):
        raise ValueError(f"band must be 'low' or 'high', got {band!r}")
    mask = (r <= cutoff_r).float() if band == "low" else (r > cutoff_r).float()

    return mask.unsqueeze(0).unsqueeze(0)


@register_loss(
    name="spectral_band_split",
    aliases=["band_split"],
    domain="kspace",
    compatible_with=["image"],
)
class SpectralBandSplitLoss(nn.Module):
    r"""Frequency-selective loss splitting k-space into anatomy and texture bands.

    Low-frequency band (anatomy): Hard L1 data consistency against ULF input.
    High-frequency band (texture): Adversarial or L1 matching against HF target.

    Args:
        center_fraction: Fraction of k-space radius for low/high boundary.
        lambda_low: Weight for low-frequency consistency loss.
        lambda_high: Weight for high-frequency synthesis loss.
        high_loss_type: Loss type for high-frequency band ('l1' or 'l2').
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{SpectralBandSplit} = \lambda_{low} \| M_{low} (\hat{k} - k) \|_1 + \lambda_{high} \| M_{high} (\hat{k} - k) \|_p"""

    def __init__(
        self,
        center_fraction: float = 0.15,
        lambda_low: float = 10.0,
        lambda_high: float = 1.0,
        high_loss_type: str = "l1",
    ) -> None:
        super().__init__()
        if high_loss_type not in ("l1", "l2"):
            raise ValueError(f"high_loss_type must be 'l1' or 'l2', got {high_loss_type!r}")
        if not 0.0 < center_fraction < 1.0:
            raise ValueError(f"center_fraction must lie in (0, 1), got {center_fraction!r}")
        self.center_fraction = center_fraction
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.high_loss_type = high_loss_type
        self._mask_cache: dict[tuple[tuple[int, int], str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_masks(
        self,
        shape: tuple[int, int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get or build cached frequency masks."""
        key = (shape, str(device))
        if key not in self._mask_cache:
            m_low = _build_frequency_mask(shape, self.center_fraction, device, "low")
            m_high = _build_frequency_mask(shape, self.center_fraction, device, "high")
            self._mask_cache[key] = (m_low, m_high)
        return self._mask_cache[key]

    def forward(
        self,
        prediction: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute spectral band-split loss.

        Args:
            prediction: Generator output G(x_ulf) [B, C, H, W].
            source: ULF input x_ulf [B, C, H, W] (for low-freq consistency).
            target: HF target y_hf [B, C, H, W] (for high-freq matching).
                None for unpaired mode (high-freq loss disabled).

        Returns:
            Dict with 'loss_low', 'loss_high', and 'spectral_total'.

        forward method for SpectralBandSplitLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            source (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        H, W = prediction.shape[-2], prediction.shape[-1]
        device = prediction.device
        m_low, m_high = self._get_masks((H, W), device)

        # Forward FFT. Pass tensors straight to fft2c, which routes through
        # ``_to_complex``: an even-channel real/imag layout ([B, 2, H, W]) is
        # interleaved into [B, 1, H, W] complex, a 1-channel real input gets a
        # zero phase, and an already-complex tensor passes through unchanged.
        # Pre-casting with ``.to(torch.complex64)`` here would defeat that
        # interleaving for the common 2-channel real/imag layout -- each channel
        # would be FFT'd as a separate real image instead of one complex image.
        k_pred = fft2c(prediction)
        k_src = fft2c(source)

        # Low-frequency consistency: G(x) must preserve ULF anatomy
        low_residual = m_low * (k_pred - k_src)
        loss_low = low_residual.abs().mean()

        # High-frequency synthesis
        if target is not None:
            k_tgt = fft2c(target)

            high_residual = m_high * (k_pred - k_tgt)
            if self.high_loss_type == "l2":
                loss_high = high_residual.abs().pow(2).mean()
            else:
                loss_high = high_residual.abs().mean()
        else:
            loss_high = loss_low.new_zeros(())

        total = self.lambda_low * loss_low + self.lambda_high * loss_high

        return {
            "loss_low": loss_low,
            "loss_high": loss_high,
            "spectral_total": total,
        }

    def extra_repr(self) -> str:
        return (
            f"center_fraction={self.center_fraction}, "
            f"λ_low={self.lambda_low}, λ_high={self.lambda_high}"
        )


__all__ = ["SpectralBandSplitLoss", "_build_frequency_mask"]
