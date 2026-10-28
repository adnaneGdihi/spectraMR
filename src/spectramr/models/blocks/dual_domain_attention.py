"""Dual-Domain Attention Mechanisms.

Applies attention in both K-Space (frequency) and Image-Space (spatial) domains
to leverage both global and local feature dependencies.

Feature-domain contract (2026-07-03): the block no longer hard-assumes its
input is k-space. The required ``feature_domain`` kwarg ("kspace" | "image",
derived from ``force_pure_kspace`` by ``FourierBridgeNetwork``) dictates how
the two views are derived at entry and conjugates the output back so that
**output domain == input domain**. ``feature_domain="kspace"`` is numerically
identical to the historical behavior.
"""

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.blocks.attention import ComplexChannelAttention
from spectramr.models.blocks.attention_domains import (
    complex_to_interleaved,
    interleaved_to_complex,
    validate_feature_domain,
)


class ComplexSignalAttention(ComplexChannelAttention):
    """Per-complex-channel gate for the dual-domain branches.

    Signature adapter over :class:`ComplexChannelAttention`, which owns the
    gate: one sigmoid gain per complex channel, shared by Re and Im and squeezed
    from ``|z|``, so a global phase rotation passes through unchanged. The former
    per-real-channel gain on a signed average pool rotated the phase and was
    driven by the DC bin.
    """

    def __init__(self, complex_channels: int, reduction: int = 4):
        """__init__.

        Args:
            complex_channels (int): Complex channel count ``C`` (input is
                ``[B, 2C, H, W]`` interleaved).
            reduction (int): Bottleneck divisor on ``C`` (floor 1).
        """
        super().__init__(2 * complex_channels, reduction_ratio=reduction)


class DualDomainAttention(nn.Module):
    """Applies attention in both K-Space and Image-Space with Interleaved layout.

    Native composition frame is **k-space** (the historical behavior). The
    required ``feature_domain`` kwarg tells the block which domain its input
    tensor actually is; under ``"image"`` the two views are derived by the
    conjugate transforms and the k-frame result is brought back to the image
    frame at exit (output domain == input domain).
    """

    def __init__(self, in_channels: int, reduction: int = 4, *, feature_domain: str):
        """__init__.

        Args:
            in_channels (int): Description.
            reduction (int): Description.
            feature_domain (str): Domain of the input feature map,
                ``"kspace"`` or ``"image"``. Derived from
                ``force_pure_kspace`` by the caller — raises on anything else.
        """
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(f"DualDomainAttention expects even input channels, got {in_channels}")

        self.feature_domain = validate_feature_domain(feature_domain)
        self.complex_channels = in_channels // 2
        self.freq_attn = ComplexSignalAttention(self.complex_channels, reduction)
        self.spat_attn = ComplexSignalAttention(self.complex_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2*C, H, W) - Interleaved, in ``self.feature_domain``
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DualDomainAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor, same domain as the input.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = interleaved_to_complex(x)

        # Derive the two views from the declared input domain. Under "kspace"
        # this is numerically identical to the historical path (h IS k-space);
        # under "image" the roles conjugate.
        if self.feature_domain == "kspace":
            h_k, h_i = h, ifft2c(h)
        else:
            h_k, h_i = fft2c(h), h

        # 1. Frequency Domain Attention (k view)
        x_freq = self.freq_attn(complex_to_interleaved(h_k))

        # 2. Spatial Domain Attention (image view)
        img_attn = self.spat_attn(complex_to_interleaved(h_i))

        # FFT the spatial branch back to the k frame for composition
        i_feat_k = fft2c(interleaved_to_complex(img_attn))
        x_spat = complex_to_interleaved(i_feat_k)

        # 3. Sum features in the native k frame
        out_k = x_freq + x_spat

        # Conjugate back so output domain == input domain
        if self.feature_domain == "kspace":
            return out_k
        return complex_to_interleaved(ifft2c(interleaved_to_complex(out_k)))
