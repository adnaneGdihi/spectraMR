"""Fourier-pair adapters: image ↔ k-space bridging.

Promotes the implicit FFT/iFFT auto-bridge already inside ``LossBuilder``
to a declarative, audited adapter. When ``losses.output_domain == 'image'``
but the YAML lists kspace_losses, today the LossBuilder silently inserts
an FFT. Declaring it as an adapter brings audit visibility to the bridge.

Phase 3: register and surface in the Spec Card. Phase 4 will rewire
LossBuilder to honor the explicit chain when present.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.data.adapters.registry import register_adapter
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


@register_adapter(
    name="fft_image_to_kspace",
    bridges_from={"domain": "image"},
    bridges_to={"domain": "kspace"},
    # ``pre_model`` is the natural insertion point when an image-domain
    # dataset feeds a k-space-input model (e.g. dummy synthetic data
    # → kspace_cold_diffusion). The audit's ``data_model_compatibility``
    # check expects an explicit adapter at that hook to cover the
    # domain bridge per CLAUDE.md #9 (no silent rescue).
    insertion_points=("pre_model", "pre_loss_target"),
    invertible=True,
    side_effects=("amp_autocast_handled_by_fft_ops",),
)
class FFTImageToKspace(nn.Module):
    """Centered orthonormal 2D FFT via the project's ``fft2c``.

    Goes through ``src/infrastructure/physics/fft_ops.py`` per
    CLAUDE.md — never raw ``torch.fft.*``. Real→complex coercion is
    delegated to ``fft2c`` (its SSOT ``_to_complex``), NOT done here.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Coercion is delegated to fft2c's SSOT ``_to_complex``, which
        # de-interleaves a real/imag-interleaved ``[B, 2C, H, W]`` tensor into
        # ``[B, C, H, W]`` complex. A manual ``x.to(complex64)`` here would
        # PREEMPT that, misreading 2-channel real_interleaved data as 2
        # zero-imaginary complex channels — the 2ch→4ch doubling that crashed
        # exp_vf_method_a/c + ib_infonce (cluster VF smoke 2026-06-16).
        return fft2c(x)


@register_adapter(
    name="ifft_kspace_to_image",
    bridges_from={"domain": "kspace"},
    bridges_to={"domain": "complex_image"},
    # ``post_model`` is the natural insertion point when a k-space-output
    # model feeds an image-domain loss. ``pre_model`` covers the inverse
    # case (added 2026-05-12): image-domain models trained on k-space
    # datasets — e.g. ``graph_unet`` / ``vit_kan_generator`` registered
    # as image-domain but pointed at a ``dataset_type: kspace`` corpus.
    # In both cases the iFFT bridges between the two halves cleanly.
    insertion_points=(
        "pre_model",
        "post_model",
        "pre_loss_target",
        "pre_metric",
    ),
    invertible=True,
    side_effects=("amp_autocast_handled_by_fft_ops",),
)
class IFFTKspaceToImage(nn.Module):
    """Centered orthonormal 2D iFFT via the project's ``ifft2c``.

    Output is complex; pair with ``magnitude_from_complex`` if the
    consumer needs a real magnitude image. Real→complex coercion is
    delegated to ``ifft2c`` (its SSOT ``_to_complex``), NOT done here.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Delegate coercion to ifft2c's SSOT ``_to_complex`` (de-interleaves
        # real/imag channels). A manual cast here doubles 2-channel
        # real_interleaved k-space into complex 2ch → 4ch downstream, which
        # crashed the VF cross-attention/distillation/IB arms at the model's
        # first conv (cluster VF smoke 2026-06-16). See FFTImageToKspace.
        return ifft2c(x)


@register_adapter(
    name="kspace_to_real_imag_interleave",
    bridges_from={"domain": "kspace"},
    bridges_to={"domain": "kspace"},
    # Feeds a k-space model that takes 2-channel real-interleaved input (e.g. FNO)
    # from a complex k-space tensor produced by ``fft_image_to_kspace``.
    insertion_points=("pre_model",),
    invertible=True,
    side_effects=(),
)
class KSpaceToRealImagInterleave(nn.Module):
    """Complex ``[B, C, H, W]`` k-space -> real ``[B, 2C, H, W]`` interleaved (Re, Im)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise ValueError(
                "kspace_to_real_imag_interleave expects a complex k-space tensor; "
                f"got real dtype={x.dtype}, shape={tuple(x.shape)}."
            )
        b, c, *spatial = x.shape
        out = x.new_zeros((b, c * 2, *spatial), dtype=torch.float32)
        out[:, 0::2] = x.real
        out[:, 1::2] = x.imag
        return out


__all__ = ["FFTImageToKspace", "IFFTKspaceToImage", "KSpaceToRealImagInterleave"]
