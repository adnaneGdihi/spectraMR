"""Intensity scaling and retrospective undersampling shared by the baseline adapters.

Each upstream normalises its own training images before its network sees them, and
none of them normalises the same way: CDiffMR divides by the peak magnitude of the fully
sampled image, Shen 2024 by the peak magnitude of the zero-filled one, FDB by the peak
of the volume. The adapters apply their author's rule internally and undo it on the way
out, so every baseline is scored in the units of this repository's target rather than
in three private ones.

All tensors here are ``[B, 2, H, W]`` real / imaginary image-domain stacks, the layout
:class:`~spectramr_baselines.strategy.UpstreamProcessStrategy` hands the adapters.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

#: Floor for a peak magnitude. A zero-padded slice has peak 0, and dividing by it turns
#: the whole batch's loss NaN; flooring leaves such a slice at zero instead.
_PEAK_FLOOR = 1e-12


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """``[B, 2, H, W]`` real / imaginary -> complex ``[B, 1, H, W]``."""
    if x.dim() != 4 or x.shape[1] != 2:
        raise ValueError(
            f"Expected a [B, 2, H, W] real/imaginary stack, got {tuple(x.shape)}. The "
            "baseline adapters are single-coil; set data.domain.target_channels: 2 and a "
            "coil combination that leaves one complex channel."
        )
    return torch.complex(x[:, 0:1], x[:, 1:2])


def require_image_size(x: torch.Tensor, size: int, owner: str, key: str) -> None:
    """Refuse an input whose spatial shape is not the ``size x size`` the adapter was built for.

    Each upstream draws its masks and schedules at its configured size, not at the
    batch's, so a patch of any other size is either a broadcast error deep inside the
    authors' code or, for FDB's frequency radius, a silently different degradation.
    """
    if tuple(x.shape[-2:]) != (size, size):
        raise ValueError(
            f"{owner} was built for {size}x{size} ({key}) but received input of spatial "
            f"shape {tuple(x.shape[-2:])}. Set {key} to the first two entries of "
            "data.sampling.patch_size -- the two must agree."
        )


def to_real_stack(z: torch.Tensor) -> torch.Tensor:
    """Complex ``[B, 1, H, W]`` -> ``[B, 2, H, W]`` real / imaginary."""
    return torch.cat([z.real, z.imag], dim=1)


def peak_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Per-sample peak of the complex magnitude, shaped ``[B, 1, 1, 1]`` to broadcast."""
    peak = to_complex(x).abs().amax(dim=(1, 2, 3), keepdim=True)
    return peak.clamp_min(_PEAK_FLOOR)


def phase_encode_mask(mask_1d: torch.Tensor, height: int) -> torch.Tensor:
    """Tile a 1-D phase-encode mask over the readout axis, ``[W]`` -> ``[1, 1, H, W]``.

    The phase-encode axis is the LAST one for M4Raw, and it is the axis CDiffMR's own
    ``get_ksu_mask`` samples (``mask.permute(0, 2, 1).repeat(1, fe, 1)``).
    """
    if mask_1d.dim() != 1:
        raise ValueError(f"Expected a 1-D phase-encode mask, got shape {tuple(mask_1d.shape)}.")
    return mask_1d.reshape(1, 1, 1, -1).expand(1, 1, height, mask_1d.shape[0])


def zero_filled(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``ifft2c(fft2c(x) * mask)`` on a real / imaginary stack.

    Equal to CDiffMR's ``ifft2(ifftshift(fftshift(fft2(x)) * mask))`` for even image
    sizes: the two differ by a ±1 phase per k-space sample, which commutes with the mask
    and cancels in the inverse.
    """
    if mask.shape[-2:] != x.shape[-2:]:
        raise ValueError(
            f"Mask spatial shape {tuple(mask.shape[-2:])} does not match the image "
            f"{tuple(x.shape[-2:])}."
        )
    kspace = fft2c(to_complex(x)) * mask.to(device=x.device, dtype=x.dtype)
    return to_real_stack(ifft2c(kspace))


__all__ = [
    "peak_magnitude",
    "phase_encode_mask",
    "require_image_size",
    "to_complex",
    "to_real_stack",
    "zero_filled",
]
