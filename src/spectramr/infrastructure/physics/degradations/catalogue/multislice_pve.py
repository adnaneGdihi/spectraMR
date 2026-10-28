"""through_plane_pve: partial-volume averaging across the slice (or partition) profile.

Multislice: a slice integrates the object over its excitation profile, a rect of width ``thk``
convolved with a Gaussian of std ``sigma = theta * thk / 2`` (slice selection, Haacke et al.,
MRI: Physical Principles and Sequence Design, 1999). Each tap is the profile averaged over one
slice extent, so the spread starts at zero and grows continuously with theta.
3-D: partitions are phase-encoded, so the through-plane PSF is the transform of the kz window
(sinc for the rectangular window of a fully sampled slab); a Hamming-like window tapers it,
``w(f) = 1 - theta * (1 - (0.54 + 0.46 cos(2 pi f)))`` with ``f`` the kz frequency in cycles per
partition (Harris, Proc. IEEE 66(1):51-83, 1978). theta = 0 is the identity in both.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex
from spectramr.infrastructure.physics.degradations.blocks.slice_physics import (
    convolve_along_slices,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_common import (
    SIGMA_PER_THICKNESS,
    profile_taps,
    thickness_over_pitch,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import check_volume
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fftnc, ifftnc

__all__ = ["kz_window", "profile_weights", "through_plane_pve"]


def profile_weights(thickness: float, theta: float) -> Tensor:
    """Normalised slice-profile taps (pitch = 1) for ``sigma = theta * thickness / 2``."""
    taps = profile_taps(thickness, float(theta) * SIGMA_PER_THICKNESS * thickness)
    return taps.to(torch.float32)


def kz_window(n_kz: int, theta: float, device=None, dtype=torch.float32) -> Tensor:
    """The kz weights ``1 - theta (1 - hamming(f))``, DC at index ``n_kz // 2``."""
    f = centred_freqs(n_kz, device, dtype=dtype)
    hamming = 0.54 + 0.46 * torch.cos(2.0 * math.pi * f)
    return 1.0 - float(theta) * (1.0 - hamming)


def through_plane_pve(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
) -> Tensor:
    """Convolve the stack with the slice profile (multislice) or taper kz (3-D)."""
    check_volume(x, geometry, "through_plane_pve")
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    if geometry.mode is AcquisitionMode.MULTISLICE_2D:
        return convolve_along_slices(
            x, profile_weights(thickness_over_pitch(geometry, "through_plane_pve"), theta)
        )
    w = kz_window(x.shape[-3], theta, x.device, x.real.dtype)
    return ifftnc(fftnc(x, dims=(-3,)) * w.view(1, 1, -1, 1, 1), dims=(-3,))
