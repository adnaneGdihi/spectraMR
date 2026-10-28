r"""Slice-profile-aware low-resolution simulation for through-plane SR.

Implements §6 of the multi-contrast brainstorm — replaces the trivial
nearest / trilinear downsampling commonly used to synthesise low-
resolution training pairs with a *physically realistic* slice profile.
The result is a forward model that matches what an actual scanner would
produce when acquiring a thicker slice from the same anatomy.

Three profile shapes are supported:

- ``sinc_gauss`` — apodised sinc, the canonical RF-pulse approximation
  for a non-selective slice. Width controlled by FWHM in mm.
- ``gauss`` — pure Gaussian, useful as an idealised baseline.
- ``shinnar_le_roux`` — Shinnar-Le Roux pulse profile [SLR], pre-tabulated
  from a sample SLR-designed pulse (90° flip, BW × T = 4). Closer to
  real clinical acquisitions than the sinc/Gauss ideals.

Implementation
--------------
The transform operates per-volume on the through-plane (z) axis only:

1. Convolve the high-res volume with the chosen 1-D profile along z.
2. Decimate by the slice-thickness factor.
3. Optionally up-sample back to the original grid (for SR pair training)
   by nearest-neighbour replication — leaving the burden of recovery on
   the network rather than the transform.

The transform is registered with the standard ``tio.Transform`` API so
it composes with every other TorchIO transform in
``src/data/transforms/``.

References
----------
Shinnar M, Eleff S, Subramanian H, Leigh JS. The synthesis of pulse
sequences yielding arbitrary magnetisation vectors. Magn Reson Med 1989;
12: 74-80.

§6 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio

from spectramr.data.transforms.registry import register_transform
from spectramr.infrastructure.physics.degradations.blocks.slice_physics import (
    slice_profile_kernel,
)

__all__ = ["SliceProfileTransform"]


# Profile names this transform accepts -> slice_physics kernel kinds.
_PROFILE_FNS = {
    "sinc_gauss": "sinc_gauss",
    "gauss": "gauss",
    "shinnar_le_roux": "slr",
}


def _convolve_along_axis(
    data: torch.Tensor,
    kernel: np.ndarray,
    axis: int,
) -> torch.Tensor:
    """Depthwise 1-D conv of the named axis. ``data`` is (C, X, Y, Z) typed by TorchIO."""
    if axis < 0 or axis >= data.dim():
        raise ValueError(f"axis {axis} out of bounds for tensor with {data.dim()} dims")
    # Move the convolution axis to position -1, then reshape so we get
    # a (B*C', 1, L) tensor we can run a single F.conv1d over.
    perm = list(range(data.dim()))
    perm[-1], perm[axis] = perm[axis], perm[-1]
    moved = data.permute(*perm).contiguous()
    orig_shape = moved.shape
    L = orig_shape[-1]
    flat = moved.view(-1, 1, L)
    weight = torch.from_numpy(kernel.astype("float32")).view(1, 1, -1).to(flat.device)
    pad = weight.shape[-1] // 2
    convolved = F.conv1d(flat.float(), weight, padding=pad)[..., :L]
    out = convolved.view(*orig_shape).permute(*perm)
    return out.to(data.dtype)


@register_transform("slice_profile", modifies=("input", "target"))
class SliceProfileTransform(tio.Transform):
    r"""Apply a physically realistic slice-profile blur + decimation.

    Args:
        profile: One of ``"sinc_gauss"``, ``"gauss"``, ``"shinnar_le_roux"``.
        slice_thickness_mm: Target slice thickness in millimetres.
        kernel_length: Number of taps in the synthesised 1-D profile.
            Ignored for ``"shinnar_le_roux"`` (fixed 33 taps).
        decimate: If True, decimate along the z axis after convolution and
            then nearest-neighbour upsample back to the original z resolution
            so the output volume has the same shape as the input. This is
            the correct shape for SR training pairs. If False, only the
            blur is applied (useful as a soft regulariser).
        z_axis: Axis index of the through-plane direction in the
            ``(C, X, Y, Z)`` TorchIO tensor layout. Default 3.
        keys: Iterable of subject keys to apply the transform to. Default
            ``("input", "target")``.
    """

    def __init__(
        self,
        profile: str = "sinc_gauss",
        slice_thickness_mm: float = 4.0,
        kernel_length: int = 21,
        decimate: bool = True,
        z_axis: int = 3,
        keys: Sequence[str] = ("input", "target"),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if profile not in _PROFILE_FNS:
            raise ValueError(f"profile must be one of {list(_PROFILE_FNS)}, got {profile!r}")
        if slice_thickness_mm <= 0:
            raise ValueError(f"slice_thickness_mm must be > 0, got {slice_thickness_mm}")
        if kernel_length < 3:
            raise ValueError(f"kernel_length must be >= 3, got {kernel_length}")
        if profile == "shinnar_le_roux":
            kernel_length = 33  # fixed
        self.profile = profile
        self.slice_thickness_mm = slice_thickness_mm
        self.kernel_length = kernel_length
        self.decimate = decimate
        self.z_axis = z_axis
        self.keys = tuple(keys)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for key in self.keys:
            if key not in subject:
                continue
            image = subject[key]
            data = image.data  # (C, X, Y, Z)
            # FWHM in voxels: thickness / spacing along z. Default to 1 mm if
            # spacing metadata is missing.
            spacing = image.spacing if hasattr(image, "spacing") else (1.0, 1.0, 1.0)
            z_spacing = float(spacing[self.z_axis - 1]) if self.z_axis >= 1 else 1.0
            fwhm_voxels = max(self.slice_thickness_mm / max(z_spacing, 1e-3), 1.0)

            kernel = slice_profile_kernel(
                _PROFILE_FNS[self.profile], fwhm_voxels, self.kernel_length
            )
            blurred = _convolve_along_axis(data, kernel, self.z_axis)

            if self.decimate:
                # Decimate by the integer ratio, then nearest-up.
                ratio = max(int(round(fwhm_voxels)), 1)
                if ratio > 1:
                    z_orig = data.shape[self.z_axis]
                    slc = [slice(None)] * data.dim()
                    slc[self.z_axis] = slice(None, None, ratio)
                    decimated = blurred[tuple(slc)]
                    # Nearest-neighbour upsample back to original z grid.
                    rep = (z_orig + decimated.shape[self.z_axis] - 1) // decimated.shape[
                        self.z_axis
                    ]
                    upsampled = decimated.repeat_interleave(rep, dim=self.z_axis)
                    # Truncate any over-shoot.
                    slc_end = [slice(None)] * data.dim()
                    slc_end[self.z_axis] = slice(0, z_orig)
                    blurred = upsampled[tuple(slc_end)].contiguous()

            image.set_data(blurred)
        return subject
