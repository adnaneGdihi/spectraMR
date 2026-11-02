r"""Scout-acquisition simulator (SCAS / idea 5).

Produces a low-resolution *scout* k-space view :math:`y_s` from a
full-resolution k-space. The scout is what the SCAS hypernet sees in
order to emit a per-sample sampling mask. Implements the
``scout_acquisition`` transform from
``TODO/integration_plan_ulf_cheap_fast_mri.md`` §5.2.

Mathematical formulation
------------------------
The scout is a centred, rectangular truncation of the full k-space:

.. math::

    y_s(k_x, k_y) = \mathbb{1}_{|k_x|<S_x/2,\,|k_y|<S_y/2} \cdot y(k_x, k_y).

Outside the central :math:`S_x \times S_y` window the scout is zero.
This is the Cartesian-rasterised approximation of a quick low-res
scout acquisition.

The transform supports both image-domain and k-space-domain inputs;
in the image-domain case it routes the input through ``fft2c → crop →
ifft2c`` to obtain the truly low-resolution image at the original
grid size.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torchio as tio

from spectramr.data.transforms.registry import register_transform
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

__all__ = ["ScoutAcquisitionTransform"]


@register_transform("scout_acquisition", produces=("scout",))
class ScoutAcquisitionTransform(tio.Transform):
    r"""Crop k-space to the central scout window, optionally returning to image domain.

    Args:
        scout_resolution: ``(S_x, S_y)`` of the scout window. Both
            must be ≤ the input spatial dims.
        domain: ``"kspace"`` or ``"image"`` — interpretation of the
            input. ``"image"`` runs ``fft2c`` first.
        keys: Subject keys to apply the transform to.

    Output is in the same domain as the input (``y_s`` lives in
    k-space when ``domain='kspace'``; the low-res image lives at
    the original grid size when ``domain='image'``, with high
    frequencies zeroed).
    """

    def __init__(
        self,
        scout_resolution: tuple[int, int] = (32, 32),
        domain: str = "kspace",
        keys: Sequence[str] = ("input",),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if domain not in ("kspace", "image"):
            raise ValueError(f"domain must be 'kspace' or 'image'; got {domain!r}")
        if any(s <= 0 for s in scout_resolution):
            raise ValueError(f"scout_resolution components must be > 0; got {scout_resolution}")
        self.scout_resolution = tuple(scout_resolution)
        self.domain = domain
        self.keys = tuple(keys)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        for key in self.keys:
            if key not in subject:
                continue
            image = subject[key]
            data = image.data  # (C, *spatial)
            if data.dim() < 3:
                continue
            cropped = self._scout_crop(data)
            # Emit the scout as a SEPARATE image rather than overwriting the
            # input in-place. The SCAS hypernet needs the scout view AND the
            # full-resolution input simultaneously (the strategy reads
            # ``batch['scout']`` to emit the sampling mask while the model
            # reconstructs from the input). Overwriting ``input`` destroyed the
            # full-res signal and produced no ``scout`` key, so the SCAS
            # density penalty was silently dead every batch (audit 2026-06).
            # Deepcopy the source image (preserving torchio metadata/affine) and
            # swap its data via the proven ``set_data`` path.
            scout_img = copy.deepcopy(image)
            scout_img.set_data(cropped)
            scout_name = "scout" if len(self.keys) == 1 else f"scout_{key}"
            subject.add_image(scout_img, scout_name)
        return subject

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _scout_crop(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the central-window crop on the in-plane axes.

        A TorchIO tensor is ``(C, W, H, D)``: the plane is axes 1 and 2 and the
        slice axis is last, so it is moved out of the way before cropping. A 3-D
        ``(C, H, W)`` tensor already has the plane last.
        """
        if data.dim() == 4:
            return self._crop_last_two(data.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self._crop_last_two(data)

    def _crop_last_two(self, data: torch.Tensor) -> torch.Tensor:
        """Crop the central window on the last two axes."""
        H, W = data.shape[-2], data.shape[-1]
        sx, sy = self.scout_resolution
        if sx > H or sy > W:
            raise ValueError(
                f"scout_resolution {self.scout_resolution} larger than "
                f"input spatial dims ({H}, {W})"
            )

        if self.domain == "image":
            # Move to k-space, zero outside the central window, return to image.
            kspace = fft2c(_promote_complex(data))
            cropped_k = self._zero_outside_centre(kspace)
            return ifft2c(cropped_k)

        # k-space domain: just zero outside the centre.
        return self._zero_outside_centre(data)

    def _zero_outside_centre(self, kspace: torch.Tensor) -> torch.Tensor:
        """Return a copy with samples outside the central window set to zero."""
        H, W = kspace.shape[-2], kspace.shape[-1]
        sx, sy = self.scout_resolution
        out = torch.zeros_like(kspace)
        x0 = (H - sx) // 2
        y0 = (W - sy) // 2
        out[..., x0 : x0 + sx, y0 : y0 + sy] = kspace[..., x0 : x0 + sx, y0 : y0 + sy]
        return out


def _promote_complex(x: torch.Tensor) -> torch.Tensor:
    """Promote a real tensor to complex for FFT (no-op when already complex)."""
    if torch.is_complex(x):
        return x
    return torch.complex(x, torch.zeros_like(x))
