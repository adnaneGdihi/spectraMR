"""TorchIO intensity-augmentation adapters.

TorchIO ships ``RandomGamma``, ``RandomNoise``, ``RandomBlur`` and
``RandomBiasField``, so :mod:`spectramr.data.transforms.augmentation_factory`
uses those directly.  It does **not** ship brightness/contrast jitter, and the
project's own :class:`~spectramr.data.transforms.realistic_degradations.RicianNoise`
and :class:`~spectramr.data.transforms.realistic_degradations.MotionBlur3D` are
``nn.Module``\\s that speak ``[B, C, D, H, W]``, not TorchIO subjects.  This
module supplies the four missing pieces as thin ``tio.Transform`` adapters.

The two MRI-physics adapters **wrap** the existing modules rather than
reimplementing them (non-negotiable 16: prefer wiring over rewriting).  The
degradation maths stays in ``realistic_degradations``; only the layout bridge
and the subject iteration live here.

Layout contract
---------------
TorchIO stores image data as ``[C, W, H, D]`` with no batch axis.  Adapters
that hand tensors to an ``nn.Module`` therefore permute to ``[1, C, D, H, W]``
and back; ``permute(0, 3, 2, 1)`` is its own inverse between the two 4-D
layouts, which is why the same call appears on both sides.

Complex data
------------
None of these adapters is defined on complex tensors — Rician noise and gamma
both assume a magnitude image.  The complex guard lives in the **factory**, not
here, so there is exactly one owner for the decision (non-negotiable 17).
"""

from __future__ import annotations

import logging

import torch
import torchio as tio

logger = logging.getLogger(__name__)

#: TorchIO spatial layout is ``[C, W, H, D]``; the ``nn.Module`` degradations in
#: ``realistic_degradations`` take ``[B, C, D, H, W]``.  This permutation maps
#: ``[C, W, H, D] -> [C, D, H, W]`` and, applied again, maps back.
_TIO_TO_MODULE_PERM = (0, 3, 2, 1)


def _sample_uniform(low: float, high: float, device: torch.device) -> float:
    """Draw one scalar from ``U(low, high)`` on the global RNG.

    Uses ``torch.rand`` rather than ``random`` so the draw is governed by the
    seed ``initialize_accelerator`` sets, keeping augmented runs reproducible.
    """
    return float(torch.rand(1, device=device).item()) * (high - low) + low


class RandomBrightness(tio.IntensityTransform):
    """Multiply intensities by a factor drawn from ``U(*brightness_range)``.

    TorchIO has no brightness transform.  ``RandomGamma`` is not a substitute:
    gamma is a *non-linear* remap that leaves 0 and 1 fixed, whereas brightness
    is a uniform scale, so the two are not interchangeable for arms that ask
    for both.

    Args:
        brightness_range: ``(low, high)`` multiplicative bounds.
        p: Probability of applying the transform to a subject.
    """

    def __init__(self, brightness_range: tuple[float, float], p: float = 1.0, **kwargs):
        super().__init__(p=p, **kwargs)
        self.brightness_range = brightness_range

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Scale every intensity image in ``subject`` by one shared factor."""
        low, high = self.brightness_range
        for image in self.get_images(subject):
            factor = _sample_uniform(low, high, image.data.device)
            image.set_data(image.data * factor)
        return subject


class RandomContrast(tio.IntensityTransform):
    """Scale intensities about their mean by ``U(*contrast_range)``.

    ``(x - mean) * c + mean`` — the mean is preserved, the spread is scaled.
    Computed per image, so a subject whose modalities have different dynamic
    ranges is not skewed by a shared pivot.

    Args:
        contrast_range: ``(low, high)`` multiplicative bounds on the spread.
        p: Probability of applying the transform to a subject.
    """

    def __init__(self, contrast_range: tuple[float, float], p: float = 1.0, **kwargs):
        super().__init__(p=p, **kwargs)
        self.contrast_range = contrast_range

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Scale each intensity image's spread about its own mean."""
        low, high = self.contrast_range
        for image in self.get_images(subject):
            data = image.data
            factor = _sample_uniform(low, high, data.device)
            mean = data.mean()
            image.set_data((data - mean) * factor + mean)
        return subject


class RandomRicianNoise(tio.IntensityTransform):
    """Add Rician noise via the existing ``RicianNoise`` module.

    ``tio.RandomNoise`` adds *Gaussian* noise, which is the wrong model for an
    MRI magnitude image: the magnitude of a complex Gaussian is Rician, and the
    difference is largest exactly where it matters, in low-SNR background.
    ``RicianNoise.forward`` is elementwise, so no layout bridge is needed.

    Args:
        noise_level: Standard deviation of each complex component.
        p: Probability of applying the transform to a subject.
    """

    def __init__(self, noise_level: float, p: float = 1.0, **kwargs):
        super().__init__(p=p, **kwargs)
        self.noise_level = noise_level
        from spectramr.data.transforms.realistic_degradations import RicianNoise

        self.noise = RicianNoise(noise_level=noise_level)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Apply Rician noise to every intensity image."""
        for image in self.get_images(subject):
            image.set_data(self.noise(image.data))
        return subject


class RandomMotionBlur(tio.IntensityTransform):
    """Apply directional motion blur via the existing ``MotionBlur3D`` module.

    ``MotionBlur3D`` convolves with a normalized 3-D kernel of size
    ``blur_kernel_size`` on every spatial axis.  On a volume thinner than the
    kernel — a 2-D slice stored as ``[C, W, H, 1]`` is the common case — the
    convolution's zero padding would dominate the (globally normalized) kernel
    and silently scale intensities down by roughly the kernel extent.  That is a
    wrong image, not a failure, so this adapter **raises** instead of degrading
    (non-negotiable 3).  No arm in ``experiments/inprogress/`` currently sets
    ``enable_motion_blur: true``, so the raise has no corpus impact today; it
    exists so the first arm that does gets told why rather than getting dimmer
    images.

    Args:
        motion_intensity: Falloff of the directional blur kernel.
        blur_kernel_size: Odd kernel extent, in voxels, on every spatial axis.
        p: Probability of applying the transform to a subject.
    """

    def __init__(
        self,
        motion_intensity: float,
        blur_kernel_size: int = 5,
        p: float = 1.0,
        **kwargs,
    ):
        super().__init__(p=p, **kwargs)
        self.motion_intensity = motion_intensity
        self.blur_kernel_size = blur_kernel_size
        from spectramr.data.transforms.realistic_degradations import MotionBlur3D

        self.blur = MotionBlur3D(
            blur_kernel_size=blur_kernel_size,
            motion_intensity=motion_intensity,
        )

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Blur every intensity image, bridging TorchIO layout to ``MotionBlur3D``."""
        for image in self.get_images(subject):
            data = image.data
            spatial = data.shape[1:]
            if min(spatial) < self.blur_kernel_size:
                raise ValueError(
                    "RandomMotionBlur needs every spatial axis to be at least "
                    f"blur_kernel_size={self.blur_kernel_size} voxels, but this "
                    f"image is {tuple(spatial)} (TorchIO [W, H, D]). A kernel "
                    "wider than the volume is dominated by zero padding and "
                    "would silently darken the image. Either raise the patch "
                    "depth, lower blur_kernel_size, or set "
                    "data.augmentation.enable_motion_blur: false."
                )
            # [C, W, H, D] -> [C, D, H, W] -> [1, C, D, H, W]
            tensor_in = data.permute(*_TIO_TO_MODULE_PERM).unsqueeze(0).contiguous()
            blurred = self.blur(tensor_in)
            # [1, C, D, H, W] -> [C, D, H, W] -> [C, W, H, D]
            image.set_data(blurred.squeeze(0).permute(*_TIO_TO_MODULE_PERM))
        return subject


__all__ = [
    "RandomBrightness",
    "RandomContrast",
    "RandomMotionBlur",
    "RandomRicianNoise",
]
