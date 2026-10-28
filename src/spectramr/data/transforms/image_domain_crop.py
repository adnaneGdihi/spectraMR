"""Bring raw k-space to a fixed field of view by cropping or padding its IMAGE.

fastMRI stores k-space with a 2x oversampled readout -- knee ``640 x 368``, brain
``640 x 320`` -- while its reconstructions and every published baseline work on the
centre ``320 x 320`` of the image. Cropping the k-space array instead would keep the
low frequencies of the oversampled field of view: a blurred, still-oversampled image,
not the one the dataset's reference reconstructions show.

So the crop runs in image space: ``ifft2c``, centre crop, ``fft2c``. An axis shorter
than the target is zero-padded in image space rather than refused, because fastMRI's
own ``UnetDataTransform`` meets narrower brain FLAIR volumes the same way and a
uniform size is what the patch sampler needs.

A never-acquired line is an exact zero in every coil and slice, and the acquired-line
budget, the line-axis gate and the acquired/unacquired bin metrics all find the
acquired lines by that rule. The round trip leaves round-off (and, on a resized axis,
interpolation leakage) in those lines, so they would all read as acquired. Each
in-plane axis's support is therefore read before the transform, mapped onto the new
grid, and the lines outside it are set back to exact zeros.
"""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


def validate_image_size(size: object) -> tuple[int, int]:
    """Return ``size`` as a ``(height, width)`` pair of positive ints, or raise."""
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError(f"image_domain_crop_or_pad must be a [height, width] pair; got {size!r}.")
    height, width = size
    for axis in (height, width):
        if isinstance(axis, bool) or not isinstance(axis, int) or axis <= 0:
            raise ValueError(
                f"image_domain_crop_or_pad must hold two positive integers; got {size!r}."
            )
    return int(height), int(width)


def _centre_crop_or_pad(image: torch.Tensor, axis: int, target: int) -> torch.Tensor:
    """Centre-crop or zero-pad one axis of ``image`` to ``target`` samples."""
    current = image.shape[axis]
    if current > target:
        start = (current - target) // 2
        return image.narrow(axis, start, target)
    if current < target:
        before = (target - current) // 2
        shape = list(image.shape)
        shape[axis] = target
        padded = image.new_zeros(shape)
        padded.narrow(axis, before, current).copy_(image)
        return padded
    return image


def _line_support(kspace: torch.Tensor, axis: int) -> torch.Tensor:
    """1-D bool over ``axis``: the lines that carry energy anywhere in the volume."""
    dims = tuple(d for d in range(kspace.ndim) if d != axis % kspace.ndim)
    return kspace.abs().sum(dim=dims) > 0


def _remap_support(support: torch.Tensor, n_out: int) -> torch.Tensor:
    """``support`` of an ``n_in``-line axis on the ``n_out``-line grid of the resized image.

    Resizing the image by ``n_out / n_in`` keeps the k-space extent and scales the line
    spacing by ``n_in / n_out``, so output line ``j`` sits at input position
    ``(j - n_out // 2) * n_in / n_out + n_in // 2``; it is acquired when the nearest
    input line is. The identity when ``n_out == n_in``.
    """
    n_in = support.numel()
    if n_in == n_out:
        return support
    j = torch.arange(n_out, dtype=torch.float64, device=support.device)
    index = torch.round((j - n_out // 2) * n_in / n_out + n_in // 2).long()
    inside = (index >= 0) & (index < n_in)
    out = torch.zeros(n_out, dtype=torch.bool, device=support.device)
    out[inside] = support[index[inside]]
    return out


def crop_or_pad_kspace_in_image_domain(kspace: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize complex ``(C, H, W, D)`` k-space to ``(C, *size, D)`` through its image.

    Args:
        kspace: Complex k-space in the TorchIO layout the fastMRI subject builder
            uses -- coils on axis 0, the two in-plane axes on 1 and 2, slices on 3.
        size: Target in-plane ``(height, width)`` in image space.

    Returns:
        Complex k-space of the cropped or padded image, same dtype and device, with
        every line that was never acquired (an exact zero before) an exact zero again.

    Raises:
        TypeError: When ``kspace`` is real. Real/imaginary-interleaved channels
            would be transformed as independent real images, which is wrong.
        ValueError: When ``kspace`` is not 4-D or ``size`` is malformed.
    """
    height, width = validate_image_size(size)
    if not torch.is_complex(kspace):
        raise TypeError(
            "crop_or_pad_kspace_in_image_domain needs complex k-space; got "
            f"{kspace.dtype}. Apply it before any coil processing splits real and "
            "imaginary parts into channels."
        )
    if kspace.ndim != 4:
        raise ValueError(f"Expected (C, H, W, D) k-space; got shape {tuple(kspace.shape)}.")
    if kspace.shape[1] == height and kspace.shape[2] == width:
        return kspace
    rows = _remap_support(_line_support(kspace, 1), height)
    columns = _remap_support(_line_support(kspace, 2), width)
    image = ifft2c(kspace.permute(0, 3, 1, 2))
    image = _centre_crop_or_pad(image, -2, height)
    image = _centre_crop_or_pad(image, -1, width)
    out = fft2c(image).permute(0, 2, 3, 1)
    acquired = rows[None, :, None, None] & columns[None, None, :, None]
    return torch.where(acquired, out, torch.zeros_like(out)).contiguous()


__all__ = ["crop_or_pad_kspace_in_image_domain", "validate_image_size"]
