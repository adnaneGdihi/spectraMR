r"""Joint k-space / image rotation operator (B-1 / Idea 1).

The equivariance-defect score
(:mod:`spectramr.core.metrics.equivariance_defect`) needs a rotation operator
:math:`R_\theta` that acts **consistently** in image space and k-space. The
2-D Fourier transform commutes with rotation about the origin,

.. math::

    \mathcal{F}\{R_\theta\, x\}(k) \;=\; R_\theta\,\bigl(\mathcal{F}\{x\}\bigr)(k),

so rotating the image by :math:`\theta` is the same operation as rotating the
(centred) k-space by :math:`\theta`. This module provides:

* :func:`rotate_image` — differentiable affine rotation of an
  ``[B, C, H, W]`` image (complex64 or real), via ``grid_sample`` on the real
  and imaginary parts separately so phase survives.
* :func:`rotate_kspace` — the same rotation applied to a centred k-space
  tensor by round-tripping through :func:`ifft2c` / :func:`fft2c` so the
  centring + ``norm="ortho"`` conventions are honoured (never raw
  ``torch.fft`` on k-space, per CLAUDE.md pitfall #2).
* :class:`JointKSpaceImageRotation` — a :class:`torchio.Transform` that draws a
  deterministic angle (and, for ``SE2``, an integer pixel shift) and applies it
  to the ``input`` / ``target`` image of a TorchIO subject.

Registration note
------------------
Registered as ``joint_rotation`` and declarable from YAML::

    data:
      processing:
        transforms:
          - name: joint_rotation
            kwargs: {group: SE2, seed: 0}

The rotation is applied jointly to every key in ``include_keys`` (``input``,
``target``, ``mri``) from ONE drawn group element. That is the property that
makes it safe to wire where a torchvision-style pipeline is not: independent
draws per key would rotate input and target differently, which is the leak
``tests/data_integrity/test_augmentation_leak.py`` exists to catch.
"""

from __future__ import annotations

import torch
import torchio as tio

from spectramr.data.transforms.registry import register_transform
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

SUPPORTED_GROUPS = ("SO2", "SE2")


def rotate_image(
    x: torch.Tensor,
    angle_rad: float,
    shift_px: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    r"""Apply :math:`R_\theta` to an image-domain tensor.

    Handles complex tensors by rotating the real and imaginary channels with
    the *same* sampling grid, so the rotation acts on the complex field as a
    whole (phase is rotated coherently, not discarded). The optional integer
    pixel shift (``SE2`` translation part) uses :func:`torch.roll` so it is
    exact and invertible.

    Args:
        x: ``[B, C, H, W]`` image (real or complex).
        angle_rad: Counter-clockwise rotation in radians.
        shift_px: ``(dy, dx)`` integer translation applied after rotation.

    Returns:
        Rotated tensor, same shape/dtype as ``x``.

    Raises:
        ValueError: if ``x`` is not 4-D.
    """
    if x.ndim != 4:
        raise ValueError(f"rotate_image expects [B, C, H, W]; got {tuple(x.shape)}.")

    if x.is_complex():
        real = _affine_rotate(x.real, angle_rad)
        imag = _affine_rotate(x.imag, angle_rad)
        out = torch.complex(real, imag)
    else:
        out = _affine_rotate(x, angle_rad)

    dy, dx = int(shift_px[0]), int(shift_px[1])
    if dy != 0 or dx != 0:
        out = torch.roll(out, shifts=(dy, dx), dims=(-2, -1))
    return out


def _affine_rotate(x: torch.Tensor, angle_rad: float) -> torch.Tensor:
    """Bilinear affine rotation about the image centre for a real ``[B,C,H,W]``."""
    b = x.shape[0]
    a = torch.tensor(angle_rad, dtype=x.dtype, device=x.device)
    cos, sin = torch.cos(a), torch.sin(a)
    theta = torch.zeros(b, 2, 3, dtype=x.dtype, device=x.device)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )


def rotate_kspace(
    k: torch.Tensor,
    angle_rad: float,
    shift_px: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    r"""Apply :math:`R_\theta` to a centred k-space tensor.

    Implemented as ``fft2c(rotate_image(ifft2c(k)))`` so the centring and
    orthonormal-FFT conventions are inherited from
    :mod:`spectramr.infrastructure.physics.fft_ops`. By the rotation-commutes-
    with-Fourier identity this equals rotating the k-space directly, up to
    bilinear-interpolation error.

    Args:
        k: ``[B, C, H, W]`` centred k-space (complex64 or real-encoded).
        angle_rad: Counter-clockwise rotation in radians.
        shift_px: ``(dy, dx)`` integer translation (applied in image space,
            which is a linear phase ramp in k-space).

    Returns:
        Rotated k-space, complex64, same spatial shape as ``k``.
    """
    image = ifft2c(k)
    rotated_image = rotate_image(image, angle_rad, shift_px)
    return fft2c(rotated_image)


@register_transform("joint_rotation", modifies=("input", "target", "mri"))
class JointKSpaceImageRotation(tio.Transform):
    r"""TorchIO transform applying a consistent rotation to image and k-space.

    Draws one group element per call (deterministically if ``seed`` is set) and
    rotates the configured subject images in **image space** (which, by the
    Fourier-rotation identity, is equivalent to the k-space rotation that
    :func:`rotate_kspace` performs). Used to build the rotated views the
    equivariance-defect score consumes.

    Args:
        angle_distribution: ``"uniform"`` over ``[0, 2*pi)`` (the only
            distribution implemented; anything else raises — no silent
            fallback per CLAUDE.md #9).
        num_samples: Retained for spec-compatibility / bookkeeping; one
            element is applied per ``apply_transform`` call.
        group: ``"SO2"`` (rotation only) or ``"SE2"`` (rotation + integer
            pixel shift).
        max_shift_px: Max integer pixel shift for the ``SE2`` translation part.
        seed: Optional RNG seed for deterministic angle/shift draws.
        include_keys: Subject image names to rotate.
    """

    def __init__(
        self,
        angle_distribution: str = "uniform",
        num_samples: int = 8,
        group: str = "SO2",
        max_shift_px: int = 4,
        seed: int | None = None,
        include_keys: tuple[str, ...] = ("input", "target", "mri"),
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if angle_distribution != "uniform":
            raise ValueError(
                f"Unsupported angle_distribution '{angle_distribution}'. "
                "Only 'uniform' is implemented."
            )
        if group not in SUPPORTED_GROUPS:
            raise ValueError(f"Unsupported group '{group}'. Expected one of {SUPPORTED_GROUPS}.")
        self.angle_distribution = angle_distribution
        self.num_samples = int(num_samples)
        self.group = group
        self.max_shift_px = int(max_shift_px)
        self.seed = seed
        # Per-instance draw counter so a seeded transform produces a *sequence*
        # of distinct elements (one per sample) rather than reseeding to the
        # same constant on every call — which would collapse the augmentation
        # to a single fixed rotation across the whole dataset. An int counter
        # stays picklable, so the transform still survives ``num_workers>0``
        # (holding a ``torch.Generator`` as state would not).
        self._draw_count = 0
        self.include_keys = tuple(include_keys)

    def _draw_element(self) -> tuple[float, tuple[int, int]]:
        gen = torch.Generator(device="cpu")
        if self.seed is not None:
            gen.manual_seed(int(self.seed) + self._draw_count)
            self._draw_count += 1
        angle = float((torch.rand(1, generator=gen) * (2.0 * torch.pi)).item())
        if self.group == "SE2":
            dy = int(
                torch.randint(-self.max_shift_px, self.max_shift_px + 1, (1,), generator=gen).item()
            )
            dx = int(
                torch.randint(-self.max_shift_px, self.max_shift_px + 1, (1,), generator=gen).item()
            )
            shift = (dy, dx)
        else:
            shift = (0, 0)
        return angle, shift

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Rotate every included image of ``subject`` by one drawn group element."""
        angle, shift = self._draw_element()
        for image_name in subject.get_images_names():
            if image_name not in self.include_keys:
                continue
            image = subject[image_name]
            data = image.data  # [C, H, W, D] (TorchIO layout)
            if data.ndim != 4:
                continue
            c, h, w, d = data.shape
            # Move spatial H,W to the [B,C,H,W] convention rotate_image wants:
            # treat (C * D) as batch, single channel, rotate the (H, W) plane.
            work = data.permute(0, 3, 1, 2).reshape(c * d, 1, h, w)
            rotated = rotate_image(work, angle, shift)
            restored = rotated.reshape(c, d, h, w).permute(0, 2, 3, 1).contiguous()
            image.set_data(restored)
        return subject


__all__ = [
    "SUPPORTED_GROUPS",
    "JointKSpaceImageRotation",
    "rotate_image",
    "rotate_kspace",
]
