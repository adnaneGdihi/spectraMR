r"""Emit complex ESPIRiT coil sensitivity maps into the subject.

Nothing in the loader produced coil maps. ``CoilCombineTransform`` *consumes* a
``sensitivity`` key for SENSE combination, and the TorchIO subject builder can
*load* maps when a manifest record carries a ``sensitivity_path``, but no route
computes them -- so an arm whose objective needs the coil geometry had no way to
get it short of precomputing files offline.

This computes them once per subject at load, from the subject's own k-space,
through the canonical :func:`estimate_csm_espirit`. Per subject, not per step:
the eigendecomposition is over ``(H*W, C, C)`` and has no business in a training
loop (non-negotiable 9).

**It emits COMPLEX maps, under a key the magnitude path cannot shadow.** The
subject builder's second branch stores ``sensitivity`` as ``sens_tensor.abs()``
and puts the complex map under ``sensitivity_complex``. A magnitude map is not
merely lossy for a consumer that needs phase -- it silently changes what is
computed, because ``I - s s^H`` with real ``s`` is a real projector that drops
the phase half of the coil null space. This writes both, so a SENSE combine
still finds what it expects while a phase-sensitive consumer gets the real
thing.

**Every slice is calibrated, not one.** Coil sensitivities vary along the slice
axis, and a depth-1 map on a depth-D subject also makes TorchIO's sampler raise
``check_consistent_space`` before a patch is ever drawn. The depth axis maps
onto the estimator's batch axis, so the whole volume is one vectorised call.

**No coil compression.** The maps describe the PHYSICAL array, so an arm using
them must serve its coils uncompressed (``data.coils.processing_mode: none``).
Compressing to virtual coils after estimating maps for the physical ones would
leave the two describing different arrays, which nothing downstream can detect.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torchio as tio

from spectramr.data.transforms.registry import register_transform

__all__ = ["ESPIRiTSensitivityTransform"]

#: Where the transform looks for k-space, in order. Mirrors the convention in
#: `PhysicsInformedMasking`, which prefers `kspace_raw` over `kspace`.
_KSPACE_KEYS: tuple[str, ...] = ("kspace_raw", "kspace", "input")


def _as_complex_slices(tensor: torch.Tensor) -> torch.Tensor:
    """``(2C, H, W, D)`` real-interleaved or ``(C, H, W, D)`` complex -> ``(D, C, H, W)``.

    The depth axis becomes the batch axis: ``estimate_csm_espirit`` is batched,
    so a D-slice volume is one call rather than D of them.
    """
    volume = tensor if tensor.ndim == 4 else tensor.unsqueeze(-1)
    if not torch.is_complex(volume):
        channels = volume.shape[0]
        if channels % 2 != 0:
            raise ValueError(
                f"espirit_sensitivity needs complex or real/imag-interleaved coil "
                f"k-space, got {channels} channels. An odd count means the coils were "
                "already combined, and ESPIRiT has no array left to calibrate."
            )
        volume = torch.complex(volume[0::2], volume[1::2])
    return volume.permute(3, 0, 1, 2)


@register_transform(
    "espirit_sensitivity",
    produces=("sensitivity", "sensitivity_complex"),
    requires=("kspace",),
)
class ESPIRiTSensitivityTransform(tio.transforms.Transform):
    """Estimate coil maps from the subject's k-space and attach them.

    Args:
        acs_size: ACS window for the calibration. The default matches the
            corpus convention; ``estimate_csm_espirit`` raises with the minimum
            for the coil count and kernel when it is too small to determine the
            kernel, so a wrong value fails loudly rather than producing a
            rank-deficient map.
        kernel_size: calibration kernel side.
        eigen_threshold: support threshold on the leading eigenvalue.
        keys: subject keys to look for k-space in, in order.
        strict: raise when no k-space key is present. ``False`` returns the
            subject untouched, which is only correct for a chain that mixes
            k-space and image-domain subjects.
    """

    def __init__(
        self,
        acs_size: int = 24,
        kernel_size: int = 6,
        eigen_threshold: float = 0.95,
        keys: Sequence[str] = _KSPACE_KEYS,
        strict: bool = True,
    ) -> None:
        super().__init__()
        self.acs_size = int(acs_size)
        self.kernel_size = int(kernel_size)
        self.eigen_threshold = float(eigen_threshold)
        self.keys = tuple(keys)
        self.strict = bool(strict)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        from spectramr.infrastructure.physics.coil_sensitivity import estimate_csm_espirit

        source = next((k for k in self.keys if k in subject), None)
        if source is None:
            if self.strict:
                raise ValueError(
                    "espirit_sensitivity found no k-space to calibrate from "
                    f"(looked for {list(self.keys)}, subject has "
                    f"{sorted(subject.keys())}). Returning the subject unchanged "
                    "would leave a consumer's maps absent with no error, so this "
                    "raises; pass strict=False to opt out deliberately."
                )
            return subject

        slices = _as_complex_slices(subject[source].data)
        maps = estimate_csm_espirit(
            slices,
            num_coils=slices.shape[1],
            kernel_size=self.kernel_size,
            acs_size=self.acs_size,
            eigen_threshold=self.eigen_threshold,
        )
        # Back to TorchIO's (C, H, W, D): the sampler crops the depth axis, and
        # both maps must travel with the images it crops.
        complex_maps = maps.permute(1, 2, 3, 0).contiguous()

        affine = subject[source].affine
        subject.add_image(tio.ScalarImage(tensor=complex_maps, affine=affine), "sensitivity_complex")
        # The magnitude spelling, for `CoilCombineTransform(method="sense")` and
        # anything else reading the builder's `.abs()` convention. Written under
        # its own key so neither shadows the other.
        subject.add_image(tio.ScalarImage(tensor=complex_maps.abs(), affine=affine), "sensitivity")
        return subject
