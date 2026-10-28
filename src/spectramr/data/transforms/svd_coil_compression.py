r"""SVD coil compression fitted on the masked input, applied to every k-space key.

The M4Raw repetition route refuses ``coils.processing_mode: svd``: the loader's
compression would fit its basis on the fully sampled repetitions, before any
mask exists, so the basis would carry information from lines the arm never
acquired and, under a leave-one-out reference, from the reference itself. This
transform runs after ``complementary_rep_masking`` and fits the basis on the
masked ``input`` alone -- the data the arm is allowed to see.

Coil compression is a per-pixel linear map :math:`U^H` on the coil axis, and the
mask acts on k-space positions, so :math:`U^H (M y) = M (U^H y)`: compressing the
masked input equals masking the compressed one, and applying the same
:math:`U` to ``target``, ``kspace``, ``measured_kspace`` and ``input_alt`` keeps
every k-space tensor of the subject in one virtual-coil space without letting
any of them choose it.

The basis math is :func:`~spectramr.infrastructure.physics.coil_compression.fit_svd_basis`
and :func:`~spectramr.infrastructure.physics.coil_compression.apply_svd_compression`;
this module only decides what is fitted on and what is compressed.
"""

from __future__ import annotations

import torch
import torchio as tio

from spectramr.core.coil_map_names import REFERENCE_MAPS_KEY
from spectramr.data.transforms.registry import register_transform
from spectramr.data.transforms.tio_physics import MEASURED_KSPACE_KEY
from spectramr.infrastructure.physics.coil_compression import (
    apply_svd_compression,
    fit_svd_basis,
)

__all__ = ["COMPRESSED_KEYS", "STAMP_KEY", "SVDCoilCompressionOnInput", "compress_subject"]

#: Subject key recording that the compression ran and to how many coils, so a
#: batch says which coil space its k-space is in.
STAMP_KEY: str = "coil_compression"

#: The k-space keys compressed with the input's basis, when present.
COMPRESSED_KEYS: tuple[str, ...] = (
    "input",
    MEASURED_KSPACE_KEY,
    "target",
    "kspace",
    "input_alt",
)

#: Keys that describe the PHYSICAL coils; after compression they would describe
#: an array the subject no longer carries, so their presence is refused.
_PHYSICAL_COIL_KEYS: tuple[str, ...] = (
    "sensitivity_complex",
    "sensitivity",
    "noise_covariance",
    REFERENCE_MAPS_KEY,
)


def _slices(data: torch.Tensor) -> torch.Tensor:
    """TorchIO's ``(2C, H, W, S)`` / ``(C, H, W, S)`` -> ``(S, 2C | C, H, W)``."""
    return data.permute(3, 0, 1, 2)


def compress_subject(subject: tio.Subject, num_virtual_coils: int) -> list[torch.Tensor]:
    """Fit one basis per slice on ``input`` and project every present k-space key.

    Returns the per-slice ``(C, K)`` bases, for the caller's provenance and tests.
    """
    source = _slices(subject["input"].data)
    bases = [fit_svd_basis(source[s : s + 1], num_virtual_coils) for s in range(source.shape[0])]
    for key in COMPRESSED_KEYS:
        if key not in subject:
            continue
        image = subject[key]
        slices = _slices(image.data)
        compressed = torch.cat(
            [apply_svd_compression(slices[s : s + 1], basis) for s, basis in enumerate(bases)]
        )
        image.set_data(compressed.permute(1, 2, 3, 0).contiguous())
    return bases


@register_transform("svd_coil_compression", produces=(STAMP_KEY,), requires=("input", "mask"))
class SVDCoilCompressionOnInput(tio.transforms.Transform):
    """Compress every k-space key to ``num_virtual_coils`` with the masked input's basis.

    Declared after ``complementary_rep_masking`` in ``data.processing.transforms``.

    Args:
        num_virtual_coils: ``K >= 1`` virtual coils; at most the physical count.

    Raises:
        ValueError: at apply time, when ``mask`` is absent (the masking has not run
            yet, so the basis would see unacquired lines) or a physical-coil key
            (coil maps, a noise covariance) is present.
    """

    def __init__(self, num_virtual_coils: int = 1) -> None:
        super().__init__()
        if int(num_virtual_coils) < 1:
            raise ValueError(f"num_virtual_coils must be >= 1, got {num_virtual_coils!r}.")
        self.num_virtual_coils = int(num_virtual_coils)

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if "mask" not in subject:
            raise ValueError(
                "svd_coil_compression fits its basis on the MASKED input and found no "
                "`mask`: declare it after complementary_rep_masking in "
                "data.processing.transforms, or the basis sees lines the arm never acquired."
            )
        physical = [key for key in _PHYSICAL_COIL_KEYS if key in subject]
        if physical:
            raise ValueError(
                f"svd_coil_compression found {physical}, which describe the physical "
                "coils and would not match the compressed array. A virtual-coil arm "
                "declares no coil maps (data.coil_maps_root, espirit_sensitivity)."
            )
        compress_subject(subject, self.num_virtual_coils)
        subject[STAMP_KEY] = f"svd_on_masked_input;num_virtual_coils={self.num_virtual_coils}"
        return subject
