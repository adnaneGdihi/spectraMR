"""Which ``dataset_type`` values deliver k-space rather than images.

One home for a fact that decides real behaviour and was previously a
function-local constant inside ``infrastructure/validation/spec_card.py`` —
unreachable from the data layer, which must not import ``infrastructure/``
(non-negotiable #5).

The distinction is load-bearing rather than cosmetic. ``subject["input"]`` means
opposite things across it: on an image arm it is an image, on a k-space arm it
IS the k-space. Any transform that resolves its source by key name has to know
which, or it will silently treat measured k-space as an image — which is exactly
how ``PhysicsSynchronization`` came to apply a second forward FFT to it
(audit A4).
"""

from __future__ import annotations

#: ``dataset_type`` values that serve raw (multi-coil) k-space.
#:
#: Membership is the union of this set and a ``"kspace" in <type>`` substring
#: test, because the corpus spells the family several ways
#: (``bart_kspace``, ``fastmri_kspace``, …) and enumerating every one has
#: already proven to drift.
KSPACE_DATASET_TYPES: frozenset[str] = frozenset(
    {
        "kspace",
        "m4raw",
        "fastmri",
        "fastmri_kspace",
        "fastmri_brain",
        "fastmri_knee",
        "fastmri_cardiac",
        "fastmri_multicoil",
        # Phase-contrast slices: input/target are multicoil k-space per encoding.
        "flow_synthetic",
    }
)

__all__ = ["KSPACE_DATASET_TYPES", "is_kspace_dataset_type"]


def is_kspace_dataset_type(dataset_type: str | None) -> bool:
    """Whether ``dataset_type`` serves k-space as its primary signal.

    Args:
        dataset_type: The ``data.dataset_type`` value; ``None`` is treated as
            image-domain, matching the schema default.

    Returns:
        ``True`` when the arm's ``input``/``target`` carry k-space.
    """
    if not dataset_type:
        return False
    name = str(dataset_type)
    return "kspace" in name or name in KSPACE_DATASET_TYPES
