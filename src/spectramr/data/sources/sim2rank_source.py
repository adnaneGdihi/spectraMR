"""The volume-source seam sim2rank consumes.

Why a Protocol and not ``DataPipelineDirector``
-----------------------------------------------
``.claude/rules/data.md`` says *file -> Dataset code lives under
``src/spectramr/data/``* -- which is exactly where this lives. It does **not** say
"everything goes through the director". The director takes a ``BuilderContext``
carrying a frozen ``TrainingSettings``; sim2rank is an argparse analysis harness
with no training config, so routing it through the director would mean
**fabricating a fake TrainingSettings** -- a facade (pitfall #16) erected in the
name of following a rule that was never about this.

The precedent is already committed:
``spectramr.data.datasets.m4raw_repetition_groups`` says in its own docstring that
this grouping logic belongs under ``src/spectramr/data/`` and *not* in
``scripts/sim2rank``, and both the sweep and the CLI import it from here.

So: a narrow Protocol, in the data layer, that yields slices. ``scripts/sim2rank``
imports it; ``src/`` never imports ``scripts/`` (non-negotiable #5).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

__all__ = [
    "DEFAULT_SYNTHETIC_VOLUME_HWD",
    "Sim2RankSlice",
    "Sim2RankVolume",
    "Sim2RankVolumeSource",
    "synthetic_phantom_volume",
]

#: ``(H, W, D)`` of the ``--mode 3d --synthetic`` phantom; stamped into every 3-D bundle.
DEFAULT_SYNTHETIC_VOLUME_HWD: tuple[int, int, int] = (128, 128, 64)


@dataclass(frozen=True, slots=True)
class Sim2RankSlice:
    """One clean reference slice, plus everything needed to degrade and region it.

    The digital-twin simulator operates **slice-wise**, so the source's job is to
    turn volumes into slices. Regions ride along with the slice because they are a
    property of the clean reference (see
    :mod:`spectramr.data.regions.synthseg_labels`) -- computing them here, once,
    keeps them from being recomputed per artifact x severity, which would be
    ~10^5 redundant segmentations.
    """

    content_id: str
    """Stable identifier: ``<file_id>#<slice_index>``. Used as the covariate C by
    the rankers, so it must not encode anything that varies across the sweep."""

    clean: torch.Tensor
    """``[1, H, W]`` magnitude reference. The degradation input AND the metric target."""

    contrast: str
    """The 'anatomy' axis of the leaderboard: AXT1 / AXT1POST / AXT2 / AXFLAIR
    (fastMRI) or T1 / T2 / FLAIR (M4Raw)."""

    file_id: str
    slice_index: int

    kspace: torch.Tensor | None = None
    """Complex k-space, when the source has it. Physics metrics need the
    *measurement* -- an image ROI cannot restrict it, which is why they are
    NOT_RESTRICTABLE in the eligibility gate."""

    coil_maps: torch.Tensor | None = None

    region_masks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    """``region_id -> [H, W]`` bool. Keys are declared in
    :mod:`spectramr.data.regions.region_registry`; ``full`` is always present."""

    provenance: Mapping[str, object] = field(default_factory=dict)
    """Where every region came from. A region you cannot reproduce is not a result."""

    def __post_init__(self) -> None:
        if self.clean.ndim != 3 or self.clean.shape[0] != 1:
            raise ValueError(
                f"clean must be [1, H, W]; got {tuple(self.clean.shape)}. The "
                "simulator is slice-wise -- a volume here would be silently "
                "degraded as a batch of independent slices."
            )
        h, w = self.clean.shape[-2:]
        for region_id, mask in self.region_masks.items():
            if mask.shape != (h, w):
                raise ValueError(
                    f"region {region_id!r} has shape {tuple(mask.shape)} but the "
                    f"slice is {h}x{w}. A mask from another geometry would move the "
                    "region boundary."
                )
            if mask.dtype is not torch.bool:
                raise TypeError(
                    f"region {region_id!r} must be a bool mask, got {mask.dtype}; a "
                    "float mask invites silent partial weighting."
                )


@runtime_checkable
class Sim2RankVolumeSource(Protocol):
    """Yields clean slices with their regions attached.

    Implementations: :class:`~spectramr.data.sources.fastmri_brain_source.FastMRIBrainSource`
    (manifest-driven, QC-gated) and
    :class:`~spectramr.data.sources.m4raw_source.M4RawSource` (NEX pseudo-GT).
    """

    @property
    def name(self) -> str: ...

    @property
    def contrasts(self) -> Sequence[str]:
        """Every contrast this source can emit -- the leaderboard's anatomy axis."""
        ...

    def __iter__(self) -> Iterator[Sim2RankSlice]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True, slots=True)
class Sim2RankVolume:
    """One clean reference volume for the ``--mode 3d`` sweep.

    The volumetric operators degrade the whole stack at once (motion across slices,
    kz encoding), so the volume is never split into :class:`Sim2RankSlice` objects.
    """

    content_id: str
    clean: torch.Tensor
    """``[1, D, H, W]`` real magnitude; D is the slice (multislice) or kz (3-D) axis."""

    contrast: str
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.clean.ndim != 4 or self.clean.shape[0] != 1:
            raise ValueError(f"clean must be [1, D, H, W]; got {tuple(self.clean.shape)}.")
        if self.clean.is_complex():
            raise TypeError("clean must be a real magnitude volume, not complex.")
        if self.clean.shape[1] < 2:
            raise ValueError(
                f"a volume needs D >= 2 slices; got D={self.clean.shape[1]}. A single slice "
                "belongs to the 2-D path."
            )

    @property
    def size_hwd(self) -> dict[str, int]:
        d, h, w = self.clean.shape[-3:]
        return {"H": int(h), "W": int(w), "D": int(d)}

    def as_batch(self) -> torch.Tensor:
        """The canonical degradation layout ``(1, 1, D, H, W)``."""
        return self.clean.unsqueeze(0)


def synthetic_phantom_volume(
    height: int = DEFAULT_SYNTHETIC_VOLUME_HWD[0],
    width: int = DEFAULT_SYNTHETIC_VOLUME_HWD[1],
    depth: int = DEFAULT_SYNTHETIC_VOLUME_HWD[2],
    *,
    device: torch.device | None = None,
) -> Sim2RankVolume:
    """An ellipsoid phantom whose cross-section and inserts change along D.

    The 3-D analogue of the 2-D ``--synthetic`` phantom. The outer ellipsoid's z
    semi-axis exceeds the stack so every slice carries signal; the two inserts are
    off-centre in z, so the stack is not one slice repeated.
    """
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, depth, device=device),
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    vol = ((x / 0.7) ** 2 + (y / 0.9) ** 2 + (z / 1.2) ** 2 < 1.0).float()
    vol = torch.where((x - 0.2) ** 2 + (y + 0.1) ** 2 + (z - 0.3) ** 2 < 0.15, vol * 0.5, vol)
    vol = torch.where((x + 0.3) ** 2 + (y - 0.2) ** 2 + (z + 0.4) ** 2 < 0.1, vol * 0.8, vol)
    return Sim2RankVolume(
        content_id="synthetic_phantom#volume",
        clean=vol.unsqueeze(0),
        contrast="synthetic",
        provenance={"source": "synthetic_phantom_volume", "size": {"H": height, "W": width, "D": depth}},
    )
