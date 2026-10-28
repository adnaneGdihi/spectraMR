"""One ``ENTRIES`` tuple per degradation family; :mod:`..registry` concatenates them.

Every entry declares, with no defaults, what its severity means physically, which image
components it perturbs and where in the imaging chain it arises, so a degradation cannot
be added without writing those down (the import fails instead).

* Severity -- each ``SeveritySpec`` is transcribed from the degrader's own theta ->
  physical mapping at its DEFAULT kwargs. If you change a degrader's ramp, change the spec
  in the same edit. A caller overriding e.g. ``R_max`` overrides the physical range too;
  the sweep records the resolved value in its provenance.
* Affects (issue #223) -- the sweep scores ``|D(x, theta)|``, so an axis that moves ONLY
  the phase is handed to the metrics as a constant. ``concomitant_phase`` (D23) is that
  axis and the only one. The flags are measured, not asserted:
  ``test_digital_twin_extensions.py::TestAffectsDeclaration`` re-derives every one.
* Origin -- SCANNER: hardware / field / receiver-chain imperfections. ACQUISITION:
  sequence + sampling choices. PATIENT: subject-induced. This is what makes
  ``list_degradations(origin=...)`` a real query rather than a label.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from torch import Tensor

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.adapters import Fold, per_slice
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.degradations.spec import (
    ActivationGate,
    DegradationSpec,
    ModeImpl,
    Realisation,
    SliceKernel,
    entry,
)
from spectramr.infrastructure.physics.severity import SeveritySpec

__all__ = ["MS", "VOL", "ModeDecl", "exact", "independent", "native_2d"]

MS = AcquisitionMode.MULTISLICE_2D
VOL = AcquisitionMode.VOLUME_3D

# One extra mode's implementation and how it relates to the physics.
ModeDecl = tuple[ModeImpl, Realisation]


def exact(fn: Callable[..., Tensor], *, volume_kwargs: Iterable[str] = ()) -> ModeDecl:
    """The 2-D kernel run once per subject over every slice: exact for separable physics."""
    return per_slice(fn, Fold.SHARED, volume_kwargs=volume_kwargs), Realisation.PER_SLICE_EXACT


def independent(fn: Callable[..., Tensor]) -> ModeDecl:
    """The 2-D kernel run with a fresh draw per slice: each slice is its own receive event."""
    return per_slice(fn, Fold.INDEPENDENT), Realisation.PER_SLICE_INDEPENDENT


def native_2d(
    *,
    code: str,
    name: str,
    family: str,
    fn: Callable[..., Tensor] | SliceKernel,
    description: str,
    severity: SeveritySpec,
    affects: frozenset[str],
    origin: ArtifactOrigin,
    modes: Mapping[AcquisitionMode, ModeDecl] | None = None,
    condition: Callable[..., dict[str, object]] | None = None,
    active_when: frozenset[ActivationGate] = frozenset(),
) -> DegradationSpec:
    """Declare a degradation by its native 2-D kernel ``fn``, plus any volume ``modes``.

    A bare ``fn`` is wrapped in :class:`SliceKernel`; pass a ``VoxelSliceKernel`` for a kernel
    whose severity is in mm, so it reads the slice geometry's voxel size.

    ``condition`` derives scanner-dependent kwargs from a profile; ``active_when`` names the
    protocol facts without which a scanner does not produce the artefact.
    """
    extra = dict(modes or {})
    if AcquisitionMode.SLICE_2D in extra:
        raise ValueError(f"{name!r}: the slice_2d implementation is fn, not a modes entry")
    return entry(
        code=code,
        name=name,
        family=family,
        description=description,
        severity=severity,
        affects=affects,
        origin=origin,
        implementations={
            AcquisitionMode.SLICE_2D: fn if isinstance(fn, SliceKernel) else SliceKernel(fn),
            **{m: impl for m, (impl, _) in extra.items()},
        },
        realisation={
            AcquisitionMode.SLICE_2D: Realisation.NATIVE,
            **{m: r for m, (_, r) in extra.items()},
        },
        requires=frozenset(),
        condition=condition,
        active_when=active_when,
    )
