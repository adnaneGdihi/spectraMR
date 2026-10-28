"""The declaration every degradation makes: what it is, and how it runs in each acquisition mode."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from torch import Tensor

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.severity import SeveritySpec

__all__ = [
    "ActivationGate",
    "DegradationSpec",
    "MissingInput",
    "ModeImpl",
    "Realisation",
    "Requirement",
    "SliceKernel",
    "UnsupportedAcquisitionMode",
    "VoxelSliceKernel",
    "entry",
]

# Invoked as impl(x, theta, seed, geometry, **kwargs).
ModeImpl = Callable[..., Tensor]

_AFFECTS = frozenset({"magnitude", "phase"})


class Realisation(StrEnum):
    """How a mode's implementation relates to the physics it simulates."""

    NATIVE = "native"
    PER_SLICE_EXACT = "per_slice_exact"
    PER_SLICE_INDEPENDENT = "per_slice_independent"


class Requirement(StrEnum):
    """An input a degradation cannot run without."""

    MULTICOIL = "multicoil"
    QMAPS = "qmaps"
    FAT_MAP = "fat_map"
    VESSEL_MAP = "vessel_map"
    TRAJECTORY_RADIAL = "trajectory_radial"
    SEQUENCE_EPI = "sequence_epi"


class ActivationGate(StrEnum):
    """A protocol fact without which a scanner does not produce the artefact at all.

    Unlike a :class:`Requirement` it never blocks a call; only ``active_axes`` reads it.
    """

    SEQUENCE_EPI = "sequence_epi"
    ACCELERATED = "accelerated"
    FAT_PRESENT = "fat_present"


class UnsupportedAcquisitionMode(ValueError):  # noqa: N818 -- name fixed by the contract
    """A degradation was asked to run in a mode it does not implement."""

    def __init__(
        self,
        name: str,
        mode: AcquisitionMode,
        supported: frozenset[AcquisitionMode],
        *,
        detail: str = "",
    ):
        self.name, self.mode, self.supported, self.detail = name, mode, supported, detail
        listed = ", ".join(sorted(m.value for m in supported))
        super().__init__(
            f"degradation {name!r} does not support acquisition mode {mode}; supported: {listed}"
            + (f"; {detail}" if detail else "")
        )

    def __reduce__(self) -> tuple[Any, ...]:
        # DataLoader workers pickle exceptions; the default reduce passes only the message.
        return (_unsupported, (self.name, self.mode, self.supported, self.detail))


class MissingInput(ValueError):  # noqa: N818 -- name fixed by the contract
    """A degradation's declared requirement was not supplied."""

    def __init__(self, name: str, requirement: Requirement):
        self.name, self.requirement = name, requirement
        super().__init__(f"degradation {name!r} requires input {requirement} but none was given")

    def __reduce__(self) -> tuple[Any, ...]:
        return (MissingInput, (self.name, self.requirement))


def _unsupported(
    name: str, mode: AcquisitionMode, supported: frozenset[AcquisitionMode], detail: str
) -> UnsupportedAcquisitionMode:
    return UnsupportedAcquisitionMode(name, mode, supported, detail=detail)


class SliceKernel:
    """A 2-D kernel ``kernel(x, theta, seed=..., **kw)`` adapted to the mode-impl signature."""

    __slots__ = ("kernel",)

    def __init__(self, kernel: Callable[..., Tensor]):
        self.kernel = kernel

    def __call__(self, x: Tensor, theta: float, seed: int, geometry: Any, **kwargs: Any) -> Tensor:
        return self.kernel(x, theta, seed=seed, **kwargs)


class VoxelSliceKernel(SliceKernel):
    """A 2-D kernel declared in mm: also handed the slice geometry's ``voxel_mm`` (``None``: unknown)."""

    __slots__ = ()

    def __call__(self, x: Tensor, theta: float, seed: int, geometry: Any, **kwargs: Any) -> Tensor:
        if "voxel_mm" in kwargs:
            raise ValueError("voxel_mm comes from the acquisition geometry, not a kwarg")
        return self.kernel(x, theta, seed=seed, voxel_mm=geometry.voxel_mm, **kwargs)


@dataclass(frozen=True, slots=True)
class DegradationSpec:
    """One degradation of the taxonomy, with an implementation per supported mode.

    ``realisation`` states how each mode's implementation relates to the physics; a
    mode whose realisation depends on the geometry declares ``native`` and branches
    inside its implementation.
    """

    code: str
    name: str
    family: str
    description: str
    severity: SeveritySpec
    affects: frozenset[str]
    origin: ArtifactOrigin
    implementations: Mapping[AcquisitionMode, ModeImpl]
    realisation: Mapping[AcquisitionMode, Realisation]
    requires: frozenset[Requirement]
    # None is a real state: the degradation takes no conditioning input. Called as
    # condition(profile, theta, geometry, shape) -> kwargs.
    condition: Callable[..., Any] | None = None
    active_when: frozenset[ActivationGate] = frozenset()
    # Why a mode is absent from ``implementations``; carried by the error raised for it.
    unsupported_detail: Mapping[AcquisitionMode, str] = MappingProxyType({})

    def __post_init__(self) -> None:
        tag = f"DegradationSpec {self.name!r}"
        if not isinstance(self.severity, SeveritySpec):
            raise ValueError(f"{tag}: severity must be a SeveritySpec")
        if not isinstance(self.origin, ArtifactOrigin):
            raise ValueError(f"{tag}: origin must be an ArtifactOrigin, got {self.origin!r}")
        if not self.affects or not set(self.affects) <= _AFFECTS:
            raise ValueError(
                f"{tag}: affects must be a non-empty subset of {sorted(_AFFECTS)}, "
                f"got {sorted(self.affects)}"
            )
        if not self.implementations:
            raise ValueError(f"{tag}: implementations must declare at least one mode")
        if set(self.implementations) != set(self.realisation):
            raise ValueError(
                f"{tag}: implementations and realisation must declare the same modes, got "
                f"{sorted(map(str, self.implementations))} vs {sorted(map(str, self.realisation))}"
            )
        for mode, impl in self.implementations.items():
            if not isinstance(mode, AcquisitionMode) or not callable(impl):
                raise ValueError(f"{tag}: implementations must map AcquisitionMode -> callable")
            if not isinstance(self.realisation[mode], Realisation):
                raise ValueError(f"{tag}: realisation[{mode}] must be a Realisation")
        if AcquisitionMode.SLICE_2D in self.realisation:
            if self.realisation[AcquisitionMode.SLICE_2D] is not Realisation.NATIVE:
                raise ValueError(f"{tag}: the slice_2d implementation must be native")
        elif any(r is not Realisation.NATIVE for r in self.realisation.values()):
            # A per-slice adapter wraps a 2-D kernel, so without one every mode is native.
            raise ValueError(f"{tag}: without a slice_2d kernel every realisation must be native")
        if not all(isinstance(r, Requirement) for r in self.requires):
            raise ValueError(f"{tag}: requires must contain Requirement members")
        if not all(isinstance(g, ActivationGate) for g in self.active_when):
            raise ValueError(f"{tag}: active_when must contain ActivationGate members")
        if self.condition is not None and not callable(self.condition):
            raise ValueError(f"{tag}: condition must be callable or None")
        malformed = [
            m
            for m, text in self.unsupported_detail.items()
            if not isinstance(m, AcquisitionMode) or not isinstance(text, str) or not text
        ]
        if malformed:
            raise ValueError(f"{tag}: unsupported_detail must map AcquisitionMode -> reason")
        clash = sorted(str(m) for m in self.unsupported_detail if m in self.implementations)
        if clash:
            raise ValueError(f"{tag}: unsupported_detail names implemented modes {clash}")
        object.__setattr__(
            self, "unsupported_detail", MappingProxyType(dict(self.unsupported_detail))
        )
        object.__setattr__(self, "affects", frozenset(self.affects))
        object.__setattr__(self, "requires", frozenset(self.requires))
        object.__setattr__(self, "active_when", frozenset(self.active_when))
        object.__setattr__(self, "implementations", MappingProxyType(dict(self.implementations)))
        object.__setattr__(self, "realisation", MappingProxyType(dict(self.realisation)))

    @property
    def supported_modes(self) -> frozenset[AcquisitionMode]:
        return frozenset(self.implementations)

    @property
    def phase_only(self) -> bool:
        """True when the degradation leaves ``|x|`` untouched and only moves phase."""
        return self.affects == frozenset({"phase"})

    @property
    def affects_magnitude(self) -> bool:
        return "magnitude" in self.affects

    @property
    def affects_phase(self) -> bool:
        return "phase" in self.affects

    @property
    def fn(self) -> Callable[..., Tensor]:
        """The bare 2-D kernel, called as ``fn(x, theta, seed=seed, **kwargs)``."""
        impl = self.implementations.get(AcquisitionMode.SLICE_2D)
        if impl is None:
            raise UnsupportedAcquisitionMode(
                self.name,
                AcquisitionMode.SLICE_2D,
                self.supported_modes,
                detail=self.unsupported_detail.get(AcquisitionMode.SLICE_2D, ""),
            )
        if not isinstance(impl, SliceKernel):
            raise TypeError(f"degradation {self.name!r}: its slice_2d impl wraps no bare kernel")
        return impl.kernel


def entry(
    *,
    code: str,
    name: str,
    family: str,
    description: str,
    severity: SeveritySpec,
    affects: frozenset[str],
    origin: ArtifactOrigin,
    implementations: Mapping[AcquisitionMode, ModeImpl],
    realisation: Mapping[AcquisitionMode, Realisation],
    requires: frozenset[Requirement],
    condition: Callable[..., Any] | None,
    active_when: frozenset[ActivationGate],
    unsupported_detail: Mapping[AcquisitionMode, str] = MappingProxyType({}),
) -> DegradationSpec:
    """Declare a degradation; no defaults, so a forgotten declaration fails at import."""
    return DegradationSpec(
        code=code,
        name=name,
        family=family,
        description=description,
        severity=severity,
        affects=affects,
        origin=origin,
        implementations=implementations,
        realisation=realisation,
        requires=requires,
        condition=condition,
        active_when=active_when,
        unsupported_detail=unsupported_detail,
    )
