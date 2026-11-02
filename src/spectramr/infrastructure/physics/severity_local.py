"""Local severity: the one owner of user theta -> physical value -> native theta.

A user can give any registered degradation its own severity range in PHYSICAL
units (``cartesian_undersamp: {start: 2, end: 16, schedule: log}``). The sweep
then walks a *local* ``theta in [0, 1]`` whose ends are exactly the user's
``start`` and ``end``, while the registered degrader still expects its *native*
``theta`` against its declared :class:`~spectramr.infrastructure.physics.severity.PhysicalParam`.

:class:`LocalSeverity` maps one to the other. When the user range reaches past the
declared one, the primary parameter is redeclared with the user's value on that
side, and the degrader is told so through the side's endpoint kwarg; a side with
no endpoint kwarg cannot be extended and is refused. Co-varying parameters follow
the native theta along their declared ramps, and :meth:`LocalSeverity.provenance`
records the values they reach so the coupling is reported, not implied.

With no setting -- or a setting that restates the declared affine endpoints under
the linear schedule -- the resolver is a passthrough: the native theta *is* the
caller's theta, so existing sweeps stay bit-identical.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from spectramr.config.schemas.enums import AccelerationSchedule
from spectramr.config.schemas.severity_axes import SeverityAxis, load_file
from spectramr.infrastructure.physics.degradations.registry import DEGRADATION_REGISTRY
from spectramr.infrastructure.physics.severity import (
    PhysicalParam,
    SeverityDirection,
    SeveritySpec,
    severity_grid,
    shape_severity_ratio,
)

if TYPE_CHECKING:
    from spectramr.infrastructure.physics.scanner_profile import ScannerProfile

__all__ = ["LocalSeverity", "load_severity_axes_file", "reject_profile_conflicts", "resolve_axes"]

#: Shape handed to a spec's ``condition`` to learn which kwargs a profile injects; the
#: key set does not depend on the matrix size.
_PROFILE_PROBE_SHAPE: tuple[int, ...] = (1, 1, 64, 64)

_DEFAULT_POWER: float = SeverityAxis.model_fields["power"].default


@dataclass(frozen=True, slots=True)
class LocalSeverity:
    """One degradation's local severity axis, validated against its declared spec."""

    name: str
    spec: SeveritySpec
    setting: SeverityAxis | None
    _native: PhysicalParam = field(init=False, repr=False, compare=False)
    _kwargs: tuple[tuple[str, float], ...] = field(init=False, repr=False, compare=False)
    _passthrough: bool = field(init=False, repr=False, compare=False)

    @classmethod
    def resolve(cls, name: str, setting: SeverityAxis | None) -> LocalSeverity:
        """Build the resolver for registered degradation ``name``; unknown names raise ``KeyError``."""
        if name not in DEGRADATION_REGISTRY:
            raise KeyError(f"unknown degradation {name!r}; it is not in DEGRADATION_REGISTRY")
        return cls(name, DEGRADATION_REGISTRY[name].severity, setting)

    def __post_init__(self) -> None:
        primary = self.spec.primary
        s = self.setting
        passthrough = s is None or (
            primary.curve == "affine"
            and s.start == primary.at_theta_min
            and s.end == primary.at_theta_max
            and s.schedule is AccelerationSchedule.LINEAR
            and s.power == _DEFAULT_POWER
        )
        object.__setattr__(self, "_passthrough", passthrough)
        if s is None or passthrough:
            object.__setattr__(self, "_native", primary)
            object.__setattr__(self, "_kwargs", ())
            return
        self._validate(primary, s)
        native, kwargs = self._redeclare(primary, s)
        object.__setattr__(self, "_native", native)
        object.__setattr__(self, "_kwargs", kwargs)

    def _validate(self, p: PhysicalParam, s: SeverityAxis) -> None:
        if s.schedule is AccelerationSchedule.LOG and not s.start * s.end > 0:
            raise ValueError(
                f"{self.name}: the log schedule needs start and end of one sign with "
                f"start * end > 0; got start={s.start}, end={s.end}"
            )
        for label, v in (("start", s.start), ("end", s.end)):
            if (p.valid_min is not None and v < p.valid_min) or (
                p.valid_max is not None and v > p.valid_max
            ):
                raise ValueError(
                    f"{self.name}: {label}={v} {p.units} is outside the valid range "
                    f"[{p.valid_min}, {p.valid_max}] of {p.name}"
                )
        sign = 1.0 if p.direction is SeverityDirection.INCREASING else -1.0
        if sign * (s.end - s.start) < 0:
            raise ValueError(
                f"{self.name}: start={s.start} -> end={s.end} runs against the declared "
                f"severity direction of {p.name} ({p.direction.value}: "
                f"{p.at_theta_min} -> {p.at_theta_max})"
            )
        if p.curve == "one_minus_exp":
            raise ValueError(
                f"{self.name}: primary {p.name} has a one_minus_exp curve with no inverse; "
                "a physical range cannot be mapped to its native theta"
            )

    def _redeclare(
        self, p: PhysicalParam, s: SeverityAxis
    ) -> tuple[PhysicalParam, tuple[tuple[str, float], ...]]:
        sign = 1.0 if p.direction is SeverityDirection.INCREASING else -1.0
        lo, hi = p.at_theta_min, p.at_theta_max
        kwargs: list[tuple[str, float]] = []
        sides = (
            ("start (theta=0)", s.start, sign * (s.start - lo) < 0, p.endpoint_kwargs[0]),
            ("end (theta=1)", s.end, sign * (s.end - hi) > 0, p.endpoint_kwargs[1]),
        )
        for index, (side, value, outside, kw) in enumerate(sides):
            if not outside:
                continue
            if kw is None:
                raise ValueError(
                    f"{self.name}: the {side} value {value} {p.units} lies outside the "
                    f"declared range [{p.at_theta_min}, {p.at_theta_max}] of {p.name}, "
                    "and that side declares no endpoint kwarg to extend it"
                )
            kwargs.append((kw, value * p.endpoint_scale))
            if index == 0:
                lo = value
            else:
                hi = value
        return dataclasses.replace(p, at_theta_min=lo, at_theta_max=hi), tuple(kwargs)

    @property
    def is_passthrough(self) -> bool:
        """True when the native theta is the caller's theta, unchanged."""
        return self._passthrough

    def thetas(self, n_steps: int, theta_min: float | None = None) -> tuple[float, ...]:
        """The local theta grid; a setting's ``n_steps`` overrides the caller's."""
        s = self.setting
        n = s.n_steps if s is not None and s.n_steps is not None else n_steps
        if self._passthrough:
            return tuple(severity_grid(n, theta_min=theta_min))
        # theta = 0 is the user's start, so it must be a level of the sweep.
        return tuple(severity_grid(n, theta_min=0.0))

    def physical(self, theta: float) -> float:
        """Primary physical value at local ``theta``; exactly ``start``/``end`` at 0/1."""
        t = float(theta)
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"{self.name}: local theta must lie in [0, 1], got {t}")
        s = self.setting
        if s is None or self._passthrough:
            p = self.spec.primary
            return p.at_theta_min if t == 0.0 else p.at_theta_max if t == 1.0 else p.value_at(t)
        if t == 0.0:
            return s.start
        if t == 1.0:
            return s.end
        if s.schedule is AccelerationSchedule.LOG:
            return float(s.start * (s.end / s.start) ** t)
        ratio = shape_severity_ratio(t, s.schedule.value, s.power)
        # A shaped ratio that reaches an end (step) must return that end exactly.
        if ratio == 0.0:
            return s.start
        if ratio == 1.0:
            return s.end
        return s.start + (s.end - s.start) * ratio

    def native_theta(self, theta: float) -> float:
        """The theta the registered degrader expects; ``theta`` itself when passthrough."""
        if self._passthrough:
            return theta
        native = self._native.theta_at(self.physical(theta))
        if not 0.0 <= native <= 1.0:
            raise ValueError(
                f"{self.name}: local theta {theta} maps to native theta {native}, outside [0, 1]"
            )
        return native

    def degrader_kwargs(self) -> dict[str, float]:
        """Endpoint kwargs that redeclare the degrader's range; empty when it is not extended."""
        return dict(self._kwargs)

    def provenance(self) -> dict[str, object]:
        """A JSON-serialisable record of the axis as declared and as applied."""
        p = self.spec.primary
        s = self.setting
        native_ends = (self.native_theta(0.0), self.native_theta(1.0))
        return {
            "name": self.name,
            "units": p.units,
            "primary": p.name,
            "start": self.physical(0.0),
            "end": self.physical(1.0),
            "schedule": AccelerationSchedule.LINEAR.value if s is None else s.schedule.value,
            "power": _DEFAULT_POWER if s is None else s.power,
            "n_steps": None if s is None else s.n_steps,
            "passthrough": self._passthrough,
            "declared": [p.at_theta_min, p.at_theta_max],
            "native_range": [self._native.at_theta_min, self._native.at_theta_max],
            "native_theta": list(native_ends),
            "degrader_kwargs": self.degrader_kwargs(),
            "co_varying": [
                {
                    "name": q.name,
                    "units": q.units,
                    "at_start": q.value_at(native_ends[0]),
                    "at_end": q.value_at(native_ends[1]),
                }
                for q in self.spec.co_varying
            ],
        }


def resolve_axes(
    names: Iterable[str], settings: Mapping[str, SeverityAxis]
) -> dict[str, LocalSeverity]:
    """Resolve every axis in ``names``; a name absent from ``settings`` is a passthrough."""
    ordered = tuple(names)
    unknown = sorted(set(settings) - set(ordered))
    if unknown:
        raise ValueError(
            f"severity settings name axes outside the swept set: {unknown}; swept: {list(ordered)}"
        )
    return {n: LocalSeverity.resolve(n, settings.get(n)) for n in ordered}


def load_severity_axes_file(path: str | Path) -> dict[str, SeverityAxis]:
    """A severity-axes YAML whose every name is a registered degradation.

    Raises ``FileNotFoundError`` on a missing file and ``ValueError`` on a schema error or
    an unregistered name.
    """
    settings = load_file(path)
    unknown = sorted(set(settings) - set(DEGRADATION_REGISTRY))
    if unknown:
        raise ValueError(f"{path}: {unknown} are not registered degradations")
    return settings


def reject_profile_conflicts(local: LocalSeverity, profile: ScannerProfile) -> None:
    """Raise when ``profile`` injects a kwarg that would override ``local``'s range.

    A scanner profile conditions a degrader through kwargs; one that sets the axis's
    physical parameter or an endpoint kwarg would replace the user's range while the
    provenance still records it.
    """
    from spectramr.infrastructure.physics.degradations.declarations import (
        PHYSICAL_OVERRIDE_KWARGS,
    )

    condition = DEGRADATION_REGISTRY[local.name].condition
    if local.is_passthrough or condition is None:
        return
    injected = set(condition(profile, 1.0, profile.geometry, _PROFILE_PROBE_SHAPE))
    owned = PHYSICAL_OVERRIDE_KWARGS.get(local.name, frozenset()) | set(local.degrader_kwargs())
    clash = sorted(injected & owned)
    if clash:
        raise ValueError(
            f"scanner profile {profile.name!r} sets {clash} on {local.name!r}, which "
            f"overrides the severity range given for {local.name!r}; drop the range or "
            "the profile"
        )
