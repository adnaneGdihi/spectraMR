"""Per-degradation severity ranges in physical units (the user-facing schema).

Each entry gives one registered degradation its own ramp: the physical value at
local ``theta = 0`` (``start``) and at ``theta = 1`` (``end``), the schedule between
them, and optionally its own step count. Only what needs no registry is checked
here; whether the range is physically valid for that degradation is decided by
:class:`~spectramr.infrastructure.physics.severity_local.LocalSeverity`, which owns
the declared spec.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from spectramr.config.schemas.enums import AccelerationSchedule

__all__ = ["SeverityAxesFile", "SeverityAxis", "load_file", "reject_severity_overlap"]


class SeverityAxis(BaseModel):
    """One degradation's physical severity range and schedule."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    start: float
    end: float
    n_steps: int | None = Field(default=None, ge=2)
    # log is accepted here on purpose: it is defined on physical endpoints.
    schedule: AccelerationSchedule = AccelerationSchedule.LINEAR
    power: float = Field(default=2.0, gt=0.0)

    @model_validator(mode="after")
    def _distinct_endpoints(self) -> SeverityAxis:
        if self.start == self.end:
            raise ValueError(f"start and end must differ, both are {self.start}")
        return self


def reject_severity_overlap(
    severity_axes: Mapping[str, object], **other_owners: Mapping[str, object]
) -> None:
    """Raise when an axis in ``severity_axes`` is also keyed in another severity owner."""
    for field, owner in other_owners.items():
        both = sorted(set(severity_axes) & set(owner))
        if both:
            raise ValueError(
                f"{both} appear in both severity_axes and {field}: one owner per axis. "
                f"severity_axes carries the range and schedule, so drop them from {field}."
            )


class SeverityAxesFile(BaseModel):
    """A file of per-degradation severity axes, keyed by registered degradation name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    axes: dict[str, SeverityAxis]


def load_file(path: str | Path) -> dict[str, SeverityAxis]:
    """Parse a severity-axes YAML into ``{name: SeverityAxis}``; shape only, no registry check.

    A missing file raises ``FileNotFoundError``; malformed YAML or a schema error raises
    ``ValueError``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file {str(p)!r}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{p}: {exc}") from exc
    return dict(SeverityAxesFile.model_validate(raw).axes)
