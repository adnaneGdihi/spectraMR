r"""Declared contrasts: which arms are compared, against which baseline, in which folder.

A contrast set is a YAML file naming the comparisons a paper needs. Each contrast
is written alone into ``<out-root>/<name>/`` (:mod:`.contrast_folder`), so the
folder of one comparison can be copied into a manuscript without the others.

Unknown keys raise at load, as do duplicate names, a baseline that is not one of
the arms, and a name that is not a safe folder name.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..views import VIEWS
from .metric_selection import PHASES, check_phases

#: A folder name that cannot climb out of the output root.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArmRef(_Frozen):
    """One run in a contrast: a directory under ``results_root`` (or absolute), and its label."""

    run: str
    label: str | None = None
    #: figure group for the grouped bar charts; arms without one share the group ``arms``
    group: str | None = None

    @property
    def name(self) -> str:
        return self.label or Path(self.run).name


class Ladder(_Frozen):
    """A metric drawn per arm against named reference values (``cmp_metric_ladder``).

    ``references`` maps a label to a value on the metric's own scale, e.g. for
    ``nex_equivalent`` the identity ``1`` and the averages ``2`` and ``3`` an acquisition
    of that many excitations would reach.
    """

    metric: str
    references: dict[str, float] = {}


class Contrast(_Frozen):
    """Two or more arms read together; every non-baseline arm is tested against the baseline."""

    name: str
    arms: tuple[ArmRef, ...]
    baseline: str | None = None
    #: contrasts sharing a family are Holm-corrected together, per metric
    family: str | None = None
    title: str | None = None
    #: optional: also draw ``metric`` per arm against reference values
    ladder: Ladder | None = None

    @field_validator("name")
    @classmethod
    def _folder_safe(cls, value: str) -> str:
        if not _SAFE_NAME.match(value):
            raise ValueError(f"contrast name {value!r} is not a safe folder name")
        return value

    @field_validator("arms", mode="before")
    @classmethod
    def _bare_run_names(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise ValueError("arms must be a list")
        return [{"run": v} if isinstance(v, str) else v for v in value]

    @model_validator(mode="after")
    def _consistent(self) -> Contrast:
        labels = [a.name for a in self.arms]
        if len(labels) < 2:
            raise ValueError(f"contrast {self.name!r} needs at least two arms")
        if len(set(labels)) != len(labels):
            raise ValueError(f"contrast {self.name!r} has duplicate arm labels: {labels}")
        if self.baseline is not None and self.baseline not in labels:
            raise ValueError(
                f"contrast {self.name!r}: baseline {self.baseline!r} is not an arm; arms are {labels}"
            )
        return self

    @property
    def labels(self) -> list[str]:
        return [a.name for a in self.arms]

    @property
    def group_map(self) -> dict[str, str]:
        """``{arm label: group}`` for the arms that declare one."""
        return {a.name: a.group for a in self.arms if a.group}

    @property
    def baseline_label(self) -> str:
        return self.baseline or self.labels[0]


class ContrastSet(_Frozen):
    """The file: shared settings and the contrasts."""

    results_root: str = "."
    view: str = "selected"
    #: per-slice columns to test; ``None`` tests every metric both arms of a contrast logged
    metrics: tuple[str, ...] | None = None
    #: ``val`` / ``train`` / ``test`` (inference); the default compares all three
    phases: tuple[str, ...] = PHASES
    contrasts: tuple[Contrast, ...]

    @field_validator("phases")
    @classmethod
    def _known_phases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("phases must name at least one of " + ", ".join(PHASES))
        return check_phases(value)

    @field_validator("view")
    @classmethod
    def _known_view(cls, value: str) -> str:
        if value not in VIEWS:
            raise ValueError(f"unknown checkpoint view {value!r}; choose from {list(VIEWS)}")
        return value

    @model_validator(mode="after")
    def _unique(self) -> ContrastSet:
        if self.metrics is not None and not self.metrics:
            raise ValueError("metrics must name at least one column; omit it to test every metric")
        if not self.contrasts:
            raise ValueError("a contrast set needs at least one contrast")
        names = [c.name for c in self.contrasts]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"duplicate contrast names: {dup}")
        return self

    def run_dir(self, arm: ArmRef, results_root: str | Path | None = None) -> Path:
        path = Path(arm.run)
        return path if path.is_absolute() else Path(results_root or self.results_root) / path


def load_contrast_set(path: str | Path) -> ContrastSet:
    """Parse and validate a contrast file; raises on any unknown or inconsistent key."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return ContrastSet.model_validate(raw)


__all__ = ["ArmRef", "Contrast", "ContrastSet", "Ladder", "load_contrast_set"]
