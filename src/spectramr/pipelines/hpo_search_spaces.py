"""Search-space specifications for HPO over arbitrary YAML / schema fields.

The point of this module is to bridge the gap between
``EnhancedHPOptimizer`` (which calls ``objective_fn(trial)`` per trial) and
``subprocess_training_objective`` (which takes an ``overrides_fn`` that
maps each trial to a dict of ``{dotted_path: value}`` overrides). Without
the bridge, every Optuna trial runs the base config unchanged and the
HPO is a no-op.

Three layers of usage, increasingly flexible:

1. **Preset name** — pick a canned search space from ``PRESETS`` by name.
   Examples: ``"kan_dual_domain"`` (the 14-dim plan §4.2 space),
   ``"loss_weights_only"``, ``"optimizer_basic"``.

2. **YAML file** — write your own spec as a YAML file mapping dotted-path
   to a distribution. See ``SearchSpace.from_yaml()`` for the schema and
   ``make_search_space_template()`` to generate a template you can edit.

3. **Programmatic dict** — pass a Python dict directly to
   ``SearchSpace.from_dict()`` if you're building HPO from inside a
   Python script (e.g. inside a notebook or a unit test).

Schema-aware enumeration
------------------------
``enumerate_schema_paths()`` walks the loaded ``TrainingSettings`` model
and returns every dotted path that's legal to override. Use this to
discover what's tunable without guessing — the CLI exposes it via
``spectramr hpo --list-schema-paths``.

Distributions supported
-----------------------
* ``uniform(low, high)``           — continuous in [low, high]
* ``loguniform(low, high)``        — log-spaced continuous in [low, high]
* ``int_uniform(low, high)``       — integer in [low, high]
* ``int_loguniform(low, high)``    — log-spaced integer in [low, high]
* ``categorical(choices)``         — pick one of the given choices

The set is deliberately small — these cover ~95% of MRI HPO needs and
keep the spec YAML readable. Add Pareto / discretized-uniform / etc.
behind the same ``Distribution`` interface if a real workload needs it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distribution specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Distribution:
    """One hyperparameter's distribution spec.

    Attributes:
        kind: One of ``uniform``, ``loguniform``, ``int_uniform``,
            ``int_loguniform``, ``categorical``.
        low / high: Bounds (inclusive of low). Required for non-categorical.
        choices: Required for categorical. List of values to pick from.
    """

    kind: str
    low: float | int | None = None
    high: float | int | None = None
    choices: list[Any] | None = None

    def __post_init__(self) -> None:
        valid_kinds = {"uniform", "loguniform", "int_uniform", "int_loguniform", "categorical"}
        if self.kind not in valid_kinds:
            raise ValueError(
                f"Distribution.kind must be one of {sorted(valid_kinds)}; got {self.kind!r}"
            )
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError("categorical distribution requires non-empty choices")
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"{self.kind} distribution requires both low and high")
            if self.low >= self.high:
                raise ValueError(f"low ({self.low}) must be < high ({self.high}) for {self.kind}")
            if "log" in self.kind and self.low <= 0:
                raise ValueError(f"loguniform distributions require low > 0; got {self.low}")

    def sample(self, trial: optuna.trial.Trial, name: str) -> Any:  # noqa: F821
        """Sample one value from this distribution via the given Optuna trial."""
        if self.kind == "uniform":
            return trial.suggest_float(name, float(self.low), float(self.high))
        if self.kind == "loguniform":
            return trial.suggest_float(name, float(self.low), float(self.high), log=True)
        if self.kind == "int_uniform":
            return trial.suggest_int(name, int(self.low), int(self.high))
        if self.kind == "int_loguniform":
            return trial.suggest_int(name, int(self.low), int(self.high), log=True)
        if self.kind == "categorical":
            return trial.suggest_categorical(name, self.choices)
        raise ValueError(f"Unknown distribution kind: {self.kind}")


# ---------------------------------------------------------------------------
# SearchSpace container
# ---------------------------------------------------------------------------


@dataclass
class SearchSpace:
    """An HPO search space: dotted-path → distribution.

    Attributes:
        params: Mapping from dotted-path config key (e.g.
            ``"optimization.optimizer.learning_rate"``) to its ``Distribution``.
            Order is preserved for reproducible suggestion ordering.
    """

    params: dict[str, Distribution] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, dict | tuple]) -> SearchSpace:
        """Build from a dict spec.

        Each value may be either a dict ``{"dist": ..., "low": ..., ...}`` or
        a tuple ``(kind, low, high)`` / ``("categorical", *choices)`` for
        terseness when writing inline Python.
        """
        params: dict[str, Distribution] = {}
        for path, spec in raw.items():
            if isinstance(spec, Distribution):
                params[path] = spec
            elif isinstance(spec, dict):
                kind = spec.get("dist") or spec.get("kind")
                if not kind:
                    raise ValueError(f"path {path!r}: dict spec requires a 'dist' or 'kind' key")
                params[path] = Distribution(
                    kind=kind,
                    low=spec.get("low"),
                    high=spec.get("high"),
                    choices=spec.get("choices"),
                )
            elif isinstance(spec, (list, tuple)):
                if not spec:
                    raise ValueError(f"path {path!r}: tuple spec is empty")
                kind = spec[0]
                if kind == "categorical":
                    choices = list(spec[1:])
                    params[path] = Distribution(kind="categorical", choices=choices)
                else:
                    if len(spec) != 3:
                        raise ValueError(
                            f"path {path!r}: tuple spec for {kind} must be "
                            f"(kind, low, high), got {spec}"
                        )
                    params[path] = Distribution(kind=kind, low=spec[1], high=spec[2])
            else:
                raise ValueError(f"path {path!r}: unsupported spec type {type(spec).__name__}")
        return cls(params=params)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> SearchSpace:
        """Load a search space from a YAML file."""
        with open(yaml_path) as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Search-space YAML must be a dict at the top level; got {type(raw).__name__}"
            )
        return cls.from_dict(raw)

    def make_overrides_fn(self) -> Callable[[optuna.trial.Trial], dict[str, Any]]:  # noqa: F821
        """Return an Optuna-compatible callable for ``subprocess_training_objective``.

        The returned function calls ``Distribution.sample(trial, path)`` for
        each spec param and assembles the result as a flat dotted-path dict
        suitable for ``_build_trial_yaml``.
        """
        params = self.params

        def _overrides_fn(trial: optuna.trial.Trial) -> dict[str, Any]:  # noqa: F821
            return {path: dist.sample(trial, path) for path, dist in params.items()}

        return _overrides_fn

    def __len__(self) -> int:
        return len(self.params)

    def __contains__(self, path: str) -> bool:
        return path in self.params

    def merge(self, other: SearchSpace, *, on_conflict: str = "error") -> SearchSpace:
        """Combine two search spaces.

        Args:
            other: The space to merge into this one.
            on_conflict: How to handle keys that appear in both spaces:

                * ``error`` (default) — raise ``ValueError``. Safest for
                  scripts where overlap is almost always a bug.
                * ``override`` — ``other`` wins. Useful when ``self`` is
                  a baseline preset and ``other`` is a user override.
                * ``keep`` — ``self`` wins.

        Returns:
            A new ``SearchSpace`` (neither input is mutated).
        """
        if on_conflict not in ("error", "override", "keep"):
            raise ValueError(
                f"on_conflict must be 'error', 'override', or 'keep'; got {on_conflict!r}"
            )
        merged = dict(self.params)
        for path, dist in other.params.items():
            if path in merged:
                if on_conflict == "error":
                    raise ValueError(
                        f"merge conflict on path {path!r}: present in both "
                        f"spaces. Pass on_conflict='override' or 'keep' "
                        f"to resolve explicitly."
                    )
                if on_conflict == "override":
                    merged[path] = dist
                # 'keep' leaves merged[path] alone
            else:
                merged[path] = dist
        return SearchSpace(params=merged)


def load_presets(names: list[str], on_conflict: str = "error") -> SearchSpace:
    """Load and merge multiple presets in one call.

    The CLI exposes this via ``--search-preset NAME`` being repeatable;
    use this helper to do the same thing programmatically. Conflicts
    raise by default — pass ``on_conflict='override'`` if you intend
    later presets to win on overlapping paths.
    """
    if not names:
        return SearchSpace(params={})
    space = load_preset(names[0])
    for name in names[1:]:
        space = space.merge(load_preset(name), on_conflict=on_conflict)
    return space


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------


PRESETS: dict[str, dict[str, Any]] = {
    # Plan §4.2's search space for the KAN dual-domain headline, less the
    # log_spectral weight: the cohort no longer trains that term.
    "kan_dual_domain": {
        # Loss weights (log-uniform; complex_l1 anchor stays at 1.0)
        "losses.kspace_losses[name=sobolev_kspace].weight": ("loguniform", 0.005, 0.5),
        "losses.kspace_losses[name=complex_spatial_gradient].weight": ("loguniform", 0.05, 1.0),
        "losses.kspace_losses[name=sense_adjoint_l1].weight": ("loguniform", 0.05, 1.0),
        # KAN configuration
        "model.model_kwargs.kan_dual_domain_kwargs.kan_grid_size": ("categorical", 4, 5, 6, 8),
        "model.model_kwargs.kan_dual_domain_kwargs.kan_spline_order": ("categorical", 2, 3, 4),
        "model.model_kwargs.kan_dual_domain_kwargs.kan_hidden": ("categorical", 8, 16, 24),
        # Radial-band attention
        "model.model_kwargs.kan_dual_domain_kwargs.num_bands": ("categorical", 4, 6, 8, 12),
        "model.model_kwargs.kan_dual_domain_kwargs.radial_beta": ("uniform", 1.0, 4.0),
        # Optimizer
        "optimization.optimizer.learning_rate": ("loguniform", 1e-5, 2e-4),
        "model.model_kwargs.kan_lr_ratio": ("loguniform", 0.05, 0.5),
        # Curriculum
        "training.curriculum_ramp_rate": ("loguniform", 1e-3, 2e-2),
    },
    # Loss-weights-only — useful for quickly tuning a fixed architecture.
    "loss_weights_only": {
        "losses.kspace_losses[name=sobolev_kspace].weight": ("loguniform", 0.005, 0.5),
        "losses.kspace_losses[name=complex_spatial_gradient].weight": ("loguniform", 0.05, 1.0),
        "losses.kspace_losses[name=sense_adjoint_l1].weight": ("loguniform", 0.05, 1.0),
    },
    # Optimizer-basic — LR + weight decay sweep, no architecture changes.
    "optimizer_basic": {
        "optimization.optimizer.learning_rate": ("loguniform", 1e-5, 2e-3),
        "optimization.optimizer.weight_decay": ("loguniform", 1e-7, 1e-3),
    },
    # Full KAN block — every KAN-specific hyperparameter exposed by the
    # composite block. Useful after picking a winning architectural variant.
    "full_kan_block": {
        "model.model_kwargs.kan_dual_domain_kwargs.kan_grid_size": ("categorical", 4, 5, 6, 8),
        "model.model_kwargs.kan_dual_domain_kwargs.kan_spline_order": ("categorical", 2, 3, 4),
        "model.model_kwargs.kan_dual_domain_kwargs.kan_hidden": ("categorical", 8, 16, 24, 32),
        "model.model_kwargs.kan_dual_domain_kwargs.num_bands": ("categorical", 4, 6, 8, 12, 16),
        "model.model_kwargs.kan_dual_domain_kwargs.radial_beta": ("uniform", 1.0, 4.0),
        "model.model_kwargs.kan_dual_domain_kwargs.num_heads": ("categorical", 2, 4, 8),
        "model.model_kwargs.kan_lr_ratio": ("loguniform", 0.01, 1.0),
        "model.model_kwargs.kan_weight_decay_ratio": ("loguniform", 0.01, 1.0),
    },
    # Diffusion schedule — for tuning curriculum and noise schedule decisions.
    "diffusion_curriculum": {
        "training.curriculum_ramp_rate": ("loguniform", 1e-3, 5e-2),
        "training.curriculum_start_timestep": ("int_uniform", 50, 500),
        "training.identity_collapse_threshold": ("loguniform", 1e-5, 1e-2),
    },
}


def list_presets() -> list[str]:
    """Return the names of all built-in search-space presets."""
    return sorted(PRESETS.keys())


def load_preset(name: str) -> SearchSpace:
    """Return the named built-in preset as a ``SearchSpace``."""
    if name not in PRESETS:
        raise KeyError(f"Unknown preset: {name!r}. Available: {list_presets()}")
    return SearchSpace.from_dict(PRESETS[name])


# ---------------------------------------------------------------------------
# Schema-aware path enumeration (Pydantic introspection)
# ---------------------------------------------------------------------------


def enumerate_schema_paths(
    model_cls: type | None = None,
    prefix: str = "",
    max_depth: int = 6,
) -> list[str]:
    """Walk a Pydantic model and return every dotted-path field.

    Use this to discover what HPO can target without manually inspecting
    the schema YAML reference. Only descends into nested Pydantic models;
    free-form dict fields (like ``model.model_kwargs``) are returned as
    leaves with a trailing ``.*`` so the caller knows they're open-ended.

    Args:
        model_cls: Pydantic model class to introspect. Defaults to
            ``TrainingSettings``.
        prefix: Internal recursion accumulator; pass empty string at top.
        max_depth: Maximum nesting depth to recurse to (safety bound).

    Returns:
        Sorted list of dotted-path strings.
    """
    if model_cls is None:
        from spectramr.config.settings import TrainingSettings

        model_cls = TrainingSettings

    if max_depth <= 0:
        return [f"{prefix}.*"] if prefix else ["*"]

    paths: list[str] = []
    if not hasattr(model_cls, "model_fields"):
        # Not a Pydantic model — terminal leaf
        return [prefix] if prefix else []

    import types

    for field_name, field_info in model_cls.model_fields.items():
        full = f"{prefix}.{field_name}" if prefix else field_name
        annotation = field_info.annotation
        # Strip Optional / Union by extracting the non-None type. Handle
        # both ``typing.Union[X, None]`` (origin=Union, has __args__) and
        # PEP 604 ``X | None`` (types.UnionType — no __origin__ but does
        # have __args__).
        is_union = (
            getattr(annotation, "__origin__", None) is not None and hasattr(annotation, "__args__")
        ) or isinstance(annotation, types.UnionType)
        if is_union and hasattr(annotation, "__args__"):
            non_none = [a for a in annotation.__args__ if a is not type(None)]
            if len(non_none) == 1:
                annotation = non_none[0]
            elif len(non_none) > 1:
                # For Union[A, B] pick the first branch that is a Pydantic model
                # so nested fields are enumerated correctly (e.g. losses sub-schemas).
                for candidate in non_none:
                    if hasattr(candidate, "model_fields"):
                        annotation = candidate
                        break
        origin = getattr(annotation, "__origin__", None)
        # Recurse if the field type is itself a Pydantic model
        if hasattr(annotation, "model_fields"):
            paths.extend(enumerate_schema_paths(annotation, prefix=full, max_depth=max_depth - 1))
        elif annotation is dict or (origin is dict):
            # Free-form dict — caller can target arbitrary sub-keys
            paths.append(f"{full}.*")
        else:
            paths.append(full)
    return sorted(paths)


# ---------------------------------------------------------------------------
# Path-application: handles plain dotted paths AND list-element selectors
#
# Implementation moved to ``spectramr.infrastructure.hpo.dotted_override`` per
# ``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 2 — the HPO coordinator
# (infrastructure/) was importing this upward from pipelines/. We keep the
# re-export here for legacy callers.
# ---------------------------------------------------------------------------


def make_search_space_template(
    paths: list[str] | None = None,
    default_dist: str = "loguniform",
    default_low: float = 1e-5,
    default_high: float = 1e-3,
) -> str:
    """Produce a YAML template string the user can edit for ``--search-space``.

    Args:
        paths: List of dotted paths to include. Defaults to a curated
            subset of TrainingSettings (LR, WD, common loss weights).
        default_dist / default_low / default_high: Default distribution
            spec applied to every path.

    Returns:
        YAML string suitable for writing to a file.
    """
    if paths is None:
        paths = [
            "optimization.optimizer.learning_rate",
            "optimization.optimizer.weight_decay",
            "training.curriculum_ramp_rate",
        ]
    block: dict[str, dict[str, Any]] = {}
    for p in paths:
        block[p] = {"dist": default_dist, "low": default_low, "high": default_high}
    return yaml.safe_dump(block, sort_keys=False, default_flow_style=False)
