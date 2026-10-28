"""The one envelope every checkpoint file holds, and its version gate.

Earlier files used two envelopes with different key spellings. Schema 1 replaces both,
and a file of any other version is refused with the converter's name rather than read
by a guess.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

from spectramr import __version__
from spectramr.infrastructure.checkpointing.layout import validate_format

SCHEMA_VERSION = 1
CONVERTER = "scripts/migrations/convert_checkpoints_v1.py"

#: Top-level parts a RunState may carry; ``meta`` and ``models`` are required.
PARTS: tuple[str, ...] = (
    "meta",
    "models",
    "ema",
    "optim",
    "sched",
    "scaler",
    "rng",
    "strategy",
    "early_stopping",
)
REQUIRED_PARTS: tuple[str, ...] = ("meta", "models")
ROLES: tuple[str, ...] = ("step", "best")
#: Why a file was written; a best file's reason is always ``best``.
REASONS: tuple[str, ...] = ("periodic", "final", "loss_schedule", "yield", "best")


#: Top-level keys of the two pre-1.0 envelopes, and the prefixes the service's
#: safetensors files flattened them to. A file holding none of them, under a name no
#: pre-1.0 run used, was not written by spectramr and the converter cannot apply to it.
LEGACY_ENVELOPE_KEYS = frozenset(
    {
        "global_step",
        "optimizer_g",
        "optimizer_d",
        "scaler_state",
        "ema_state",
        "strategy_state",
        "early_stopping_state",
        "rng_state",
        "model_state_dict",
        "optimizer_state_dict",
        "ema_state_dict",
    }
)
LEGACY_FLAT_PREFIXES: tuple[str, ...] = (
    "model_state_dict_",
    "ema_state_dict_",
    "_pickle_",
    "_meta_",
    "optimizer_state_dict__",
    "extra_optimizer_states__",
)
#: A pre-1.0 run named every file ``checkpoint_*``; DeepSpeed's consolidated best was a
#: bare state dict, so for it the name is the only signal.
LEGACY_NAME_PREFIX = "checkpoint_"
DOCS = "docs/checkpointing.rst"


class SchemaVersionError(ValueError):
    """A checkpoint written before schema 1, or by a newer build."""


class NotARunStateError(SchemaVersionError):
    """A weights file no spectramr run wrote: a bare state dict or another tool's file."""


def _has_envelope_keys(keys: Iterable[object]) -> bool:
    names = [k for k in keys if isinstance(k, str)]
    return any(k in LEGACY_ENVELOPE_KEYS or k.startswith(LEGACY_FLAT_PREFIXES) for k in names)


def _names_of(source: str) -> tuple[str, ...]:
    """The file's own name and, for a symlink, the name it finally resolves to."""
    path = Path(source)
    return (path.name, path.resolve().name) if path.is_symlink() else (path.name,)


def refuse_without_meta(source: str, keys: Iterable[object]) -> SchemaVersionError | None:
    """The refusal for a file without ``meta``, or ``None`` when it is a pre-1.0 envelope.

    Envelope keys mean a pre-1.0 spectramr file, which the caller sends to the converter.
    A ``checkpoint_*`` name alone may be DeepSpeed's bare consolidated best or another
    tool's weights, so that refusal names both remedies. Anything else is foreign.
    """
    keys = list(keys)
    if _has_envelope_keys(keys):
        return None
    shown = sorted(str(k) for k in keys)[:5]
    scope = (
        "model.checkpoint_path, a stage's checkpoint_path, the stage-1 VAE and the "
        "diffusion and teacher priors read only checkpoints a spectramr run wrote; "
        f"external weights are out of scope (see {DOCS})."
    )
    # A pre-1.0 run published its best as a ``best.*`` symlink to a legacy-named file, so
    # the target's name decides, not the alias a child arm was handed.
    if any(n.startswith(LEGACY_NAME_PREFIX) for n in _names_of(source)):
        return SchemaVersionError(
            f"{source} has no 'meta' part (first keys {shown}). If a pre-1.0 spectramr run "
            f"wrote it, convert the run first: python {CONVERTER} <run_dir>. Otherwise: "
            f"{scope}"
        )
    return NotARunStateError(
        f"{source} is not a spectramr RunState (no 'meta' part; first keys {shown}). {scope}"
    )


class Topology(TypedDict):
    world_size: int
    parallel_strategy: str


class BestMetric(TypedDict):
    name: str
    value: float


class Meta(TypedDict):
    schema_version: int
    role: str
    reason: str
    epoch: int
    step: int
    format: str
    config_fingerprint: str | None
    topology: Topology
    best_metric: BestMetric | None
    native_tag: str | None
    spectramr_version: str
    git_sha: str | None


class RunState(TypedDict, total=False):
    meta: Meta
    models: dict[str, dict[str, Any]]
    ema: dict[str, Any]
    optim: dict[str, Any]
    sched: dict[str, Any]
    scaler: dict[str, Any]
    rng: dict[str, Any]
    strategy: dict[str, Any]
    early_stopping: dict[str, Any]


def _non_negative(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"checkpoint meta.{name} must be a non-negative int, got {value!r}")
    return value


def build_meta(
    *,
    role: str,
    reason: str,
    epoch: int,
    step: int,
    fmt: str,
    topology: Topology,
    config_fingerprint: str | None = None,
    best_metric: BestMetric | None = None,
    native_tag: str | None = None,
    git_sha: str | None = None,
) -> Meta:
    """Build the ``meta`` part, refusing a combination the reader could misread."""
    if role not in ROLES:
        raise ValueError(f"checkpoint role {role!r} is not one of {list(ROLES)}")
    if reason not in REASONS:
        raise ValueError(f"checkpoint reason {reason!r} is not one of {list(REASONS)}")
    if (role == "best") != (reason == "best"):
        raise ValueError(f"a {role!r} checkpoint cannot have reason {reason!r}")
    if (role == "best") != (best_metric is not None):
        raise ValueError("best_metric is required on a best checkpoint and only there")
    return Meta(
        schema_version=SCHEMA_VERSION,
        role=role,
        reason=reason,
        epoch=_non_negative("epoch", epoch),
        step=_non_negative("step", step),
        format=validate_format(fmt),
        config_fingerprint=config_fingerprint,
        topology=topology,
        best_metric=best_metric,
        native_tag=native_tag,
        spectramr_version=__version__,
        git_sha=git_sha,
    )


def validate(state: Any, *, source: str) -> RunState:
    """Return ``state`` typed as a RunState, or raise naming what is wrong.

    Raises:
        TypeError: ``state`` is not a mapping.
        SchemaVersionError: ``meta.schema_version`` is absent or not ``SCHEMA_VERSION``.
        ValueError: An unknown part, or ``models`` missing or not a mapping of mappings.
    """
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint {source} holds {type(state).__name__}, not a mapping")
    meta = state.get("meta")
    if meta is None and (refusal := refuse_without_meta(source, state)) is not None:
        raise refusal
    version = meta.get("schema_version") if isinstance(meta, Mapping) else None
    if version != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"checkpoint {source} has schema_version {version!r}; this build reads "
            f"{SCHEMA_VERSION}. Convert the run first: python {CONVERTER} <run_dir>"
        )
    unknown = sorted(set(state) - set(PARTS))
    if unknown:
        raise ValueError(f"checkpoint {source} has unknown part(s) {unknown}")
    models = state.get("models")
    if not isinstance(models, Mapping) or not all(isinstance(v, Mapping) for v in models.values()):
        raise ValueError(f"checkpoint {source} needs 'models' as a mapping of state dicts")
    return cast(RunState, state)


__all__ = [
    "CONVERTER",
    "DOCS",
    "LEGACY_ENVELOPE_KEYS",
    "PARTS",
    "REASONS",
    "REQUIRED_PARTS",
    "ROLES",
    "SCHEMA_VERSION",
    "BestMetric",
    "Meta",
    "NotARunStateError",
    "RunState",
    "SchemaVersionError",
    "Topology",
    "build_meta",
    "refuse_without_meta",
    "validate",
]
