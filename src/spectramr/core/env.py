"""Single source of truth for every environment variable spectraMR reads.

The list mirrors `.env.example` at the repo root. Each constant is the name
of the variable; the helper functions are thin wrappers around
``os.environ.get`` / ``os.getenv`` that return parsed defaults so call sites
don't repeat the literal name + default everywhere.

Usage::

    from spectramr.core import env

    data_root = env.data_root()                  # str path, with fallback
    if env.suppress_clinical_warning():
        ...

Adding a new env var requires three lines: a constant in
:mod:`spectramr.core.env_names`, a `.env.example` entry, and a getter here (if it
has a parsed type).

The constant must also be listed in that module's ``__all__`` — :func:`names` is
derived from this module's ``__all__``, which is spliced from the registry's, so
a constant that is declared but not exported is invisible to every consumer
*and* to the tests that check `.env.example` coverage. That is exactly how
``SPECTRAMR_GPU_MEMORY_FRACTION`` stayed unadvertised while its own resolver
claimed it was "registered in core/env.py". The guard needs two tests, not one:
``test_env.py::test_all_constants_are_exported`` scans ``vars(env)`` and so can
only see constants that were successfully re-exported, while
``test_env_names.py::test_all_constants_are_exported`` scans the registry itself
and catches the one that never left it.

Scope note: the registry registers names, this module reads them. Consumers keep
their own ``os.environ.get`` calls and their own defaults; several are read for
bare truthiness rather than through :func:`as_bool` (see ``SPECTRAMR_DEBUG``), so
do not assume a uniform parse just because a name appears here.
"""

from __future__ import annotations

import os
from pathlib import Path

# Every env-var NAME lives in the sibling registry (300-LOC ceiling, NN20) and is
# re-exported wholesale here, so `from spectramr.core import env` stays the single
# import a consumer needs and each name keeps exactly one owner (NN17). The star
# is deliberate: naming all 46 explicitly makes ruff's isort split each aliased
# re-export into its own three-line block, which put this file back over the
# ceiling the split exists to clear.
from spectramr.core.env_names import *  # noqa: F403

# The subset the accessors below actually read, imported explicitly so their uses
# resolve statically rather than through the star (no F405, and a reader can see
# what this module consumes).
from spectramr.core.env_names import (
    FORCE_CPU,
    LOCAL_RANK,
    SPECTRAMR_CACHE_ROOT,
    SPECTRAMR_DATA_ROOT,
    SPECTRAMR_SUPPRESS_CLINICAL_WARNING,
    PROJECT_ROOT,
    RANK,
    WORLD_SIZE,
    XDG_CACHE_HOME,
)
from spectramr.core.env_names import __all__ as _constant_names

#: Spliced from the registry rather than restated, so the two cannot drift.
#: :func:`names` filters this for upper-case entries exactly as before.
__all__ = [
    *_constant_names,
    "data_root",
    "project_root",
    "cache_root",
    "suppress_clinical_warning",
    "force_cpu",
    "is_distributed",
    "distributed_rank",
    "is_secondary_rank",
    "as_bool",
    "as_int",
    "names",
]


def names() -> tuple[str, ...]:
    """Return every framework env-var name in declaration order.

    Intended for anything that wants to echo the current resolved environment.
    Its one consumer in the tree is ``scripts/release/print_env.py`` (verified
    2026-08-16, repo-wide) -- in particular **not** ``spectramr diagnostics``,
    which iterates its own hand-maintained ``_ENV_KNOBS`` tuple. Registering a
    name here therefore does not make it show up there; the two lists are
    unconnected, and this docstring previously implied otherwise.

    Derived from ``__all__``, so a constant left out of it is not returned --
    see the module docstring.
    """
    return tuple(n for n in __all__ if n.isupper())


def as_bool(var: str, default: bool = False) -> bool:
    """Parse an env var as a boolean (1/true/yes/on are True; everything else False).

    Missing → ``default``.
    """
    v = os.environ.get(var)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def as_int(var: str, default: int) -> int:
    v = os.environ.get(var)
    if v is None or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Parsed-default helpers — the names mirror the CLAUDE.md / .env.example terms
# so call sites read naturally: ``env.data_root()`` not
# ``os.environ.get("SPECTRAMR_DATA_ROOT", "./databases")``.
# ─────────────────────────────────────────────────────────────────────────────


def data_root() -> Path:
    """The configured data root, with the documented fallback to ./databases."""
    return Path(os.environ.get(SPECTRAMR_DATA_ROOT, "./databases"))


def project_root() -> Path | None:
    """Return PROJECT_ROOT / SPECTRAMR_DATA_ROOT or None if neither is set."""
    v = os.environ.get(PROJECT_ROOT) or os.environ.get(SPECTRAMR_DATA_ROOT)
    return Path(v) if v else None


def cache_root() -> Path:
    """Cache root — honours SPECTRAMR_CACHE_ROOT, then XDG_CACHE_HOME, then ~/.cache."""
    explicit = os.environ.get(SPECTRAMR_CACHE_ROOT)
    if explicit:
        return Path(explicit)
    xdg = os.environ.get(XDG_CACHE_HOME)
    if xdg:
        return Path(xdg) / "spectramr"
    return Path.home() / ".cache" / "spectramr"


def suppress_clinical_warning() -> bool:
    """True if the import-time clinical warning should be silenced."""
    return as_bool(SPECTRAMR_SUPPRESS_CLINICAL_WARNING)


def force_cpu() -> bool:
    """True if FORCE_CPU is set — caller should avoid CUDA paths."""
    return as_bool(FORCE_CPU)


def is_distributed() -> bool:
    """True if torchrun / a multi-container launcher set RANK / WORLD_SIZE."""
    return WORLD_SIZE in os.environ and as_int(WORLD_SIZE, 1) > 1


def distributed_rank() -> int | None:
    """This process's global rank, or ``None`` when not launched distributed.

    Reads the environment torchrun sets *before* the process starts, so it
    answers long before ``dist.init_process_group`` runs — which is the whole
    point: the four startup lines that used to print once per rank are all
    emitted before the process group exists.

    ``WORLD_SIZE`` must be present: torchrun always sets it with ``RANK``, and requiring it
    avoids mistaking an unrelated environment that happens to export ``RANK`` for a
    distributed launch. The rank itself comes from :func:`spectramr.core.topology.resolve_rank`,
    the one rule ``setup_distributed`` and the run topology also use, so the console gate cannot
    disagree with the process group about which process is rank 0.
    """
    if WORLD_SIZE not in os.environ:
        return None
    # Imported here, not at the top: this module is loaded to build the CLI parser, and the
    # topology module pulls in the resource probes.
    from spectramr.core.topology import resolve_rank

    rank, _source = resolve_rank(os.environ)
    return rank


def is_secondary_rank() -> bool:
    """True only on a non-zero rank of an actual distributed launch.

    False on a single-process run (rank is ``None``) and False on rank 0, so a
    caller can gate console output on it without changing single-process
    behaviour at all.
    """
    rank = distributed_rank()
    return rank is not None and rank != 0
