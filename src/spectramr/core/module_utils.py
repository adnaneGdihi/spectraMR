"""Wrapper-unwrapping SSOT for ``nn.Module`` and its state dicts.

Training wraps the model up to four times, and every wrapper renames the keys
``state_dict()`` emits:

===================================  =========================================
wrapper                              key prefix it adds
===================================  =========================================
``torch.compile``                    ``_orig_mod.``
``ModelEma``                         ``module.``
``DataParallel`` / ``DDP``           ``module.``
FSDP                                 ``_fsdp_wrapped_module.``
activation checkpointing             ``_checkpoint_wrapped_module.``
===================================  =========================================

The checkpoint writers called ``model.state_dict()`` with no unwrapping, so a
``compile_model: true`` or DDP run wrote prefixed keys that no inference path
could read back: every ``infer`` / ``predict`` / evaluation path builds a *bare*
model and loads with ``strict=True``. Under ``strict=False`` the failure is worse
than an exception — nothing matches, nothing loads, and the load reports success.

Meanwhile 26 call sites had each grown their own inline unwrap. Only two of them
handled ``_orig_mod``, none handled FSDP, and ``pipelines.parallel.unwrap_model``
— the one that looked canonical — handled only DP/DDP and had **zero callers**.

Two entry points, because unwrapping is needed at two different layers:

* :func:`unwrap_model` — object level, before reading ``state_dict()``.
* :func:`strip_wrapper_prefixes` — key level, for reading a checkpoint that was
  written *before* this module existed.

**Layer note.** This lives in ``core/`` (the rightmost layer) rather than beside
the distributed code, because ``pipelines/``, ``infrastructure/``, ``models/``
and ``cli/`` all need it and the dependency rule points inward only —
``infrastructure/`` may not import from ``pipelines/``, which is where the old
helper sat.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CHECKPOINT_STATE_KEYS",
    "WRAPPER_ATTRS",
    "WRAPPER_PREFIXES",
    "is_wrapped",
    "resolve_state_dict",
    "strip_wrapper_prefixes",
    "unwrap_model",
]

#: Attributes a wrapper stores its wrapped module under, in the order an
#: unwrapping loop should try them. ``module`` is last: ``DataParallel``, ``DDP``
#: and ``ModelEma`` all use it, and a compiled-then-DDP'd model nests as
#: ``DDP(OptimizedModule)``, so peeling the specific names first keeps the walk
#: deterministic regardless of wrap order.
WRAPPER_ATTRS: tuple[str, ...] = (
    "_orig_mod",  # torch.compile -> OptimizedModule
    "_fsdp_wrapped_module",  # FullyShardedDataParallel
    "_checkpoint_wrapped_module",  # activation checkpointing
    "module",  # DataParallel / DistributedDataParallel / ModelEma
)

#: The state-dict key prefixes the wrappers in :data:`WRAPPER_ATTRS` introduce.
WRAPPER_PREFIXES: tuple[str, ...] = tuple(f"{attr}." for attr in WRAPPER_ATTRS)

#: Wrapper attributes safe to strip at ANY depth, not just leading.
#:
#: Regional compilation wraps individual children of an ``nn.ModuleList``,
#: which puts the marker mid-path -- ``blocks.0._orig_mod.conv.weight``. A
#: leading-only strip leaves that in the checkpoint, where every inference
#: path builds a bare model and loads with ``strict=True``.
#:
#: ``module`` is deliberately NOT here. It is an ordinary attribute name a
#: model may legitimately use for a real submodule (``encoder.module.weight``),
#: so stripping it mid-path would delete a user's own structure. The three
#: below are framework-synthesised and underscore-prefixed; no model names a
#: submodule ``_orig_mod``.
_DEPTH_SAFE_WRAPPER_ATTRS: frozenset[str] = frozenset(
    {"_orig_mod", "_fsdp_wrapped_module", "_checkpoint_wrapped_module"}
)

#: Depth cap for the unwrap walk. Four wrappers is the realistic maximum
#: (compile + FSDP + checkpointing + DDP); the cap exists so a module that
#: exposes a self-referential ``.module`` cannot spin forever.
_MAX_UNWRAP_DEPTH = 10


def unwrap_model[T](model: T) -> T:
    """Return the innermost module beneath any stack of training wrappers.

    Idempotent, and a no-op on a bare module — safe to call unconditionally at a
    save site rather than guarding it with an ``isinstance`` chain that has to
    enumerate every wrapper type.

    Deliberately duck-typed rather than ``isinstance``-based: ``FSDP`` and
    ``OptimizedModule`` cannot be imported on a torch-less/CPU-shimmed
    environment, and this module is imported by the config-adjacent layers.
    """
    inner: Any = model
    for _ in range(_MAX_UNWRAP_DEPTH):
        for attr in WRAPPER_ATTRS:
            nxt = getattr(inner, attr, None)
            # `is not inner` guards the self-referential case; the None check
            # keeps a wrapper that declares the attribute but leaves it unset
            # from collapsing the model to None.
            if nxt is not None and nxt is not inner:
                inner = nxt
                break
        else:
            return inner  # type: ignore[return-value]
    return inner  # type: ignore[return-value]


def is_wrapped(model: Any) -> bool:
    """True when ``model`` sits behind at least one training wrapper."""
    return unwrap_model(model) is not model


def _strip_key(key: str) -> str:
    """Drop synthetic wrapper segments at any depth, then leading ``module.``.

    Two rules because the names carry different risk. The underscore-prefixed
    wrappers cannot collide with a real attribute, so they go wherever they
    appear. ``module`` can, so it is only removed while it leads -- which is the
    only position DDP, DataParallel and ModelEma ever put it in.
    """
    name = ".".join(
        part for part in key.split(".") if part not in _DEPTH_SAFE_WRAPPER_ATTRS
    )
    while True:
        for prefix in WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        else:
            return name


def strip_wrapper_prefixes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Remove every wrapper segment from a state dict's keys, at any depth.

    For reading checkpoints written before the save sites unwrapped — including
    the doubly-prefixed ``module._orig_mod.conv.weight`` a compiled-then-EMA'd
    model produced.

    **Two rules, because the names carry different risk.** Regional compilation
    wraps individual children of an ``nn.ModuleList``, putting the marker in the
    *middle* of the path::

        blocks.0._orig_mod.conv.weight

    A leading-only strip leaves that in the checkpoint, and every inference path
    builds a bare model and loads with ``strict=True`` — or under
    ``strict=False`` matches nothing, loads nothing, and reports success. So the
    synthetic wrappers (:data:`_DEPTH_SAFE_WRAPPER_ATTRS`) are dropped wherever
    they appear.

    ``module`` is not, because it is an ordinary attribute name: a model may own
    a real submodule called ``module``, and ``encoder.module.weight`` must
    survive. It is removed only while it leads, which is the only position DDP,
    DataParallel and ModelEma ever put it in.

    Returns a plain ``dict`` (never the input object) so callers cannot mutate a
    live module's state dict by accident.

    Raises:
        ValueError: if stripping would collide two distinct keys onto one name.
            That means the checkpoint holds two different modules' weights, and
            silently keeping whichever came last would load a hybrid of the two.
    """
    stripped: dict[str, Any] = {}
    # Keyed by stripped name -> the original key it came from, so a collision is
    # detected whichever order the two arrive in. The previous guard compared
    # `key != name`, which asked "was THIS key transformed" rather than "did two
    # keys land on one name": `{"module.w", "w"}` raised in one order and
    # silently kept the last value in the other, and a state dict's order is
    # just module-traversal order.
    sources: dict[str, str] = {}
    for key, value in state_dict.items():
        name = _strip_key(key)
        if name in sources:
            raise ValueError(
                f"stripping wrapper prefixes collides {key!r} onto {name!r}, "
                f"already claimed by {sources[name]!r}. The checkpoint holds two "
                "distinct modules under names that differ only by a wrapper "
                "segment; load them separately rather than merging them."
            )
        sources[name] = key
        stripped[name] = value
    return stripped


#: Container keys a checkpoint writer may wrap a parameter state dict under.
#: Two writers disagree -- ``CheckpointDirector`` emits ``generator`` while
#: ``CheckpointService`` emits ``model_state_dict`` -- and the readers each grew
#: around whichever one they happened to meet, which is the whole reason this
#: list exists. ``generator_state_dict`` is read by the GAN inference strategies
#: and kept so consolidating onto this SSOT loses none of their reach.
#:
#: ``ema_state_dict`` is deliberately NOT here: the EMA copy is a *different
#: model*, not another spelling of this one, and silently preferring it would
#: swap the weights under the caller.
CHECKPOINT_STATE_KEYS: tuple[str, ...] = (
    "model",
    "state_dict",
    "generator",
    "model_state_dict",
    "generator_state_dict",
)


def resolve_state_dict(
    payload: Any,
    model_keys: Collection[str],
    *,
    source: str,
) -> dict[str, Any]:
    """Resolve a loaded checkpoint payload to a state dict matching ``model_keys``.

    Three failures are folded into one place, because every checkpoint reader in
    the repo has to solve all three and each solved a different subset:

    1. **The envelope.** Writers wrap the parameters under a container key, and
       the two writers here disagree (see :data:`CHECKPOINT_STATE_KEYS`). A
       reader that knows only its own spelling hands the *whole envelope* to
       ``load_state_dict``.
    2. **The prefixes.** :func:`strip_wrapper_prefixes`, applied after unwrapping,
       so a compiled / DDP / FSDP checkpoint reads back into a bare model.
    3. **The silent miss.** Zero key overlap raises *here* rather than reaching a
       ``load_state_dict(strict=False)`` that loads nothing and reports success
       (CLAUDE.md pitfalls #9/#16).

    Candidates are chosen by **overlap, not by order**: every recognised
    container plus the payload itself is tried, and the first one sharing a
    parameter name with ``model_keys`` wins. Ranking by list order instead would
    pick the config out of a ``{"model": <config>, "generator": <weights>}``
    payload -- ``"model"`` is a container key *and* a common config section
    name, and both are mappings, so the two cannot be told apart from outside.

    Strictness is left to the caller: warm-start transfer wants ``strict=False``
    (a partially-overlapping architecture still shares its trunk), inference
    wants ``strict=True``. The overlap check is what makes ``strict=False`` safe.

    Args:
        payload: Whatever ``torch.load`` returned.
        model_keys: ``model.state_dict().keys()`` for the destination model.
        source: Path or description of the checkpoint, used in error messages.

    Returns:
        Plain dict of parameter name -> value, ready for ``load_state_dict``.

    Raises:
        TypeError: If ``payload`` is not a mapping at all.
        ValueError: If no candidate shares a parameter name with ``model_keys``.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"checkpoint {source!r} holds {type(payload).__name__}, not a "
            "mapping; expected a state dict, or a dict wrapping one under one "
            f"of {list(CHECKPOINT_STATE_KEYS)}."
        )

    candidates: list[tuple[str | None, Mapping[str, Any]]] = [
        (key, inner)
        for key in CHECKPOINT_STATE_KEYS
        if isinstance(inner := payload.get(key), Mapping)
    ]
    candidates.append((None, payload))  # the payload may itself be the state dict

    wanted = set(model_keys)
    collision: ValueError | None = None
    for envelope, candidate in candidates:
        try:
            resolved = strip_wrapper_prefixes(candidate)
        except ValueError as exc:
            # Two modules under names differing only by a wrapper prefix. That
            # disqualifies this candidate, not the whole payload -- but keep it
            # so a total failure reports the real reason, not "no overlap".
            collision = collision or exc
            continue
        if wanted & set(resolved):
            if envelope is not None:
                logger.debug("checkpoint %s: parameters found under %r", source, envelope)
            return resolved

    if collision is not None:
        raise collision
    tried = [k for k, _ in candidates if k is not None] or ["<none>"]
    found = ", ".join(sorted(map(str, payload))[:6]) or "<empty>"
    expected = ", ".join(sorted(wanted)[:3]) or "<none>"
    raise ValueError(
        f"checkpoint {source!r} shares zero parameter names with the model. "
        f"Tried containers {tried} and the payload itself; payload keys e.g. "
        f"[{found}]; the model expects e.g. [{expected}]. Recognised container "
        f"keys are {list(CHECKPOINT_STATE_KEYS)} -- refusing to load nothing "
        "and report success."
    )
