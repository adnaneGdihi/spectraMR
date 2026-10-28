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

from collections.abc import Mapping
from typing import Any

__all__ = [
    "WRAPPER_ATTRS",
    "WRAPPER_PREFIXES",
    "is_wrapped",
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
            return inner  # type: ignore[no-any-return]
    return inner  # type: ignore[no-any-return]


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
