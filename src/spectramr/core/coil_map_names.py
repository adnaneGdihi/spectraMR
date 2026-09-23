"""The spellings per-coil sensitivity maps travel under, and the one place they reconcile.

There are five, and they are all live. The data layer produces ``sensitivity``
(``torchio_subject_builder``); consumers ask for ``sensitivity_maps`` (10 files),
``coil_sensitivities`` (5), ``smaps`` (3) and ``coil_maps`` (1). Renaming 19
consumer files was rejected once already (audit C11) — the table is cheaper than
the churn, but only if there is exactly one of it.

It lives in ``core/`` rather than beside either consumer because it has two
boundaries to serve and they sit in different layers. ``data.batch_types``
reconciles the spellings when a batch crosses into the strategy;
``kwargs_accepted_by`` reconciles them again wherever a loss's kwargs are narrowed
to its signature, because a loss naming ``coil_sensitivities`` never sees a
caller's ``smaps`` through a filter on the CALLEE's parameter names. ``models/``
cannot import ``data/``, so a single table in ``data/`` could not serve both and
the second boundary was reconciled by passing the same tensor twice.

``coil_maps`` is canonical: ``sensitivity`` alone is ambiguous with the physics
sense of the word.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Container, Mapping
from typing import Any

__all__ = [
    "CANONICAL_COIL_MAP_KEY",
    "COIL_MAP_ALIASES",
    "coil_map_kwargs_for",
    "kwargs_accepted_by",
]

#: Every spelling the codebase uses for per-coil sensitivity maps. Ordered by
#: preference, so a caller holding several resolves deterministically.
COIL_MAP_ALIASES: tuple[str, ...] = (
    "coil_maps",
    "sensitivity",
    "sensitivity_maps",
    "coil_sensitivities",
    "smaps",
)

#: The one the rest of the framework normalises onto.
CANONICAL_COIL_MAP_KEY = COIL_MAP_ALIASES[0]


def coil_map_kwargs_for(
    accepted: Container[str],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """The coil-map entries ``accepted`` names that ``kwargs`` spells differently.

    Returns ``{}`` unless the caller actually holds maps under some alias AND the
    callee declares a *different* one — so a callee that names the same alias, or
    names none, is untouched and the caller's own value always wins over a
    re-filing of it.

    Args:
        accepted: the parameter names the callee declares. Pass the explicit
            names only, never a ``**kwargs``-expanded universe: a callee with
            ``**kwargs`` would otherwise be handed five copies of one tensor.
        kwargs: what the caller holds.

    Why a helper rather than "just pass every alias": the caller would then have
    to know the table, five keys would reach every ``**kwargs`` loss in the
    framework, and the reconciliation would live at each call site instead of
    once (non-negotiable 17).
    """
    held = next((k for k in COIL_MAP_ALIASES if kwargs.get(k) is not None), None)
    if held is None:
        return {}
    return {
        alias: kwargs[held]
        for alias in COIL_MAP_ALIASES
        if alias in accepted and kwargs.get(alias) is None
    }


def kwargs_accepted_by(callee: Callable[..., Any], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The part of ``kwargs`` that ``callee`` receives, with the coil maps reconciled.

    Every entry the callee's signature names (all of them, if it takes
    ``**kwargs``), plus the coil maps re-filed under whichever alias it declares.
    This is the one filter for every hop that narrows a loss's kwargs to its
    signature. A hop that filters without reconciling drops the maps for any term
    spelled differently from its caller, and a transparent wrapper in front of it
    cannot help, because the wrapper's own ``**kwargs`` names no alias.

    A module is read through ``forward``: its ``__call__`` is
    ``(*args, **kwargs)`` and would admit everything.

    Args:
        callee: the loss, or its ``forward``.
        kwargs: what the caller holds.
    """
    params = inspect.signature(getattr(callee, "forward", callee)).parameters
    takes_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    accepted = {k: v for k, v in kwargs.items() if takes_varkw or k in params}
    accepted.update(coil_map_kwargs_for(params, kwargs))
    return accepted
