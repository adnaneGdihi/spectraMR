"""Unified registry for meta-evaluation rankers (all generations).

One registry, one contract. Every ranker — the Gen-1 statistical rankers, the
Gen-2 information/sensitivity rankers, the Gen-3 structural rankers, and the
Gen-4 clinical/causal rankers — registers here under a single
:class:`~spectramr.core.metrics.meta_evaluation.rankers.base.BaseRanker` contract
(``rank(metric_set, dataset) -> RankingResult``). The ``generation`` field is
*metadata* (lineage), not a pipeline fork: there is no longer a "legacy" vs
"novel" split — the unified :class:`RankingPipeline` resolves every registered
ranker whose data dependencies are satisfied and runs them together.

This is the canonical home (it moved inward from ``scripts/sim2rank`` so the
core layer owns ranker discovery — pitfall #12 / #13). It mirrors the strict
registry-dispatcher pattern of ``models/registry.py`` / ``losses``: re-registering
a name raises rather than silently overwriting.

Capability flags gate rankers whose ground truth is external data:

* ``requires_likert``   — needs radiologist Likert observations.
* ``requires_task_net`` — needs a frozen downstream task network (the ΔP target).
* ``opt_in``            — never resolved into a default pool; a run must name it.
* ``data_kwargs``       — the ``__init__`` keyword(s) that CARRY that external
  data. Declaring the flag without declaring the kwarg makes the gate decorative:
  ``available_rankers(have_task_net=True)`` would happily hand back a ranker
  constructed with ``task_performance=None`` that then raises inside ``rank()``.
  :func:`check_injected_data` closes that hole — the flag and the injected object
  are checked together (issue #258).

:func:`available_rankers` returns only the specs whose declared dependencies are
present, so a run without Likert data simply omits HiBSL/ACSS instead of crashing.

Redundancy (``fuses``)
----------------------
A ranker that is a deterministic **fusion of other registered rankers** must not
vote alongside its own constituents: ``Sim2RankRanker`` is exactly
``0.5·minmax(ADR) + 0.3·minmax(SCVR) + 0.2·minmax(CDRS)``, so a pool holding all
four gives the ADR axis 1.5 votes, SCVR 1.3 and CDRS 1.2 against 1.0 for every
genuinely independent ranker — and the "N independent rankers concur" claim is
then counting one axis twice (issue #256). ``fuses`` declares that relationship;
:func:`available_rankers` drops the fusion when *all* of its constituents are in
the pool (they carry strictly more information than the fusion of them).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .base import BaseRanker

logger = logging.getLogger(__name__)

__all__ = [
    "RankerSpec",
    "available_rankers",
    "check_data_dependencies",
    "check_injected_data",
    "get_ranker",
    "has_ranker",
    "list_rankers",
    "redundant_rankers",
    "register_ranker",
]


@dataclass(frozen=True)
class RankerSpec:
    """Static metadata about a registered ranker."""

    name: str
    cls: type[BaseRanker]
    generation: int
    requires_likert: bool
    requires_task_net: bool
    description: str
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: ``__init__`` keyword(s) carrying the external ground truth this ranker
    #: needs. Declared together with ``requires_*`` so the capability gate can
    #: verify the data was actually *injected*, not merely declared (#258).
    data_kwargs: tuple[str, ...] = ()
    #: Names of the registered rankers this one is a deterministic fusion of. A
    #: fusion must not vote next to its own constituents (#256).
    fuses: tuple[str, ...] = ()
    #: Never resolved by :func:`available_rankers`: the ranker is run only when a
    #: caller names it, so registering it cannot change an existing pool's aggregate.
    opt_in: bool = False

    def instantiate(self, **overrides: Any) -> BaseRanker:
        """Build the ranker from its default kwargs (overridable)."""
        return self.cls(**{**self.default_kwargs, **overrides})


_REGISTRY: dict[str, RankerSpec] = {}


def register_ranker(
    name: str,
    *,
    generation: int,
    requires_likert: bool = False,
    requires_task_net: bool = False,
    description: str = "",
    default_kwargs: Mapping[str, Any] | None = None,
    data_kwargs: tuple[str, ...] = (),
    fuses: tuple[str, ...] = (),
    opt_in: bool = False,
) -> Callable[[type], type]:
    """Decorator: register a :class:`BaseRanker` subclass under ``name``.

    Re-registering the same ``name`` raises ``ValueError`` (strict-registry
    convention). ``generation`` is lineage metadata (1=statistical, 2=info/
    sensitivity, 3=structural, 4=clinical/causal).

    Args:
        data_kwargs: ``__init__`` keyword(s) through which the external ground
            truth is injected (e.g. ``("task_performance",)``). Mandatory
            whenever ``requires_likert`` or ``requires_task_net`` is set —
            without it the capability gate can only *omit* the ranker, never
            *supply* it, which is what made the Gen-4 pipeline path dead code.
        fuses: Registered ranker names this one deterministically fuses.
        opt_in: Keep the ranker out of every :func:`available_rankers` pool.
    """
    if (requires_likert or requires_task_net) and not data_kwargs:
        raise ValueError(
            f"Ranker {name!r} declares a data dependency (requires_likert="
            f"{requires_likert}, requires_task_net={requires_task_net}) but no "
            "data_kwargs. Declare the __init__ keyword(s) that carry the external "
            "ground truth, otherwise the capability gate can only omit the ranker "
            "or hand back one whose rank() will raise (issue #258)."
        )

    def _decorator(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(
                f"Ranker {name!r} is already registered (existing: "
                f"{_REGISTRY[name].cls.__module__}.{_REGISTRY[name].cls.__name__}, "
                f"new: {cls.__module__}.{cls.__name__})."
            )
        if not (isinstance(cls, type) and issubclass(cls, BaseRanker)):
            raise TypeError(
                f"Ranker {name!r} ({cls.__name__}) must subclass BaseRanker; "
                "the unified registry has one contract — rank(metric_set, dataset)."
            )
        _REGISTRY[name] = RankerSpec(
            name=name,
            cls=cls,
            generation=generation,
            requires_likert=requires_likert,
            requires_task_net=requires_task_net,
            description=description,
            default_kwargs=dict(default_kwargs or {}),
            data_kwargs=tuple(data_kwargs),
            fuses=tuple(fuses),
            opt_in=opt_in,
        )
        logger.debug("Registered ranker %r (gen=%d, cls=%s)", name, generation, cls.__name__)
        return cls

    return _decorator


def get_ranker(name: str) -> RankerSpec:
    """Resolve a registered ranker by name (KeyError lists all names on a typo)."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Ranker {name!r} is not registered. Available: {available}.")
    return _REGISTRY[name]


def list_rankers() -> list[RankerSpec]:
    """Every registered ranker spec, sorted by (generation, name)."""
    return sorted(_REGISTRY.values(), key=lambda s: (s.generation, s.name))


def redundant_rankers(specs: list[RankerSpec]) -> dict[str, tuple[str, ...]]:
    """``{fusion_name: constituents}`` for every fusion whose parts all vote.

    A deterministic fusion carries no information its constituents do not already
    carry, so keeping both in one equal-weight pool silently re-weights the fused
    axes (issue #256). This reports the offenders; :func:`available_rankers` drops
    them.
    """
    present = {s.name for s in specs}
    return {s.name: s.fuses for s in specs if s.fuses and set(s.fuses) <= present}


def available_rankers(
    *,
    have_likert: bool = False,
    have_task_net: bool = False,
    drop_redundant_fusions: bool = True,
) -> list[RankerSpec]:
    """Registered rankers whose declared data dependencies are satisfied.

    A run without Likert / task-net data simply omits the rankers that need it
    (no silent fallback — the omission is explicit and auditable), instead of
    instantiating a ranker that would produce garbage.

    When ``drop_redundant_fusions`` (the default), a ranker that merely *fuses*
    other members of the resolved pool is dropped: it would otherwise cast a
    second, correlated vote for axes that already vote for themselves, inflating
    their effective weight and corrupting any "N independent rankers concur"
    claim (issue #256). Set it to ``False`` only to reproduce a legacy pool.
    """
    out = []
    for spec in list_rankers():
        if spec.opt_in:
            continue
        if spec.requires_likert and not have_likert:
            continue
        if spec.requires_task_net and not have_task_net:
            continue
        out.append(spec)

    if drop_redundant_fusions:
        redundant = redundant_rankers(out)
        if redundant:
            logger.info(
                "available_rankers: dropping fusion ranker(s) %s — their "
                "constituents %s already vote independently, so keeping both "
                "would count one axis twice (issue #256).",
                sorted(redundant),
                sorted({c for parts in redundant.values() for c in parts}),
            )
            out = [s for s in out if s.name not in redundant]
    return out


def has_ranker(name: str) -> bool:
    return name in _REGISTRY


def check_injected_data(
    specs: list[RankerSpec], ranker_kwargs: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    """Errors for capability-gated rankers whose external data was not injected.

    ``requires_task_net=True`` says the ranker *needs* a frozen task net; it does
    not say anybody handed one over. Before this check, ``from_registry(
    have_task_net=True)`` built ``DCMSRanker(task_performance=None)`` and the
    failure only surfaced (as a ``ValueError``) once ``rank()`` ran — the gate
    could express "omit" or "crash", never "supply" (issue #258).
    """
    errors: list[str] = []
    for spec in specs:
        supplied = ranker_kwargs.get(spec.name, {})
        for kw in spec.data_kwargs:
            value = supplied.get(kw, spec.default_kwargs.get(kw))
            if value is None:
                errors.append(
                    f"Ranker {spec.name!r} was resolved as available but its "
                    f"external ground truth {kw!r} was not injected. Pass it via "
                    f"ranker_kwargs={{{spec.name!r}: {{{kw!r}: <data>}}}}, or drop "
                    f"the capability flag from the run."
                )
    return errors


def check_data_dependencies(
    enabled: list[str], *, have_likert: bool, have_task_net: bool
) -> list[str]:
    """Human-readable errors for unmet ranker data dependencies (Tier-1 audit)."""
    errors: list[str] = []
    for name in enabled:
        if not has_ranker(name):
            errors.append(
                f"Ranker {name!r} is enabled but not registered (typo? module not imported?)."
            )
            continue
        spec = _REGISTRY[name]
        if spec.requires_likert and not have_likert:
            errors.append(f"Ranker {name!r} requires Likert data (requires_likert=True).")
        if spec.requires_task_net and not have_task_net:
            errors.append(f"Ranker {name!r} requires a task network (requires_task_net=True).")
    return errors
