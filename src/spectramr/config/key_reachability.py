"""Can a config key's reader actually run?

``tools/audit/schema_key_consumption.py`` answers a weaker question: it indexes
the package with ripgrep and calls a key *consumed* when any line mentions it.
Whether that line can execute is not part of the question. That is how
``logging.tracking.tensorboard_dir`` passed the gate while its only reader sat in
``ComprehensiveLoggingService.__init__`` -- a class the bootstrap never
constructs (#928; #932 fixed that one instance, not the gate).

This module owns the harder predicate so ``tests/``, ``scripts/ci/`` and the
audit share one implementation instead of each growing a private copy.

The bias is one-sided, and deliberately so
------------------------------------------
This repo resolves components through ``@register_*`` tables and a DI container
by design, so most call edges are dynamic and *cannot* be resolved statically.
An analysis that guessed "unreachable" on ambiguity would call most of the tree
dead and license deleting live code. Therefore:

* every ambiguity resolves to **reachable**, with the ambiguity named in
  ``reason``;
* a verdict of *unreachable* is only ever returned when the analysis positively
  showed the enclosing scope dead -- a method of a class nothing constructs, or
  a function whose name is never referenced from any live scope.

So ``reachable=True`` means "not shown dead". ``reachable=False`` is weaker than
it looks and **must be read together with**
:class:`ReadEvidence`: ``NO_LIVE_READ`` is a call-graph finding, while
``NO_READ_FOUND`` is an *absence of evidence* that a token-naming-nothing
consumer (``**model_dump()``, ``**kwargs``, a runtime-built field name) is
indistinguishable from. Branch on ``verdict.evidence``, never on the boolean
alone and never on ``reason``.

What counts as evidence
-----------------------
*Scopes* (the places a read can sit) are taken from ``src/spectramr/`` only,
excluding ``config/schemas/`` -- a schema module naming its own field is a
declaration, not consumption, which is the same zone rule the rg index uses.

*Liveness evidence* (constructions, references, registry escapes) additionally
reads ``runners/``, ``scripts/`` and ``tools/`` when they exist. Those are not
entry points of the shipped package, but a class they construct is demonstrably
alive, and ignoring them is exactly the trap that would delete live code:
``runners/run_explicit.py`` constructs ``ComprehensiveLoggingService``.
``tests/`` is excluded -- a test constructing a class does not make it reachable
in production, and including it would make every class live.

Known unsoundness
-----------------
* **Keys are matched by leaf name**, like the rg index, because a Python
  consumer reaches a field through an attribute chain no single token spells.
  Two blocks sharing a leaf share their sites. This over-counts reads, which is
  the safe direction here.
* Call edges are resolved **by name, globally**: a call to ``foo(...)`` marks
  every ``def foo`` in the tree live. Over-approximating, again by design.
* A read reached only from outside the analysed roots (a notebook, a downstream
  package, ``tests/``) is reported unreachable. That is the intended scope, not
  an oversight -- but it means an *unreachable* verdict is a prompt to look, not
  a licence to delete unread.
* **A consumer that names no token is invisible.** ``**cfg.model_dump()``
  splatting and a field name built at runtime
  (``getattr(cfg, f"compute_{name}")``) both reach a field without spelling it,
  and land as ``NO_READ_FOUND``. This is the same blind spot the rg index
  documents, and no analysis fixes it -- a name never written cannot be
  recovered. It can only be *declared*, beside the accessor that knows: pass
  ``external_reads`` to :func:`is_key_reachable` and the path is reported as
  :attr:`ReadEvidence.ACCESSOR_READ` rather than as an absence. See
  ``docs/contributing/ci.rst``, "Declaring a read the token index cannot see".
"""

from __future__ import annotations

from collections.abc import Mapping

# ``PACKAGE_DIR`` is re-exported: it is a documented public constant that
# moved to the index module with the code that derives it. ``as`` is the
# explicit re-export spelling, so it is not an unused import.
from spectramr.config.key_reachability_index import PACKAGE_DIR as PACKAGE_DIR
from spectramr.config.key_reachability_index import _index
from spectramr.config.key_reachability_model import (
    _LINE_STRIDE,
    ClassVerdict,
    ReachabilityVerdict,
    ReadEvidence,
    _Index,
    _Scope,
    accessor_verdict,
)

__all__ = [
    "ClassVerdict",
    "ReachabilityVerdict",
    "ReadEvidence",
    "class_liveness",
    "is_key_reachable",
]


# ---------------------------------------------------------------------------
# Public predicates
# ---------------------------------------------------------------------------


def _unpack(packed: int) -> tuple[int, int]:
    return divmod(packed, _LINE_STRIDE)


def _describe(index: _Index, packed: int) -> str:
    scope_id, lineno = _unpack(packed)
    return f"{index.scopes[scope_id].file}:{lineno}"


def class_liveness(name: str) -> ClassVerdict:
    """Can ``name`` be instantiated on a path that runs?

    Args:
        name: A bare class name. Resolved globally: every ``class name`` in the
            analysed trees is one candidate, because an import alias makes the
            qualified spelling unreliable.

    Returns:
        A :class:`ClassVerdict`. ``live=False`` means no construction, no
        decorator, no reference from any live scope and no live subclass -- not
        merely "no ``name(...)`` was found".
    """
    index = _index()
    if name not in index.classes:
        return ClassVerdict(
            live=False, reason=f"no `class {name}` in the analysed trees", evidence=()
        )

    constructions = tuple(
        _describe(index, packed)
        for packed in index.constructions.get(name, ())
        if _unpack(packed)[0] in index.live_scopes
    )
    if constructions:
        return ClassVerdict(
            live=True,
            reason=f"`{name}(...)` is called from a live scope",
            evidence=constructions,
        )

    if name not in index.live_classes:
        definitions = tuple(f"{r.file}:{r.lineno}" for r in index.classes[name])
        return ClassVerdict(
            live=False,
            reason=(
                f"`{name}` is not constructed anywhere in the analysed trees, "
                "carries no registering decorator, is referenced from no live "
                "scope and has no live subclass"
            ),
            evidence=definitions,
        )

    escapes = tuple(
        _describe(index, packed)
        for packed in index.sites.get(name, ())
        if _unpack(packed)[0] in index.live_scopes
    )
    return ClassVerdict(
        live=True,
        reason=(
            f"`{name}` is live by ambiguity ({index.live_classes[name]}): the name "
            "escapes into a value position -- a registry table, a DI "
            "registration or a container -- and this analysis cannot rule out "
            "that something calls it"
        ),
        evidence=escapes[:20],
    )


def is_key_reachable(
    dotted_path: str,
    *,
    external_reads: Mapping[str, str] | None = None,
) -> ReachabilityVerdict:
    """Can any read of ``dotted_path`` execute?

    Args:
        dotted_path: A dotted config path, e.g.
            ``logging.tracking.enable_tensorboard``. Matched by its **leaf**
            name, like the rg index it replaces, because a consumer reaches the
            field through an attribute chain no single token spells.
        external_reads: Paths an accessor declares it reads, mapped to prose
            naming that accessor. Keyed by the **full dotted path**, unlike the
            token index above -- an accessor receives a specific config block, so
            ``losses.gan.lambda_adv`` being read says nothing about
            ``training.multi.stages.stage_config.loss.gan.lambda_adv``, and
            matching this map by leaf would call 54 such stage-scoped paths
            consumed. Optional, and there is no default: this module sits at the
            bottom of the dependency chain and may not import the ``models/``
            package where the loss accessor lives (NN5), so the caller that owns
            both ends injects it.

    Returns:
        A :class:`ReachabilityVerdict`. ``reachable=True`` means "not shown
        dead" -- it is the answer for every unresolved dynamic dispatch, and
        ``reason`` says which. When ``reachable`` is ``False``, branch on
        ``evidence``: ``NO_LIVE_READ`` means every read sits in a scope shown
        unable to run, while ``NO_READ_FOUND`` means no token was found at all
        and a splatted or runtime-named consumer cannot be ruled out.

        The four evidence kinds are decided in strict order: a live token read
        wins (``LIVE_READ``), then a declared accessor read (``ACCESSOR_READ``),
        then dead token reads (``NO_LIVE_READ``), then nothing (``NO_READ_FOUND``).
        The order changes only which *evidence* is reported, never ``reachable``:
        a path that is both live-read and declared is reachable either way, and
        reporting the call graph is strictly more informative than reporting a
        declaration. Checking the map before the dead-sites case is what makes a
        key with real-but-dead token reads answer ``ACCESSOR_READ`` instead of
        ``NO_LIVE_READ`` -- that key IS read, by the accessor.
    """
    index = _index()
    leaf = dotted_path.rsplit(".", 1)[-1]

    # Deduped: one source line naming the leaf twice is one read site, so that
    # the count in ``reason`` agrees with the length of ``sites``.
    reads = list(
        dict.fromkeys(
            packed
            for packed in index.sites.get(leaf, ())
            if index.scopes[_unpack(packed)[0]].read_zone
        )
    )
    sites = tuple(_describe(index, packed) for packed in reads)

    live = [packed for packed in reads if _unpack(packed)[0] in index.live_scopes]
    if live:
        scope = index.scopes[_unpack(live[0])[0]]
        return ReachabilityVerdict(
            reachable=True,
            sites=sites,
            reason=(
                f"{len(live)} of {len(reads)} read(s) sit in a live scope; "
                f"first at {_describe(index, live[0])} in "
                f"{scope.kind} `{scope.qualname}`"
                + (f" -- reached because {_scope_reason(scope)}" if scope.kind != "module" else "")
            ),
            evidence=ReadEvidence.LIVE_READ,
        )

    accessor = (external_reads or {}).get(dotted_path)
    if accessor is not None:
        return accessor_verdict(dotted_path, leaf, sites, accessor)

    if not reads:
        return ReachabilityVerdict(
            reachable=False,
            sites=(),
            reason=(
                f"no read of `{leaf}` anywhere in src/spectramr/ outside "
                "config/schemas/ -- nothing names it. NOT a call-graph finding: a "
                "consumer that names no token (`**model_dump()` splatting, "
                "`**kwargs` forwarding, a runtime-built field name) looks "
                "identical from here. Read the would-be consumer before acting."
            ),
            evidence=ReadEvidence.NO_READ_FOUND,
        )

    return ReachabilityVerdict(
        reachable=False,
        sites=sites,
        reason=_dead_reason(index, reads),
        evidence=ReadEvidence.NO_LIVE_READ,
    )


def _scope_reason(scope: _Scope) -> str:
    if scope.class_name is None:
        return f"the name `{scope.qualname.rsplit('.', 1)[-1]}` is called from a live scope"
    return class_liveness(scope.class_name).reason


def _dead_reason(index: _Index, reads: list[int]) -> str:
    """Why every read is dead, leading with the class-construction case."""
    dead_classes: dict[str, str] = {}
    dead_functions: set[str] = set()
    for packed in reads:
        scope = index.scopes[_unpack(packed)[0]]
        if scope.class_name is not None and scope.class_name not in index.live_classes:
            dead_classes.setdefault(scope.class_name, _describe(index, packed))
        else:
            dead_functions.add(scope.qualname)

    parts: list[str] = []
    if dead_classes:
        named = ", ".join(
            f"`{name}` (read at {site})" for name, site in sorted(dead_classes.items())
        )
        parts.append(
            f"every read sits in a class that is not constructed on any path "
            f"this analysis can reach: {named}"
        )
    if dead_functions:
        parts.append(
            "reads sit in functions whose names are referenced from no live "
            f"scope: {', '.join(sorted(dead_functions)[:5])}"
        )
    return "; ".join(parts)
