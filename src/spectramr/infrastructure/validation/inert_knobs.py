"""Declared model knobs that the resolved model provably never reads.

The sibling guard ``check_component_kwargs_reach_constructor`` asks whether a
declared ``model_kwargs`` entry **arrived** at the constructor. This module asks
the next question: having arrived, was it ever **read**?

The two are different, and the gap between them is where the defect lives. A
parameter that is declared in ``__init__``'s signature, documented in its
docstring, and then never referenced in the body arrives perfectly — the
execution ledger records a clean delivery — and changes nothing. Flipping
``activation: complex`` to ``activation: relu`` on
``KSpaceColdDiffusionGenerator`` leaves the module tree, the parameter count and
the forward output bit-identical. Nothing raises, and the value is still stamped
into provenance, so the artifact asserts a choice the run never made.

Not a duplicate of ``scripts/ci/check_model_kwargs_are_read.py``, and the
difference is the whole point. That gate asks whether a key exists **anywhere**
in ``src/spectramr`` -- as a dict key or as any function's named parameter -- and
targets "a name that exists NOWHERE" (issue #1075). It is deliberately
permissive, and that permissiveness is exactly what lets this class through:
``activation`` and ``use_complex_conv`` *are* named parameters of
``KSpaceUNetBlock.__init__``, so the package-wide vocabulary counts them as
read, and neither appears in that gate's 763-entry baseline. This module asks
the narrower, arm-specific question instead -- does the class **this arm
resolves to** read the key? -- so the two are complementary:

===================================  ==========================================
``check_model_kwargs_are_read`` (CI)  ``declared_model_kwargs_are_read`` (audit)
===================================  ==========================================
name exists anywhere in the package?  does *this* ``__init__`` read it?
package-wide vocabulary               one constructor body
catches ``process_type`` (nowhere)    catches ``activation`` (elsewhere only)
ratcheted baseline, CI-only           per-arm, on every audit
===================================  ==========================================

Why this is arm-scoped rather than class-scoped. A dead parameter nobody sets is
a tidiness issue; a dead parameter an experiment **declares** is a false
controlled variable. The check therefore reports the intersection of "unread by
this model" and "written by this arm", so what surfaces is always something the
author typed and expected to matter (CLAUDE.md non-negotiable 8, pitfall #15/#16).

Scope and its limits. This detector is deliberately **static and conservative**:
it answers only for parameters named in the signature. It cannot see two other
ways a knob goes inert, both of which are properties of a *configuration* rather
than of a class, and both of which need the Tier-2 probe to observe:

* **inert by method** — ``dc_weight`` is live code that cannot matter under
  ``dc_method: hard``, because hard data consistency replaces observed lines
  rather than blending toward them, so there is no blend coefficient to apply.
  **This one is now reported** — by
  ``config_health_checker.check_dc_knobs_inert_by_method``, whose readership
  table lives in :mod:`spectramr.infrastructure.physics.dc_settings` (#1525). It
  is still out of scope *here*: this module answers about a class, that one
  about a configuration;
* **inert by branch** — ``reflect_padding_bottleneck_layers`` is read, but only
  on the ``PureKSpaceUNet`` construction path, which ``backbone_type:
  complex_unet`` never takes. Still reported by nothing.

Both are out of scope here, so a clean verdict from this
module means "no *provably-unread* declared knob", never "every declared knob
matters".
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DELIBERATELY_UNREAD",
    "InertKnob",
    "find_inert_declared_knobs",
    "unread_init_params",
]

#: Parameters a constructor accepts and knowingly ignores, with the reason.
#: Keyed by ``(class name, parameter)``. An entry here is a *declaration of
#: intent*, not a suppression: it records that someone looked and decided the
#: parameter earns its place, so the next reader does not rediscover it. Mirrors
#: the ``KNOWN_INERT`` idiom in ``witness/checks/meta_orphan_checks.py``.
DELIBERATELY_UNREAD: dict[tuple[str, str], str] = {
    ("ComplexUNet", "img_size"): (
        "documented in-signature as 'unused, kept for factory compatibility' — "
        "the shared model factory passes img_size to every backbone positionally"
    ),
}

#: Names that make a constructor's parameter reads undecidable by AST. A body
#: reaching for any of these can consume a parameter without ever naming it, so
#: the detector declines to answer rather than accuse (a false positive here
#: costs more than a miss: it sends an author to delete a live knob).
_REFLECTIVE_ESCAPES = frozenset({"locals", "vars", "globals", "eval", "exec"})


@dataclass(frozen=True)
class InertKnob:
    """One declared knob the resolved model never reads."""

    key: str  # the parameter name, e.g. "activation"
    yaml_path: str  # "model.model_kwargs.activation"
    declared_value: Any  # what the arm set it to
    model_type: str  # the arm's declared model_type
    class_name: str  # the class whose __init__ ignores it


def _init_source_tree(cls: type) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Parse ``cls.__init__`` to an AST, or ``None`` when it cannot be read.

    ``inspect.getsource`` follows inheritance, which is the behaviour we want:
    the question is whether the ``__init__`` that actually *runs* reads the
    parameter, not whether some override further down the MRO declares it.
    """
    init = getattr(cls, "__init__", None)
    if init is None or init is object.__init__:
        return None
    try:
        source = textwrap.dedent(inspect.getsource(init))
    except (OSError, TypeError):  # C-implemented, or source unavailable
        return None
    try:
        node = ast.parse(source).body[0]
    except (SyntaxError, IndexError, ValueError):
        return None
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    return node


def unread_init_params(cls: type) -> frozenset[str]:
    """Named ``__init__`` parameters that never appear in the body.

    A parameter counts as READ when its name appears anywhere in the body as a
    load — ``self.x = x``, ``super().__init__(x)``, ``if x:``, an f-string
    interpolation. That deliberately over-counts reads: forwarding a parameter
    straight to a superclass is a genuine use, and only a name that appears
    *nowhere* is reported. Annotations and the docstring are not body reads,
    which is precisely the case this exists to catch.

    Returns an empty set — "no answer", never "no problems" — when the body
    cannot be parsed or reaches for a reflective escape hatch.
    """
    node = _init_source_tree(cls)
    if node is None:
        return frozenset()

    args = node.args
    declared = [
        p.arg
        for p in (*args.posonlyargs, *args.args, *args.kwonlyargs)
        if p.arg not in ("self", "cls")
    ]
    if not declared:
        return frozenset()

    used: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            if sub.id in _REFLECTIVE_ESCAPES:
                return frozenset()  # undecidable — decline to answer
            used.add(sub.id)

    return frozenset(p for p in declared if p not in used)


def declared_knobs_out_of_scope(
    declared_kwargs: dict[str, Any] | None, model_class: type | None
) -> frozenset[str]:
    """Declared keys this detector structurally cannot answer about.

    A key absorbed by ``**kwargs`` is never a candidate for
    :func:`unread_init_params`, which enumerates only named parameters. On the
    ``kspace_filling`` cohort that is 1365 of 1590 declared keys (85.8 %), so a
    verdict that does not separate *measured* from *in scope* reads as a clean
    bill of health for a population the check never looked at. Callers report
    this count beside the verdict; the package-wide reader census in
    ``scripts/ci/check_model_kwargs_are_read.py`` is what covers these keys.
    """
    if not declared_kwargs or model_class is None or not inspect.isclass(model_class):
        return frozenset()
    node = _init_source_tree(model_class)
    if node is None:
        return frozenset(declared_kwargs)
    args = node.args
    named = {
        p.arg
        for p in (*args.posonlyargs, *args.args, *args.kwonlyargs)
        if p.arg not in ("self", "cls")
    }
    return frozenset(k for k in declared_kwargs if k not in named)


def find_inert_declared_knobs(
    model_type: str | None,
    declared_kwargs: dict[str, Any] | None,
    model_class: type | None,
) -> list[InertKnob]:
    """Declared ``model_kwargs`` entries this model provably never reads.

    Args:
        model_type: The arm's ``model.model_type`` (used for the message only).
        declared_kwargs: The arm's ``model.model_kwargs`` mapping.
        model_class: The class ``model_type`` resolves to, or ``None`` when it
            could not be resolved — in which case the answer is "not measured"
            and an empty list is returned.

    Returns:
        One :class:`InertKnob` per offending key, sorted by key. Entries in
        :data:`DELIBERATELY_UNREAD` are excluded.
    """
    if not declared_kwargs or model_class is None:
        return []
    if not inspect.isclass(model_class):
        return []

    unread = unread_init_params(model_class)
    if not unread:
        return []

    class_name = model_class.__name__
    return [
        InertKnob(
            key=key,
            yaml_path=f"model.model_kwargs.{key}",
            declared_value=declared_kwargs[key],
            model_type=str(model_type),
            class_name=class_name,
        )
        for key in sorted(set(declared_kwargs) & unread)
        if (class_name, key) not in DELIBERATELY_UNREAD
    ]
