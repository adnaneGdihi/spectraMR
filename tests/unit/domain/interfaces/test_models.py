"""Package-level naming invariant for ``spectramr.domain.interfaces``.

A name this package advertises must mean one thing. A sibling module may
re-export an advertised name, but it must not *define* a second object under
it: ``from spectramr.domain.interfaces import IDiscriminator`` and
``from spectramr.domain.interfaces.models import IDiscriminator`` then hand
back different classes, and only the fully-qualified path reaches the second
one -- so the shadowed definition is unreachable to every caller who imports
the way the package's own docstring recommends.

The check is by identity rather than by name, because re-export is the normal
case here and is indistinguishable from shadowing on names alone: three
siblings define an advertised name and are correct, since the package binds
*their* object.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from spectramr.domain import interfaces
from spectramr.domain.interfaces import models

_PKG = "spectramr.domain.interfaces"
_PKG_DIR = Path(interfaces.__file__).parent
_MODELS_PY = Path(models.__file__)

# The advertised surface. This package ships in the public export, so a change
# to this list is an API change and has to be made deliberately.
_PUBLIC_SURFACE = (
    "IDiffusionModel",
    "IDiffusionProcess",
    "IDiscriminator",
    "IDownstreamModelService",
    "IGenerator",
    "IModel",
    "IModelFactory",
    "IPrivacyAccountant",
    "ITransformer",
)

# Protocols in ``models.py`` that no sibling and no re-export competes for.
_UNCONTESTED_PROTOCOLS = (
    "GenerativeModel",
    "ReconstructionModel",
    "IEncoder",
    "IDecoder",
)


def _module_level_definitions(path: Path) -> set[str]:
    """Names bound at module level by ``path``, by AST.

    Only ``tree.body`` is walked: ``ast.walk`` descends into function bodies,
    where a local class of the same name is not a module-level binding and
    cannot shadow anything.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def test_no_sibling_module_redefines_an_advertised_name() -> None:
    """A sibling may re-export an advertised name; it may not redefine it."""
    advertised = set(interfaces.__all__)
    offenders: list[str] = []
    examined = 0

    for path in sorted(_PKG_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = importlib.import_module(f"{_PKG}.{path.stem}")
        for name in sorted(_module_level_definitions(path) & advertised):
            examined += 1
            if getattr(interfaces, name) is not getattr(module, name):
                package_binds = getattr(interfaces, name).__module__
                offenders.append(
                    f"{path.name} defines {name}, but the package binds {package_binds}.{name}"
                )

    assert examined >= 3, (
        "the check found no sibling defining an advertised name at all -- it is "
        "passing over an empty set, not passing on the invariant"
    )
    assert not offenders, "shadowed advertised names:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("name", ["IGenerator", "IDiscriminator"])
def test_models_module_defines_neither_shadowed_protocol(name: str) -> None:
    """Pinned by absence from the AST, not from the source text.

    A ``getsource`` pin is satisfiable by a docstring that merely mentions the
    name; only the parse tree distinguishes a definition from prose about one.
    """
    assert name not in _module_level_definitions(_MODELS_PY)


def test_the_uncontested_protocols_survive() -> None:
    """The deletion is narrow: only the two names the package re-binds go."""
    defined = _module_level_definitions(_MODELS_PY)
    assert set(_UNCONTESTED_PROTOCOLS) <= defined
    for name in _UNCONTESTED_PROTOCOLS:
        assert getattr(models, name)._is_protocol is True, f"{name} is not a Protocol"


def test_the_advertised_surface_is_unchanged() -> None:
    """None of the above may quietly shrink what the package exports."""
    assert tuple(interfaces.__all__) == _PUBLIC_SURFACE
    for name in _PUBLIC_SURFACE:
        assert getattr(interfaces, name) is not None
