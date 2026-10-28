"""A strategy ``__init__`` must not run the setup hook the base ``__init__`` already ran.

``BaseTrainingStrategy.__init__`` ends with ``self._setup_strategy_specific_components()``.
A subclass that calls it again after ``super().__init__()`` runs the whole setup twice, and the
base-class comment's claim that setup is idempotent is false once it adds optimizer param
groups: the first (discarded) copy of each owned module leaves an orphan group on ``opt_g``.

Twelve constructors were nothing but ``super().__init__()`` plus the redundant call and are
fixed. The classes below set state before the call, so removing it needs a per-class read;
they are baselined here and this test fails on any NEW one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

STRATEGIES = Path("src/spectramr/infrastructure/training/strategies")
HOOK = "_setup_strategy_specific_components"

#: ``module::Class`` whose ``__init__`` still calls the hook after doing other work first.
#: Debt, not approval: each needs its constructor state traced before the call can go.
KNOWN_DOUBLE_SETUP = frozenset(
    {
        "diffusion.py::DiffusionTrainingStrategy",
        "domain_adaptation.py::DomainAdaptationTrainingStrategy",
        "gan.py::GANTrainingStrategy",
        "pretraining.py::MAEPretrainingStrategy",
        "reconstruction.py::ReconstructionStepMixin",
        "vae.py::VAETrainingStrategy",
        "vae.py::VQVAETrainingStrategy",
        "volumetric.py::TRELLISTrainingStrategy",
    }
)


def _calls_hook(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == HOOK
    )


def double_setup_classes(source: str, module: str) -> set[str]:
    """``module::Class`` for every class whose own ``__init__`` calls the setup hook."""
    found: set[str] = set()
    for cls in (n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"):
            if any(_calls_hook(s) for s in ast.walk(fn) if isinstance(s, ast.stmt)):
                found.add(f"{module}::{cls.name}")
    return found


def _scan() -> set[str]:
    found: set[str] = set()
    for path in sorted(STRATEGIES.rglob("*.py")):
        if path.name == "base.py":
            continue  # the base __init__ is where the one legitimate call lives
        found |= double_setup_classes(path.read_text(), str(path.relative_to(STRATEGIES)))
    return found


class TestTheDetector:
    """Planted violations: a detector is only a gate for the shapes it has been watched to catch."""

    def test_flags_the_super_then_hook_shape(self) -> None:
        source = (
            "class S(Base):\n"
            "    def __init__(self, *a, **k):\n"
            "        super().__init__(*a, **k)\n"
            "        self._setup_strategy_specific_components()\n"
        )
        assert double_setup_classes(source, "m.py") == {"m.py::S"}

    def test_flags_the_call_after_other_state(self) -> None:
        source = (
            "class S(Base):\n"
            "    def __init__(self, *a, **k):\n"
            "        super().__init__(*a, **k)\n"
            "        self.x = 1\n"
            "        self._setup_strategy_specific_components()\n"
        )
        assert double_setup_classes(source, "m.py") == {"m.py::S"}

    def test_flags_a_call_nested_in_a_branch(self) -> None:
        source = (
            "class S(Base):\n"
            "    def __init__(self, *a, **k):\n"
            "        super().__init__(*a, **k)\n"
            "        if self.flag:\n"
            "            self._setup_strategy_specific_components()\n"
        )
        assert double_setup_classes(source, "m.py") == {"m.py::S"}

    def test_a_constructor_without_the_call_and_the_hook_definition_are_fine(self) -> None:
        source = (
            "class S(Base):\n"
            "    def __init__(self, *a, **k):\n"
            "        super().__init__(*a, **k)\n"
            "    def _setup_strategy_specific_components(self):\n"
            "        self.x = 1\n"
        )
        assert double_setup_classes(source, "m.py") == set()


def test_no_new_strategy_constructor_runs_the_setup_hook_a_second_time() -> None:
    new = _scan() - KNOWN_DOUBLE_SETUP
    assert not new, (
        f"{sorted(new)} call {HOOK}() in their own __init__, but BaseTrainingStrategy.__init__ "
        "already runs it, so the setup executes twice and leaves orphan optimizer param groups. "
        "Delete the call."
    )


def test_the_baseline_has_no_stale_entries() -> None:
    """Fixing one of the nine must shrink the list, so it can only ever go down."""
    stale = KNOWN_DOUBLE_SETUP - _scan()
    assert not stale, f"fixed: remove from KNOWN_DOUBLE_SETUP: {sorted(stale)}"
