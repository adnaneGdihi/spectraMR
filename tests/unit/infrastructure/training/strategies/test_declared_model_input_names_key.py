"""Every ``_declare_model_input`` call names the tensor the model is fed.

``save_debug_snapshot`` raises when the declared model-input snapshot carries no
``model_input_key`` (``require_model_input_key``). The strategies' unit tests
stub ``_declare_model_input``, so a call site without the key passes them and
fails on the first training step of a real run with snapshots enabled: all five
EI arms and three SSDU arms of the self_supervised cohort did on sabine
(array 17877924, 2026-10-08).

This scans the strategy sources for call sites whose ``extra`` dict literal
lacks the key. ``_BASELINE`` held the seven sites that lacked it when the gate was
added; #2728 named their keys, and it stays empty. The predicate is exercised against
planted violations (non-negotiable 15).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_STRATEGIES = (
    Path(__file__).resolve().parents[5]
    / "src"
    / "spectramr"
    / "infrastructure"
    / "training"
    / "strategies"
)

#: (file stem, enclosing function) of call sites allowed to omit the key. Empty
#: since #2728 named the model input at the seven sites that predated the gate.
_BASELINE: frozenset[tuple[str, str]] = frozenset()


def _names_key(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "extra" and isinstance(kw.value, ast.Dict):
            return any(
                isinstance(k, ast.Constant) and k.value == "model_input_key" for k in kw.value.keys
            )
        if kw.arg == "extra":
            return True  # a non-literal extra cannot be read statically; trust it
    return False


def unnamed_declarations(source: str) -> list[str]:
    """Enclosing function of each ``_declare_model_input`` call without a key."""
    found: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "_declare_model_input"
                and not _names_key(node)
            ):
                found.append(self.stack[-1])
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return found


def _offenders() -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for path in sorted(_STRATEGIES.rglob("*.py")):
        if path.name == "base.py":
            continue
        for fn in unnamed_declarations(path.read_text()):
            out.add((path.stem, fn))
    return out


def test_no_new_call_site_omits_the_key() -> None:
    new = _offenders() - _BASELINE
    assert not new, f"_declare_model_input without extra={{'model_input_key': ...}}: {sorted(new)}"


def test_the_baseline_only_shrinks() -> None:
    fixed = _BASELINE - _offenders()
    assert not fixed, f"remove these from _BASELINE, they name their key now: {sorted(fixed)}"
    assert not _BASELINE, "the baseline is spent (#2728); a new site names its key instead"


def test_the_two_sabine_crash_sites_name_their_key() -> None:
    offenders = _offenders()
    assert ("equivariant_imaging_strategy", "reconstruct") not in offenders
    assert ("ssdu_strategy", "_compute_losses_impl") not in offenders


@pytest.mark.parametrize(
    "source",
    [
        "def f(self):\n    self._declare_model_input({'x': x}, in_kspace_keys=set())\n",
        "def f(self):\n    self._declare_model_input({'x': x}, extra={'why': 'noised'})\n",
        "def f(self):\n    def g():\n        self._declare_model_input({'x': x})\n",
    ],
    ids=["no-extra", "extra-without-key", "nested"],
)
def test_the_predicate_flags_a_planted_site(source: str) -> None:
    assert unnamed_declarations(source)


def test_the_predicate_passes_a_named_site() -> None:
    source = (
        "def f(self):\n    self._declare_model_input({'x': x}, extra={'model_input_key': 'x'})\n"
    )
    assert unnamed_declarations(source) == []
