"""Every training strategy reads the live loop iteration, never a frozen ``step``.

``TrainingEnvironment`` is ``frozen=True`` and declares no ``step`` field, and the
training loop passes ``iteration=`` -- never ``step=`` -- into every step hook. So
``getattr(self.env, "step", 0)``, ``self.env.step`` and ``kwargs.get("step", 0)``
each evaluate to a constant ``0`` for the whole run (pitfall #16). The elected owner
is :func:`resolve_loop_iteration` (``infrastructure/training/loop_state.py``), which
reads the ``LoopState`` the loop writes at every step.

Frozen at ``0`` a metrics throttle fires on every batch and a warm-up gate never
opens: ``vf_admm_strategy.py`` dropped an ``l1`` term at weight 10.0 from every step
of five live arms (#1937).

**Why AST and not a text scan.** Ten lines under ``strategies/`` spell these forms
inside *comments* describing the historical defect; a grep gate false-positives on
all ten and gets switched off. Comments are absent from the tree, and a docstring
that quotes the spelling is one ``Constant``, never a ``Call``.

**Why the receiver must be the ``**kwargs``-bound name** rather than the literal
``kwargs``: ``pipelines/hpo.py:548`` reads ``metrics.get("step", ...)`` off a metrics
CSV and ``services/checkpoint_service.py:300`` reads a ``step`` its callers really do
pass -- both correct, both caught by a bare-name rule. Binding to the parameter also
means a renamed bag (``**kw``) is still covered.

**Allowlisted by construction:** ``kwargs.get("iteration", kwargs.get("step", 0))``
reaches the real value through the first key, so its nested ``step`` read is a
fallback rather than a frozen read. Four sites use that form. The inverted spelling
(``step`` first, ``iteration`` as the fallback) is *not* allowlisted -- it resolves
to ``0`` before it ever consults ``iteration``.

Scope note, stated rather than implied: this scans ``infrastructure/training/``
recursively, so ``strategies/`` **and** ``strategies/mixins/`` are covered. Shape 2
(``self.env.step``) has no live instance today and is carried by its plant alone --
that is the shape a fix-then-regress would most easily reintroduce.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_TRAINING = (
    Path(__file__).resolve().parents[2] / "src" / "spectramr" / "infrastructure" / "training"
)


def _is_self_env(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "env"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_step_const(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value == "step"


def _exempt_calls(tree: ast.AST) -> set[int]:
    """``id()`` of every Call nested in the default of a ``.get("iteration", ...)``.

    That is the two-key fallback form, which reaches the real iteration first.
    """
    exempt: set[int] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "iteration"
        ):
            continue
        for default in node.args[1:]:
            exempt.update(id(sub) for sub in ast.walk(default) if isinstance(sub, ast.Call))
    return exempt


def _frozen_read(node: ast.AST, varkw: str | None, exempt: set[int]) -> str | None:
    """The violation shape ``node`` is, or ``None``."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and _is_self_env(node.args[0])
        and _is_step_const(node.args[1])
    ):
        return 'getattr(self.env, "step", ...)'
    if isinstance(node, ast.Attribute) and node.attr == "step" and _is_self_env(node.value):
        return "self.env.step"
    if (
        varkw is not None
        and isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == varkw
        and node.args
        and _is_step_const(node.args[0])
        and id(node) not in exempt
    ):
        return f'{varkw}.get("step", ...)'
    return None


def _walk(node: ast.AST, varkw: str | None, exempt: set[int], out: list[str]) -> None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        varkw = node.args.kwarg.arg if node.args.kwarg else varkw
    shape = _frozen_read(node, varkw, exempt)
    if shape is not None:
        out.append(f"{getattr(node, 'lineno', 0)}: {shape}")
    for child in ast.iter_child_nodes(node):
        _walk(child, varkw, exempt, out)


def violations(source: str) -> list[str]:
    """``"<lineno>: <shape>"`` for every frozen-``step`` read in ``source``."""
    tree = ast.parse(source)
    out: list[str] = []
    _walk(tree, None, _exempt_calls(tree), out)
    return sorted(set(out))


# --------------------------------------------------------------------------- #
# Planted violations -- one per shape the rule can take (non-negotiable 15)
# --------------------------------------------------------------------------- #
_PLANT_GETATTR = """
class S(BaseTrainingStrategy):
    def _compute_losses_impl(self, batch, epoch, **kwargs):
        current_step = getattr(self.env, "step", 0) if self.env else 0
        return self._metrics(current_step=current_step)
"""

_PLANT_ATTRIBUTE = """
class S(BaseTrainingStrategy):
    def _compute_losses_impl(self, batch, epoch, **kwargs):
        return self._metrics(current_step=self.env.step)
"""

_PLANT_KWARGS = """
class S(BaseTrainingStrategy):
    def _compute_losses_impl(self, batch, epoch, **kwargs):
        return self._metrics(current_step=kwargs.get("step", 0))
"""

_PLANT_RENAMED_BAG = """
class S(BaseTrainingStrategy):
    def _compute_losses_impl(self, batch, epoch, **kw):
        return self._metrics(current_step=int(kw.get("step", 0)))
"""

_PLANT_INVERTED = """
class S(BaseTrainingStrategy):
    def _compute_losses_impl(self, batch, epoch, **kwargs):
        return self._metrics(current_step=kwargs.get("step", kwargs.get("iteration", 0)))
"""

_FIXED = """
class S(BaseTrainingStrategy):
    def _compute_losses_impl(self, batch, epoch, **kwargs):
        iteration = resolve_loop_iteration(self)
        return self._metrics(current_step=iteration)
"""

_ALLOWED_TWO_KEY = """
class S(BaseTrainingStrategy):
    def _compute_losses_impl(self, batch, epoch, **kwargs):
        iteration = int(kwargs.get("iteration", kwargs.get("step", 0)) or 0)
        return self._metrics(current_step=iteration)
"""

_ALLOWED_COMMENT = """
class S(BaseTrainingStrategy):
    def _compute_losses_impl(self, batch, epoch, **kwargs):
        # Live iteration (loop_state seam), not the frozen getattr(self.env, "step", 0)
        # (=0) and not kwargs.get("step", 0) -- see pitfall #16.
        \"\"\"Historically this read self.env.step and kwargs.get("step", 0).\"\"\"
        return self._metrics(current_step=resolve_loop_iteration(self))
"""

_ALLOWED_OTHER_RECEIVER = """
def poll(metrics, last_step, **kwargs):
    step = int(metrics.get("step", metrics.get("iteration", last_step)))
    return step
"""


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (_PLANT_GETATTR, ['getattr(self.env, "step", ...)']),
        (_PLANT_ATTRIBUTE, ["self.env.step"]),
        (_PLANT_KWARGS, ['kwargs.get("step", ...)']),
        (_PLANT_RENAMED_BAG, ['kw.get("step", ...)']),
        (_PLANT_INVERTED, ['kwargs.get("step", ...)']),
    ],
    ids=["getattr", "attribute", "kwargs", "renamed_bag", "inverted_two_key"],
)
def test_planted_frozen_read_is_caught(source: str, expected: list[str]) -> None:
    """Each shape the rule can take turns the scanner red on its own."""
    found = [v.split(": ", 1)[1] for v in violations(source)]
    assert found == expected


@pytest.mark.parametrize(
    "source",
    [_FIXED, _ALLOWED_TWO_KEY, _ALLOWED_COMMENT, _ALLOWED_OTHER_RECEIVER],
    ids=["resolve_loop_iteration", "two_key_fallback", "comment_and_docstring", "other_receiver"],
)
def test_correct_forms_are_not_flagged(source: str) -> None:
    assert violations(source) == []


def test_the_scanner_sees_the_mixins_subpackage() -> None:
    """A non-recursive glob would miss ``strategies/mixins/`` -- three files there
    carry the spelling in comments and one carries the allowlisted form."""
    scanned = {p.name for p in _TRAINING.rglob("*.py")}
    assert "adversarial.py" in scanned and "kspace.py" in scanned


@pytest.mark.parametrize(
    "path",
    sorted(p for p in _TRAINING.rglob("*.py") if "__pycache__" not in p.parts),
    ids=lambda p: str(p.relative_to(_TRAINING)),
)
def test_no_strategy_reads_a_frozen_step(path: Path) -> None:
    found = violations(path.read_text(encoding="utf-8"))
    assert not found, (
        f"{path.relative_to(_TRAINING)} reads a frozen step at {found}. "
        "TrainingEnvironment has no `step` field and the loop passes `iteration=`, "
        "so each of these is a constant 0 for the whole run. Use "
        "`resolve_loop_iteration(self)` from infrastructure.training.loop_state "
        "(pitfall #16, #1937)."
    )
