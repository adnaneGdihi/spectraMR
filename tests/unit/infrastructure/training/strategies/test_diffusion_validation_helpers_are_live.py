"""The validation orchestrator names only helpers it calls (#2062).

``_generate_validation_prediction``'s docstring advertised
``_extract_mask_from_batch`` as its mask path. Nothing called that method, and
its fallback built a mask at a hard-coded R=4. The method is deleted. The
detector below fails if the docstring names a ``self._helper`` that the body
does not call, or that the class does not define.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap

from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy

_NAMED_HELPER = re.compile(r"``(_[a-z][a-z0-9_]*)``|^\s*-\s*(_[a-z][a-z0-9_]*):", re.MULTILINE)


def _named_but_not_called(func) -> list[str]:
    """Helpers the docstring names that the function body never calls on ``self``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    node = tree.body[0]
    doc = ast.get_docstring(node) or ""
    named = {a or b for a, b in _NAMED_HELPER.findall(doc)}
    called = {
        c.func.attr
        for c in ast.walk(node)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and isinstance(c.func.value, ast.Name)
        and c.func.value.id == "self"
    }
    return sorted(n for n in named if n not in called)


def test_the_dead_mask_helper_is_gone():
    assert not hasattr(DiffusionTrainingStrategy, "_extract_mask_from_batch")


def test_the_orchestrator_docstring_names_only_called_helpers():
    func = DiffusionTrainingStrategy._generate_validation_prediction
    assert _named_but_not_called(func) == []


def test_every_named_helper_exists_on_the_class():
    func = DiffusionTrainingStrategy._generate_validation_prediction
    doc = inspect.getdoc(func) or ""
    for a, b in _NAMED_HELPER.findall(doc):
        assert hasattr(DiffusionTrainingStrategy, a or b), a or b


def _planted_bullet(self):
    """Orchestrator.

    Refactored to delegate to helper methods:
    - _extract_mask_from_batch: Mask extraction and validation
    - _apply_masking_and_verify: Apply masking and detect leakage
    """
    return self._apply_masking_and_verify(None, None)


def _planted_literal(self):
    """Builds the mask with ``_extract_mask_from_batch``."""
    return None


def test_the_detector_is_red_on_the_pre_2062_bullet_shape():
    assert _named_but_not_called(_planted_bullet) == ["_extract_mask_from_batch"]


def test_the_detector_is_red_on_the_inline_literal_shape():
    assert _named_but_not_called(_planted_literal) == ["_extract_mask_from_batch"]
