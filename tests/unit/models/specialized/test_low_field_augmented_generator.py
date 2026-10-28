"""``LowFieldStyleAugmentedGenerator`` composes three modules and registered none.

It implements ``IGenerator`` -- which resolves to ``IModel(ABC)``, carrying no
``nn.Module`` -- so ``base_generator``, ``style_augmentation`` and
``condition_encoder`` were ordinary attributes. The wrapper forwarded correctly
and trained nothing it wrapped (#801 group 1).

This class takes its parts by constructor argument, so composition is the thing
to check: a wrapper that fails to register its children is indistinguishable
from a working one until a checkpoint is restored or an optimizer is built.

CPU-only, tiny tensors. ``forward`` is deliberately not exercised here -- the
registration contract is what this change moves.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from spectramr.models.specialized.low_field_augmented_generator import (  # noqa: E402
    LowFieldStyleAugmentedGenerator,
)


def _build() -> LowFieldStyleAugmentedGenerator:
    return LowFieldStyleAugmentedGenerator(
        base_generator=nn.Conv2d(1, 1, 3, padding=1),
        style_augmentation=nn.Conv2d(1, 1, 3, padding=1),
    )


def test_the_generator_is_an_nn_module() -> None:
    assert issubclass(LowFieldStyleAugmentedGenerator, nn.Module)


def test_the_composed_parts_register_as_submodules() -> None:
    g = _build()
    children = dict(g.named_children())
    assert "base_generator" in children
    assert "style_augmentation" in children
    assert sum(1 for _ in g.parameters()) > 0, "an optimizer would receive no tensors"


def test_the_wrapped_weights_reach_the_state_dict() -> None:
    """Without registration a checkpoint of this wrapper restored nothing."""
    keys = _build().state_dict()
    assert any(k.startswith("base_generator.") for k in keys)
    assert any(k.startswith("style_augmentation.") for k in keys)


def test_call_is_not_shadowed() -> None:
    assert LowFieldStyleAugmentedGenerator.__call__ is nn.Module.__call__
