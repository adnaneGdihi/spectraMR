"""Regional compilation: compile repeated blocks, not the whole model."""

from __future__ import annotations

import pytest
import torch.nn as nn

from spectramr.infrastructure.training.builders.compile_regions import (
    NoRegionsToCompileError,
    compile_regions,
    has_repeated_blocks,
    is_repeated_blocks,
)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class OtherBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)


class Stacked(nn.Module):
    def __init__(self, n: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(n)])
        self.head = nn.Conv2d(2, 2, 1)


class _Stage(nn.Module):
    """A stage that itself stacks blocks -- the common nested shape."""

    def __init__(self, layers: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Nested(nn.Module):
    def __init__(self, stages: int = 2, layers: int = 3) -> None:
        super().__init__()
        self.stages = nn.ModuleList([_Stage(layers) for _ in range(stages)])


class TestDetection:
    def test_a_uniform_block_list_qualifies(self):
        assert is_repeated_blocks(nn.ModuleList([Block(), Block()]))

    def test_mixed_classes_do_not(self):
        """Planted: dynamo caches on the code object, so a mixed list compiles
        each class separately and gains nothing from being done regionally."""
        assert not is_repeated_blocks(nn.ModuleList([Block(), OtherBlock()]))

    def test_a_single_element_does_not(self):
        """Planted: one child is just the module, and compiling it regionally is
        strictly worse than compiling it."""
        assert not is_repeated_blocks(nn.ModuleList([Block()]))

    @pytest.mark.parametrize(
        "make_leaf",
        [
            lambda: nn.Conv2d(2, 2, 1),
            lambda: nn.Linear(2, 2),
            lambda: nn.AvgPool2d(2),
            lambda: nn.LayerNorm(2),
        ],
        ids=["conv2d", "linear", "avgpool", "layernorm"],
    )
    def test_leaf_primitives_do_not(self, make_leaf):
        """Planted: uniform by the class test, and long enough to pass the
        minimum -- so only the leaf rule can reject these. A single-element list
        would pass for the wrong reason.
        """
        blocks = nn.ModuleList([make_leaf() for _ in range(3)])
        assert len(blocks) >= 2, "must clear _MIN_BLOCKS or this proves nothing"
        assert not is_repeated_blocks(blocks)

    def test_a_plain_module_is_not_a_block_list(self):
        assert not is_repeated_blocks(nn.Sequential(Block(), Block()))

    def test_has_repeated_blocks_searches_the_tree(self):
        assert has_repeated_blocks(Stacked())
        assert not has_repeated_blocks(nn.Sequential(nn.Conv2d(2, 2, 1)))


class TestCompilation:
    def test_every_block_is_compiled(self):
        model = Stacked(n=4)
        assert compile_regions(model, dynamic=True) == 4
        assert all(type(b).__name__ == "OptimizedModule" for b in model.blocks)

    def test_the_parent_is_left_bare(self):
        """Compiling the root as well would reinstate exactly the cost regional
        compilation exists to avoid."""
        model = Stacked()
        compile_regions(model, dynamic=True)
        assert type(model).__name__ == "Stacked"
        assert type(model.blocks).__name__ == "ModuleList"

    def test_nothing_qualifying_raises(self):
        """Planted: an arm that declared regional and silently got whole-model
        compilation would be reporting a configuration it did not run."""
        with pytest.raises(NoRegionsToCompileError, match="nothing to compile regionally"):
            compile_regions(nn.Sequential(nn.Conv2d(2, 2, 1)))

    def test_the_refusal_is_its_own_type(self):
        """So the caller can tell a refusal from a compile failure. Wrapped as
        the latter it picked up the wrong closing instruction -- "set
        compile.enabled: false" -- when the fix is `regional: false`.

        Still a RuntimeError, so anything catching the base class is unaffected.
        """
        assert issubclass(NoRegionsToCompileError, RuntimeError)

    def test_the_refusal_names_the_way_out(self):
        with pytest.raises(NoRegionsToCompileError) as excinfo:
            compile_regions(nn.Sequential(nn.Conv2d(2, 2, 1)))
        assert "regional: false" in str(excinfo.value)


class TestTheStateDictConsequence:
    """The sharp edge. Wrapping a child rather than the root puts the marker
    mid-path, and a leading-prefix strip leaves it in the checkpoint -- where
    every inference path builds a bare model and loads with strict=True."""

    def test_regional_compilation_moves_the_marker_mid_path(self):
        model = Stacked(n=2)
        compile_regions(model, dynamic=True)
        assert any("_orig_mod" in k for k in model.state_dict())
        assert not any(k.startswith("_orig_mod") for k in model.state_dict())

    def test_the_keys_still_load_into_a_bare_model(self):
        """The property that matters: a regionally-compiled checkpoint must be
        readable by the inference path, which builds bare and loads strict."""
        from spectramr.core.module_utils import strip_wrapper_prefixes

        model = Stacked(n=2)
        compile_regions(model, dynamic=True)
        bare = Stacked(n=2)
        bare.load_state_dict(strip_wrapper_prefixes(model.state_dict()), strict=True)

    def test_a_real_submodule_named_module_is_not_destroyed(self):
        """``module`` is an ordinary attribute name, so it is stripped only when
        it leads -- unlike the underscore-prefixed synthetic wrappers."""
        from spectramr.core.module_utils import strip_wrapper_prefixes

        assert strip_wrapper_prefixes({"encoder.module.weight": 1}) == {"encoder.module.weight": 1}


class TestNestingIsNotCompiledTwice:
    """`named_modules()` yields a parent before its children.

    So compiling on that walk descends into the `OptimizedModule` it has just
    created and compiles the blocks inside it as well. The fixture above is
    single-level and could never see it; this one is the U-Net / transformer
    shape the module docstring is actually about.
    """

    def test_only_the_outermost_list_is_compiled(self):
        """Planted: measured 8 compilations where 2 were intended."""
        model = _Nested(stages=2, layers=3)
        assert compile_regions(model, dynamic=True) == 2

    def test_no_key_is_wrapped_twice(self):
        """`stages.0._orig_mod.layers.0._orig_mod.conv.weight` is a nested
        torch.compile: the outer graph traces an inner compiled module, and the
        cache reuse the whole idea rests on is gone."""
        model = _Nested(stages=2, layers=3)
        compile_regions(model, dynamic=True)
        doubled = [k for k in model.state_dict() if k.count("_orig_mod") > 1]
        assert doubled == [], doubled

    def test_the_inner_blocks_are_left_bare(self):
        model = _Nested(stages=2, layers=3)
        compile_regions(model, dynamic=True)
        inner = model.stages[0]._orig_mod.layers
        assert all(type(b).__name__ == "Block" for b in inner)

    def test_the_keys_still_load_into_a_bare_model(self):
        from spectramr.core.module_utils import strip_wrapper_prefixes

        model = _Nested(stages=2, layers=2)
        compile_regions(model, dynamic=True)
        bare = _Nested(stages=2, layers=2)
        bare.load_state_dict(strip_wrapper_prefixes(model.state_dict()), strict=True)
